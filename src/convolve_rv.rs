use crate::interpolate::{BinaryComponents, FluxFloat, WlGrid};
use anyhow::{anyhow, bail, Result};
use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use nalgebra as na;
use realfft::RealFftPlanner;
use std::iter::once;
use std::ops::{AddAssign, Index};

/// Trait for wavelength dispersion of the instrument. It provides the wavelength grid and
/// spectral resolution.
#[enum_dispatch]
pub trait WavelengthDispersion: Sync + Send + Clone {
    fn wavelength(&self) -> &na::DVector<f64>;
    fn float_indices_in_synth_grid(&self) -> &na::DVector<f64>;
    fn convolve(&self, flux: na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>>;
    fn convolve_segment(&self, flux: ArraySegment) -> Result<ArraySegment> {
        Ok(self.convolve(flux.as_array())?.into())
    }
}

#[derive(Clone)]
pub struct ArraySegment {
    array: na::DVector<FluxFloat>,
    start: usize,
    len: usize,
}

impl ArraySegment {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_view(&self) -> na::DVectorView<FluxFloat> {
        self.array.rows(self.start, self.len)
    }

    pub fn as_array(&self) -> na::DVector<FluxFloat> {
        self.as_view().into_owned()
    }
}

impl From<na::DVector<FluxFloat>> for ArraySegment {
    fn from(value: na::DVector<FluxFloat>) -> Self {
        Self {
            start: 0,
            len: value.len(),
            array: value,
        }
    }
}

impl From<Vec<FluxFloat>> for ArraySegment {
    fn from(value: Vec<FluxFloat>) -> Self {
        Self {
            start: 0,
            len: value.len(),
            array: na::DVector::from_vec(value),
        }
    }
}
impl Index<usize> for ArraySegment {
    type Output = FluxFloat;
    fn index(&self, index: usize) -> &Self::Output {
        self.array.index(index + self.start)
    }
}

/// Used when the models are already convolved with the instrument resolution.
/// The input flux is returned as is.
#[derive(Clone, Debug)]
pub struct NoConvolutionDispersionTarget {
    wavelength: na::DVector<f64>,
    float_indices_in_synth_grid: na::DVector<f64>,
}

impl NoConvolutionDispersionTarget {
    pub fn new(wavelength: na::DVector<f64>, modeltarget_wl: &WlGrid) -> Self {
        let float_indices_in_synth_grid =
            wavelength.map(|wl| modeltarget_wl.get_float_index_of_wl(wl));
        Self {
            wavelength,
            float_indices_in_synth_grid,
        }
    }
}

impl WavelengthDispersion for NoConvolutionDispersionTarget {
    fn wavelength(&self) -> &na::DVector<f64> {
        &self.wavelength
    }

    fn float_indices_in_synth_grid(&self) -> &na::DVector<f64> {
        &self.float_indices_in_synth_grid
    }

    /// Convolve the input flux with the instrument resolution dispersion kernel.
    fn convolve(&self, flux: na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
        Ok(flux)
    }

    fn convolve_segment(&self, flux: ArraySegment) -> Result<ArraySegment> {
        Ok(flux)
    }
}

fn make_gaussian(arr: &mut [FluxFloat], sigma: FluxFloat) {
    let n = arr.len();
    let mid = (n - 1) / 2;
    for i in 0..n {
        let x = (i as FluxFloat - mid as FluxFloat) / sigma;
        arr[i] = (-0.5 * x * x).exp();
    }
    let sum = arr.iter().sum::<FluxFloat>();
    for a in arr.iter_mut() {
        *a /= sum;
    }
}

/// Convolution by overlap-add method with Fast Fourier Transform.
pub fn oa_convolve(
    input_array: &na::DVector<FluxFloat>,
    kernel: &na::DVector<FluxFloat>,
) -> Result<na::DVector<FluxFloat>> {
    let n = FFTSIZE;
    let mut planner: RealFftPlanner<FluxFloat> = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method
    let m = kernel.len();
    let step_size = n - (m - 1);
    let l = input_array.len();
    let nx = l.next_multiple_of(step_size);

    let mut kernel_padded = pad_with_zeros(kernel.as_view(), n);
    let mut h = na::DVector::zeros(n / 2 + 1);
    fft.process(kernel_padded.as_mut_slice(), h.as_mut_slice())?;

    let mut y = na::DVector::zeros(nx + m - 1);
    for position in (0..nx).step_by(step_size) {
        let slice = if position + step_size < l {
            input_array.rows(position, step_size)
        } else {
            input_array.rows(position, l - position)
        };
        let mut x = pad_with_zeros(slice, n);
        let mut transformed = na::DVector::zeros(n / 2 + 1);
        fft.process(x.as_mut_slice(), transformed.as_mut_slice())?;
        transformed.component_mul_assign(&h);
        let mut re_transformed = na::DVector::zeros(n);
        ifft.process(transformed.as_mut_slice(), re_transformed.as_mut_slice())?;
        y.rows_mut(position, n)
            .add_assign(&re_transformed / n as FluxFloat);
    }
    Ok(y)
}

/// For spectra with constant spectral resolution.
#[derive(Clone, Debug)]
pub struct FixedTargetDispersionLogSpace {
    wl: na::DVector<f64>,
    kernel: na::DVector<FluxFloat>,
    n: usize,
    float_indices_in_synth_grid: na::DVector<f64>,
}

impl FixedTargetDispersionLogSpace {
    pub fn new(
        wavelength: na::DVector<f64>,
        resolution: f64,
        modeltarget_wl: &WlGrid,
    ) -> Result<Self> {
        if resolution <= 0.0 {
            return Err(anyhow!("Resolution must be positive"));
        }
        let log_step = match modeltarget_wl {
            WlGrid::Logspace(_, step, _) => step,
            _ => bail!("Use FixedTargetDispersionNonLog for non-logspaced grids"),
        };
        let sigma = 1.0 / (std::f64::consts::LN_10 * 2.355 * resolution * log_step);

        let n_maybe_even = (6.0 * sigma).ceil() as usize;
        let n = n_maybe_even + 1 - n_maybe_even % 2;
        let mut kernel = na::DVector::zeros(n);
        make_gaussian(kernel.as_mut_slice(), sigma as f32);
        let float_indices_in_synth_grid =
            wavelength.map(|wl| modeltarget_wl.get_float_index_of_wl(wl));
        Ok(Self {
            wl: wavelength,
            kernel,
            n: (n - 1) / 2,
            float_indices_in_synth_grid,
        })
    }
}

impl WavelengthDispersion for FixedTargetDispersionLogSpace {
    fn wavelength(&self) -> &na::DVector<f64> {
        &self.wl
    }

    fn float_indices_in_synth_grid(&self) -> &na::DVector<f64> {
        &self.float_indices_in_synth_grid
    }

    fn convolve(&self, flux_arr: na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
        let convolved = oa_convolve(&flux_arr, &self.kernel)?;
        Ok(convolved.rows(self.n, flux_arr.len()).into_owned())
    }
}

fn convolve_per_pixel(
    flux_arr: na::DVector<FluxFloat>,
    kernels: &na::DMatrix<FluxFloat>,
) -> Result<na::DVector<FluxFloat>> {
    if flux_arr.len() != kernels.ncols() {
        return Err(anyhow!(
            "flux_arr and wavelength grid don't match. flux_arr: {:?}, synth_wl: {:?}",
            flux_arr.len(),
            kernels.ncols()
        ));
    }
    let n = (kernels.nrows() - 1) / 2_usize;
    Ok(na::DVector::from_fn(flux_arr.len(), |i, _| {
        let kernel = kernels.column(i);
        if (i < n) || (i >= kernels.ncols() - n) {
            return flux_arr[i];
        }
        let fluxs = flux_arr.rows(i - n, 2 * n + 1);
        kernel.dot(&fluxs)
    }))
}

#[derive(Clone, Debug)]
pub struct FixedTargetDispersionNonLog {
    pub wl: na::DVector<f64>,
    pub modeltarget_wl: WlGrid,
    pub kernels: na::DMatrix<FluxFloat>,
    pub n: usize,
    pub float_indices_in_synth_grid: na::DVector<f64>,
}

impl FixedTargetDispersionNonLog {
    pub fn new(
        wavelength: na::DVector<f64>,
        resolution: f64,
        modeltarget_wl: WlGrid,
    ) -> Result<Self> {
        let sigmas: Vec<FluxFloat> = match &modeltarget_wl {
            WlGrid::Logspace(_, _, _) => {
                bail!("Use FixedTargetDispersionLogSpace for logspaced grids")
            }
            WlGrid::Linspace(_, step, _) => modeltarget_wl
                .iterate()
                .map(|wl| (wl / (resolution * step * 2.355)) as FluxFloat)
                .collect(),
            WlGrid::NonUniform(v) => once(2.0)
                .chain(v.iter().tuple_windows().map(|(left, middle, right)| {
                    let step = (right - left) / 2.0;
                    (middle / (resolution * step * 2.355)) as FluxFloat
                }))
                .chain(once(2.0))
                .collect(),
        };

        let largest_sigma = sigmas.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        // Kernel size must be made uneven
        let kernel_size_maybe_even = (largest_sigma * 6.0).ceil() as usize;
        let kernel_size = kernel_size_maybe_even + 1 - kernel_size_maybe_even % 2;

        let mut kernels = na::DMatrix::zeros(kernel_size, modeltarget_wl.n());
        for (mut kernel, sigma) in kernels.column_iter_mut().zip(sigmas.into_iter()) {
            make_gaussian(kernel.as_mut_slice(), sigma);
        }
        let float_indices_in_synth_grid =
            wavelength.map(|wl| modeltarget_wl.get_float_index_of_wl(wl));
        Ok(Self {
            wl: wavelength,
            modeltarget_wl,
            kernels,
            n: (kernel_size - 1) / 2_usize,
            float_indices_in_synth_grid,
        })
    }
}

impl WavelengthDispersion for FixedTargetDispersionNonLog {
    fn wavelength(&self) -> &nalgebra::DVector<f64> {
        &self.wl
    }

    fn float_indices_in_synth_grid(&self) -> &nalgebra::DVector<f64> {
        &self.float_indices_in_synth_grid
    }

    fn convolve(&self, flux: nalgebra::DVector<FluxFloat>) -> Result<nalgebra::DVector<FluxFloat>> {
        convolve_per_pixel(flux, &self.kernels)
    }
}

fn resample_observed_to_synth(
    wl_obs: &na::DVector<f64>,
    wl_synth: WlGrid,
    y: &na::DVector<FluxFloat>,
) -> Result<na::DVector<FluxFloat>> {
    if wl_obs.len() != y.len() {
        return Err(anyhow!(
            "Length of observed wavelength and flux are different. wl_obs: {:?}, y: {:?}",
            wl_obs.len(),
            y.len()
        ));
    }

    let mut j = 0;
    Ok(na::DVector::from_iterator(
        wl_synth.n(),
        wl_synth.iterate().map(|x| {
            while j < wl_obs.len() - 1 && wl_obs[j + 1] < x {
                j += 1;
            }
            if j == wl_obs.len() - 1 {
                y[j]
            } else {
                let weight = ((x - wl_obs[j]) / (wl_obs[j + 1] - wl_obs[j])) as FluxFloat;
                (1.0 - weight) * y[j] + weight * y[j + 1]
            }
        }),
    ))
}

/// Used for spectra where the spectral resolution is wavelength dependent.
/// Every pixel of the synthetic spectrum will be convolved with its own gaussian kernel
/// to simulate the effect of variable spectral resolution.
/// The kernels are precomputed and stored in a matrix.
#[derive(Clone, Debug)]
pub struct VariableTargetDispersion {
    pub wl: na::DVector<f64>,
    pub modeltarget_wl: WlGrid,
    pub kernels: na::DMatrix<FluxFloat>,
    pub n: usize,
    pub float_indices_in_synth_grid: na::DVector<f64>,
}

impl VariableTargetDispersion {
    /// Create a new VariableTargetDispersion object.
    /// dispersion: per-pixel spectral resolution in angstrom (FWHM)
    /// The observed wavelength and resolution array are resampled to the synthetic wavelength grid.
    pub fn new(
        wavelength: na::DVector<f64>,
        dispersion: &na::DVector<FluxFloat>,
        modeltarget_wl: WlGrid,
    ) -> Result<Self> {
        // Resample observed dispersion to synthetic wavelength grid
        let dispersion_resampled =
            resample_observed_to_synth(&wavelength, modeltarget_wl.clone(), dispersion)?;

        let sigmas: Vec<f32> = match &modeltarget_wl {
            WlGrid::Logspace(_, step, _) => modeltarget_wl
                .iterate()
                .zip(dispersion_resampled.iter())
                .map(|(wl, disp)| disp / (step * std::f64::consts::LN_10 * wl) as FluxFloat)
                .collect(),
            WlGrid::Linspace(_, step, _) => dispersion_resampled
                .iter()
                .map(|disp| disp / *step as FluxFloat)
                .collect(),
            WlGrid::NonUniform(v) => once(2.0)
                .chain(
                    v.iter()
                        .tuple_windows()
                        .zip(dispersion_resampled.iter().skip(1))
                        .map(|((left, _, right), disp)| {
                            let step = (right - left) / 2.0;
                            disp / step as FluxFloat
                        }),
                )
                .chain(once(2.0))
                .collect(),
        };

        let largest_sigma = sigmas.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        // Kernel size must be made uneven
        let kernel_size_maybe_even = (largest_sigma * 6.0).ceil() as usize;
        let kernel_size = kernel_size_maybe_even + 1 - kernel_size_maybe_even % 2;

        let mut kernels = na::DMatrix::zeros(kernel_size, modeltarget_wl.n());
        for (mut kernel, sigma) in kernels.column_iter_mut().zip(sigmas.into_iter()) {
            make_gaussian(kernel.as_mut_slice(), sigma);
        }
        let float_indices_in_synth_grid =
            wavelength.map(|wl| modeltarget_wl.get_float_index_of_wl(wl));
        Ok(Self {
            wl: wavelength,
            modeltarget_wl,
            kernels,
            n: (kernel_size - 1) / 2_usize,
            float_indices_in_synth_grid,
        })
    }
}

impl WavelengthDispersion for VariableTargetDispersion {
    fn wavelength(&self) -> &na::DVector<f64> {
        &self.wl
    }

    fn float_indices_in_synth_grid(&self) -> &na::DVector<f64> {
        &self.float_indices_in_synth_grid
    }

    fn convolve(&self, flux_arr: na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
        if flux_arr.len() != self.modeltarget_wl.n() {
            return Err(anyhow!(
                "flux_arr and wavelength grid don't match. flux_arr: {:?}, synth_wl: {:?}",
                flux_arr.len(),
                self.modeltarget_wl.n()
            ));
        }
        Ok(na::DVector::from_fn(flux_arr.len(), |i, _| {
            let kernel = self.kernels.column(i);
            if (i < self.n) || (i >= self.kernels.ncols() - self.n) {
                return flux_arr[i];
            }
            let fluxs = flux_arr.rows(i - self.n, 2 * self.n + 1);
            kernel.dot(&fluxs)
        }))
    }
}

const FFTSIZE: usize = 2048;

/// Build a kernel for rotational broadening.
/// vsini: projected rotational velocity in km/s.
/// dvelo: pixel velocity step in km/s.
pub fn build_rotation_kernel(vsini: f64, dvelo: f64) -> na::DVector<FluxFloat> {
    let epsilon = 0.6;
    let vrot = vsini / 299792.0;

    // Number of pixels required for the kernel
    let n_maybe_even = 1.max(
        (2.0 * vrot / dvelo
            * (1.0 + (4.0 * (epsilon - 1.0) / (std::f64::consts::PI * epsilon)).powf(2.0)).sqrt())
        .ceil() as usize
            + 2,
    );
    let n = n_maybe_even + 1 - n_maybe_even % 2;
    let mut velo_k: Vec<f64> = (0..n).map(|i| i as f64 * dvelo).collect();
    // Center around midpoint
    let mid = velo_k[n - 1] / 2.0;
    for v in &mut velo_k {
        *v -= mid;
    }

    let y: Vec<f64> = velo_k.iter().map(|&v| 1.0 - (v / vrot).powi(2)).collect();
    let mut kernel: Vec<FluxFloat> = y
        .iter()
        .map(|&y| {
            if y > 0.0 {
                ((2.0 * (1.0 - epsilon) * y.sqrt() + std::f64::consts::PI * epsilon / 2.0 * y)
                    / (std::f64::consts::PI * vrot * (1.0 - epsilon / 3.0)))
                    as FluxFloat
            } else {
                0.0
            }
        })
        .collect();
    let sum: FluxFloat = kernel.iter().sum();
    for k in &mut kernel {
        *k /= sum;
    }
    na::DVector::from_vec(kernel)
}

fn pad_with_zeros(arr: na::DVectorView<FluxFloat>, new_width: usize) -> na::DVector<FluxFloat> {
    let mut padded = na::DVector::zeros(new_width);
    padded.rows_mut(0, arr.len()).copy_from(&arr);
    padded
}

// Convolve the input spectrum with a rotational broadening kernel.
pub fn convolve_rotation(
    input_wl: &WlGrid,
    input_flux: &na::DVector<FluxFloat>,
    vsini: f64,
) -> Result<ArraySegment> {
    if vsini < 0.0 {
        return Err(anyhow!("vsini must be positive"));
    }
    if input_wl.n() != input_flux.len() {
        return Err(anyhow!(
            "Length of model and wavelength grid don't match. model: {:?}, synth_wl: {:?}",
            input_flux.len(),
            input_wl.n()
        ));
    }
    let dvelo = match input_wl {
        WlGrid::Linspace(first, step, total) => {
            // Use the middle wavelength as an approximation
            let avg_wl = (first + first + step * (*total as f64)) / 2.0;
            step / avg_wl
        }
        WlGrid::Logspace(_, step, _) => std::f64::consts::LN_10 * step,
        WlGrid::NonUniform(_) => bail!("Non-equidistant wavelength grid not supported"),
    };
    let kernel = build_rotation_kernel(vsini, dvelo);
    if vsini == 0.0 || kernel.len() <= 2 {
        return Ok(input_flux.clone().into());
    }
    if kernel.len() > FFTSIZE {
        return Err(anyhow!(
            "Kernel size is larger than FFTSIZE. Kernel size: {}, FFTSIZE: {}, vsini: {}",
            kernel.len(),
            FFTSIZE,
            vsini
        ));
    }
    let convolved = oa_convolve(input_flux, &kernel)?;
    Ok(ArraySegment {
        array: convolved,
        start: kernel.len() / 2,
        len: input_flux.len(),
    })
    // Ok(convolved
    //     .rows(kernel.len() / 2, input_flux.len())
    //     .into_owned())
}

/// Shift the input spectrum by rv and resample it to the target wavelength grid.
pub fn shift_and_resample(
    input_wl: &WlGrid,
    input_flux: &ArraySegment,
    target_dispersion: &impl WavelengthDispersion,
    rv: f64,
) -> Result<na::DVector<FluxFloat>> {
    let target_wl = target_dispersion.wavelength();
    if input_wl.n() != input_flux.len() {
        return Err(anyhow!(
                "Length of input_array and wavelength grid don't match. input_array: {:?}, synth_wl: {:?}",
                input_flux.len(),
                input_wl.n()
            ));
    }

    let shift_factor = 1.0 - rv / 299792.0;

    let (synth_first, synth_last) = input_wl.get_first_and_last();
    let first_source_wl = target_wl[0] * shift_factor;
    if first_source_wl < synth_first {
        return Err(anyhow!(
            "Observed wavelength is smaller than synthetic wavelength (RV={}, {:.1} < {:.1})",
            rv,
            first_source_wl,
            synth_first
        ));
    }
    let last_source_wl = target_wl[target_wl.len() - 1] * shift_factor;
    if last_source_wl > synth_last {
        return Err(anyhow!(
            "Observed wavelength is larger than synthetic wavelength (RV={}, {:.1} > {:.1})",
            rv,
            last_source_wl,
            synth_last
        ));
    }

    let resampler = |index: f64| {
        let j = index.floor() as usize;
        let weight = index as FluxFloat - j as FluxFloat;
        (1.0 - weight) * input_flux[j] + weight * input_flux[j + 1]
    };

    Ok(match input_wl {
        WlGrid::Linspace(start, step, _) => {
            let offset = (start / step) * (shift_factor - 1.0);
            na::DVector::from_iterator(
                target_wl.len(),
                target_dispersion
                    .float_indices_in_synth_grid()
                    .iter()
                    .map(|x| resampler(shift_factor * x + offset)),
            )
        }
        WlGrid::Logspace(_, step, _) => {
            let shift_log = shift_factor.log10() / step;
            na::DVector::from_iterator(
                target_wl.len(),
                target_dispersion
                    .float_indices_in_synth_grid()
                    .iter()
                    .map(|x| resampler(x + shift_log)),
            )
        }
        WlGrid::NonUniform(target_wl) => na::DVector::from_iterator(
            target_wl.len(),
            target_wl.iter().map(|wl| {
                let index = input_wl.get_float_index_of_wl(wl * shift_factor);
                resampler(index)
            }),
        ),
    })
}

pub fn shift_resample_and_add_binary_components(
    input_wl: &WlGrid,
    components: &BinaryComponents,
    target_dispersion: &impl WavelengthDispersion,
    rvs: [f64; 2],
) -> Result<na::DVector<FluxFloat>> {
    let BinaryComponents {
        norm_model1,
        norm_model2,
        continuum1,
        continuum2,
        lr: light_ratio,
    } = components;
    let [rv1, rv2] = rvs;
    let target_wl = target_dispersion.wavelength();

    if input_wl.n() != components.norm_model1.len() {
        return Err(anyhow!(
                "Length of input_array and wavelength grid don't match. input_array: {:?}, synth_wl: {:?}",
                components.norm_model1.len(),
                input_wl.n()
            ));
    }

    let shift_factor1 = 1.0 - rv1 / 299792.0;
    let shift_factor2 = 1.0 - rv2 / 299792.0;

    let (synth_first, synth_last) = input_wl.get_first_and_last();
    let first_source_wl = target_wl[0] * shift_factor1.min(shift_factor2);
    if first_source_wl < synth_first {
        return Err(anyhow!(
            "Observed wavelength is smaller than synthetic wavelength (RV={}, {:.1} < {:.1})",
            rv1.max(rv2),
            first_source_wl,
            synth_first
        ));
    }
    let last_source_wl = target_wl[target_wl.len() - 1] * shift_factor1.max(shift_factor2);
    if last_source_wl > synth_last {
        return Err(anyhow!(
            "Observed wavelength is larger than synthetic wavelength (RV={}, {:.1} > {:.1})",
            rv1.min(rv2),
            last_source_wl,
            synth_last
        ));
    }

    let resampler = |index1: f64, index2: f64| {
        let j = index1.floor() as usize;
        let weight = index1 as FluxFloat - j as FluxFloat;
        let n1 = (1.0 - weight) * norm_model1[j] + weight * norm_model1[j + 1];
        let c1 = (1.0 - weight) * continuum1[j] + weight * continuum1[j + 1];

        let j = index2.floor() as usize;
        let weight = index2 as FluxFloat - j as FluxFloat;
        let n2 = (1.0 - weight) * norm_model2[j] + weight * norm_model2[j + 1];
        let c2 = (1.0 - weight) * continuum2[j] + weight * continuum2[j + 1];

        // Weighted sum of fluxes divided by weighted sum of continua
        (n1 * c1 + light_ratio * n2 * c2) / (c1 + light_ratio * c2)
    };

    Ok(match input_wl {
        WlGrid::Linspace(start, step, _) => {
            let offset1 = (start / step) * (shift_factor1 - 1.0);
            let offset2 = (start / step) * (shift_factor2 - 1.0);
            na::DVector::from_iterator(
                target_wl.len(),
                target_dispersion
                    .float_indices_in_synth_grid()
                    .iter()
                    .map(|x| resampler(shift_factor1 * x + offset1, shift_factor2 * x + offset2)),
            )
        }
        WlGrid::Logspace(_, step, _) => {
            let shift_log1 = shift_factor1.log10() / step;
            let shift_log2 = shift_factor2.log10() / step;
            na::DVector::from_iterator(
                target_wl.len(),
                target_dispersion
                    .float_indices_in_synth_grid()
                    .iter()
                    .map(|x| resampler(x + shift_log1, x + shift_log2)),
            )
        }
        WlGrid::NonUniform(target_wl) => na::DVector::from_iterator(
            target_wl.len(),
            target_wl.iter().map(|wl| {
                let index1 = input_wl.get_float_index_of_wl(wl * shift_factor1);
                let index2 = input_wl.get_float_index_of_wl(wl * shift_factor2);
                resampler(index1, index2)
            }),
        ),
    })
}
