use crate::interpolate::{FluxFloat, WlGrid};
use anyhow::{anyhow, Result};
use core::f32;
use enum_dispatch::enum_dispatch;
use nalgebra as na;
use realfft::RealFftPlanner;
use std::ops::AddAssign;

/// Trait for wavelength dispersion of the instrument. It provides the wavelength grid and
/// spectral resolution.
#[enum_dispatch]
pub trait WavelengthDispersion: Sync + Send {
    fn wavelength(&self) -> &na::DVector<f64>;
    fn convolve(&self, flux: na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>>;
}

#[derive(Clone, Debug)]
pub struct NoConvolutionDispersionTarget(pub na::DVector<f64>);

impl WavelengthDispersion for NoConvolutionDispersionTarget {
    fn wavelength(&self) -> &na::DVector<f64> {
        &self.0
    }
    fn convolve(&self, flux: na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
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

/// Convolution by overlap-add method with Fast Fourier transform.
pub fn oa_convolve(
    input_array: &na::DVector<FluxFloat>,
    kernel: &na::DVector<FluxFloat>,
) -> na::DVector<FluxFloat> {
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
    fft.process(kernel_padded.as_mut_slice(), h.as_mut_slice())
        .unwrap();

    let mut y = na::DVector::zeros(nx + m - 1);
    for position in (0..nx).step_by(step_size) {
        let slice = if position + step_size < l {
            input_array.rows(position, step_size)
        } else {
            input_array.rows(position, l - position)
        };
        let mut x = pad_with_zeros(slice, n);
        let mut transformed = na::DVector::zeros(n / 2 + 1);
        fft.process(x.as_mut_slice(), transformed.as_mut_slice())
            .unwrap();
        transformed.component_mul_assign(&h);
        let mut re_transformed = na::DVector::zeros(n);
        ifft.process(transformed.as_mut_slice(), re_transformed.as_mut_slice())
            .unwrap();
        y.rows_mut(position, n)
            .add_assign(&re_transformed / n as FluxFloat);
    }
    y
}

#[derive(Clone, Debug)]

pub struct FixedTargetDispersion {
    pub observed_wl: na::DVector<f64>,
    pub synth_wl: WlGrid,
    pub kernel: na::DVector<FluxFloat>,
    pub n: usize,
}

impl FixedTargetDispersion {
    pub fn new(wavelength: na::DVector<f64>, resolution: f64, synth_wl: WlGrid) -> Result<Self> {
        let sigma = match synth_wl {
            WlGrid::Linspace(_, _, _) => {
                return Err(anyhow!(
                "Fixed resolution convolution with linear wavelength dispersion is not supported"
            ))
            }
            WlGrid::Logspace(_, step, _) => 1.0 / (std::f64::consts::LN_10 * resolution * step),
        };
        let n_maybe_even = (6.0 * sigma).ceil() as usize;
        let n = n_maybe_even + 1 - n_maybe_even % 2;
        let mut kernel = na::DVector::zeros(n);
        make_gaussian(kernel.as_mut_slice(), sigma as f32);
        Ok(Self {
            observed_wl: wavelength,
            synth_wl,
            kernel,
            n: (n - 1) / 2,
        })
    }
}

impl WavelengthDispersion for FixedTargetDispersion {
    fn wavelength(&self) -> &na::DVector<f64> {
        &self.observed_wl
    }

    fn convolve(&self, flux_arr: na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
        if flux_arr.len() != self.synth_wl.n() {
            return Err(anyhow!(
                "flux_arr and wavelength grid don't match. flux_arr: {:?}, synth_wl: {:?}",
                flux_arr.len(),
                self.synth_wl.n()
            ));
        }
        let convolved = oa_convolve(&flux_arr, &self.kernel);
        Ok(convolved.rows(self.n, flux_arr.len()).into_owned())
    }
}

fn resample_observed_to_synth(
    wl_obs: &na::DVector<f64>,
    wl_synth: WlGrid,
    y: &na::DVector<FluxFloat>,
) -> Result<na::DVector<FluxFloat>> {
    if wl_obs.len() != y.len() {
        return Err(anyhow!(
            "Length of observed wavelength and flux are different"
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
    pub observed_wl: na::DVector<f64>,
    pub synth_wl: WlGrid,
    pub kernels: na::DMatrix<FluxFloat>,
    pub n: usize,
}

impl VariableTargetDispersion {
    /// Create a new VariableTargetDispersion object.
    /// dispersion: per-pixel spectral resolution in angstrom (1 sigma)
    /// The observed wavelength and resolution array are resampled to the synthetic wavelength grid.
    pub fn new(
        wavelength: na::DVector<f64>,
        dispersion: &na::DVector<FluxFloat>,
        synth_wl: WlGrid,
    ) -> Result<Self> {
        // Resample observed dispersion to synthetic wavelength grid
        let dispersion_resampled = resample_observed_to_synth(&wavelength, synth_wl, dispersion)?;

        let larges_disp = match synth_wl {
            WlGrid::Linspace(_, step, _) => {
                dispersion_resampled
                    .iter()
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap()
                    / step as FluxFloat
            }
            WlGrid::Logspace(_, step, _) => dispersion_resampled
                .iter()
                .zip(synth_wl.iterate())
                .map(|(disp, wl)| disp / (step * std::f64::consts::LN_10 * wl) as FluxFloat)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap(),
        };

        // Kernel size must be made uneven
        let kernel_size_maybe_even = (larges_disp * 6.0).ceil() as usize;
        let kernel_size = kernel_size_maybe_even + 1 - kernel_size_maybe_even % 2;

        let mut kernels = na::DMatrix::zeros(kernel_size, synth_wl.n());
        for (mut kernel, (wl, disp)) in kernels
            .column_iter_mut()
            .zip(synth_wl.iterate().zip(dispersion_resampled.iter()))
        {
            let sigma = match synth_wl {
                WlGrid::Linspace(_, step, _) => disp / step as FluxFloat,
                WlGrid::Logspace(_, step, _) => {
                    disp / (step * std::f64::consts::LN_10 * wl) as FluxFloat
                }
            };
            make_gaussian(kernel.as_mut_slice(), sigma);
        }
        Ok(Self {
            observed_wl: wavelength,
            synth_wl,
            kernels,
            n: (kernel_size - 1) / 2_usize,
        })
    }
}

impl WavelengthDispersion for VariableTargetDispersion {
    fn wavelength(&self) -> &na::DVector<f64> {
        &self.observed_wl
    }

    fn convolve(&self, flux_arr: na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
        if flux_arr.len() != self.synth_wl.n() {
            return Err(anyhow!(
                "flux_arr and wavelength grid don't match. flux_arr: {:?}, synth_wl: {:?}",
                flux_arr.len(),
                self.synth_wl.n()
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
/// vsini: projected rotational velocity in km/s
/// dvelo: pixel velocity step in km/s
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

pub fn convolve_rotation(
    input_array: &na::DVector<FluxFloat>,
    vsini: f64,
    synth_wl: WlGrid,
) -> Result<na::DVector<FluxFloat>> {
    if vsini < 1.0 {
        return Ok(input_array.clone());
    }
    match synth_wl {
        WlGrid::Linspace(_, _, _) => {
            return Err(anyhow!(
                "Rotational broadening with linear wavelength dispersion is not supported"
            ));
        }
        WlGrid::Logspace(_, step, _) => {
            let dvelo = std::f64::consts::LN_10 * step;
            let kernel = build_rotation_kernel(vsini, dvelo);
            let convolved = oa_convolve(input_array, &kernel);
            Ok(convolved
                .rows(kernel.len() / 2, input_array.len())
                .into_owned())
        }
    }
}

pub fn shift_and_resample(
    input_array: &na::DVector<FluxFloat>,
    synth_wl: WlGrid,
    observed_wl: &na::DVector<f64>,
    rv: f64,
) -> Result<na::DVector<FluxFloat>> {
    let shift_factor = 1.0 - rv / 299792.0;
    let mut output = na::DVector::zeros(observed_wl.len());

    let (synth_first, synth_last) = synth_wl.get_first_and_last();
    let first_source_wl = observed_wl[0] * shift_factor;
    if first_source_wl < synth_first {
        return Err(anyhow!(
            "Observed wavelength is smaller than synthetic wavelength (RV={}, {:.1} < {:.1})",
            rv,
            first_source_wl,
            synth_first
        ));
    }
    let last_source_wl = observed_wl[observed_wl.len() - 1] * shift_factor;
    if last_source_wl > synth_last {
        return Err(anyhow!(
            "Observed wavelength is larger than synthetic wavelength (RV={}, {:.1} > {:.1})",
            rv,
            last_source_wl,
            synth_last
        ));
    }

    for i in 0..observed_wl.len() {
        let obs_wl = observed_wl[i];
        let source_wl = obs_wl * shift_factor;
        let index = synth_wl.get_float_index_of_wl(source_wl);
        let j = index.floor() as usize;
        let weight = index as FluxFloat - j as FluxFloat;
        output[i] = (1.0 - weight) * input_array[j] + weight * input_array[j + 1];
    }
    Ok(output)
}

pub fn rot_broad_rv(
    input_array: na::DVector<FluxFloat>,
    synth_wl: WlGrid, 
    target_dispersion: &impl WavelengthDispersion,
    vsini: f64,
    rv: f64,
) -> Result<na::DVector<FluxFloat>> {
    if vsini < 0.0 {
        return Err(anyhow!("vsini must be positive"));
    }
    let observed_wl = target_dispersion.wavelength();
    let convolved_for_rotation = convolve_rotation(&input_array, vsini, synth_wl)?;
    let output = target_dispersion.convolve(convolved_for_rotation)?;
    shift_and_resample(&output, synth_wl, observed_wl, rv)
}
