use anyhow::{anyhow, Result};
use enum_dispatch::enum_dispatch;
use nalgebra as na;
use realfft::RealFftPlanner;
use std::ops::AddAssign;

use crate::interpolate::WlGrid;

#[enum_dispatch]
pub trait WavelengthDispersion: Sync + Send {
    fn wavelength(&self) -> &na::DVector<f64>;
    fn convolve(&self, flux: na::DVector<f64>) -> Result<na::DVector<f64>>;
}

#[derive(Clone, Debug)]
pub struct NoDispersionTarget(pub na::DVector<f64>);

impl WavelengthDispersion for NoDispersionTarget {
    fn wavelength(&self) -> &na::DVector<f64> {
        &self.0
    }
    fn convolve(&self, flux: na::DVector<f64>) -> Result<na::DVector<f64>> {
        Ok(flux)
    }
}

fn resample_observed_to_synth(
    wl_obs: &na::DVector<f64>,
    wl_synth: WlGrid,
    y: &na::DVector<f64>,
) -> Result<na::DVector<f64>> {
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
                let weight = (x - wl_obs[j]) / (wl_obs[j + 1] - wl_obs[j]);
                (1.0 - weight) * y[j] + weight * y[j + 1]
            }
        }),
    ))
}

fn make_gaussian(arr: &mut [f64], sigma: f64) {
    let n = arr.len();
    let mid = (n - 1) / 2;
    for i in 0..n {
        let x = (i as f64 - mid as f64) / sigma;
        arr[i] = (-0.5 * x * x).exp();
    }
    let sum = arr.iter().sum::<f64>();
    for a in arr.iter_mut() {
        *a /= sum;
    }
}

#[derive(Clone, Debug)]
pub struct VariableTargetDispersion {
    observed_wl: na::DVector<f64>,
    synth_wl: WlGrid,
    kernels: na::DMatrix<f64>,
    n: usize,
}

impl VariableTargetDispersion {
    pub fn new(
        wavelength: na::DVector<f64>,
        dispersion: &na::DVector<f64>,
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
                    / step
            }
            WlGrid::Logspace(_, step, _) => dispersion_resampled
                .iter()
                .zip(synth_wl.iterate())
                .map(|(disp, wl)| disp / (step * std::f64::consts::LN_10 * wl))
                .max_by(|a, b| a.total_cmp(b))
                .unwrap(),
        };

        let kernel_size_maybe_even = (larges_disp * 6.0).ceil() as usize;
        let kernel_size = if kernel_size_maybe_even % 2 == 0 {
            kernel_size_maybe_even + 1
        } else {
            kernel_size_maybe_even
        };

        let mut kernels = na::DMatrix::zeros(kernel_size, synth_wl.n());
        for (mut kernel, (wl, disp)) in kernels
            .column_iter_mut()
            .zip(synth_wl.iterate().zip(dispersion_resampled.iter()))
        {
            let sigma = match synth_wl {
                WlGrid::Linspace(_, step, _) => disp / step,
                WlGrid::Logspace(_, step, _) => disp / (step * std::f64::consts::LN_10 * wl),
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

    pub fn get_kernels(&self) -> &na::DMatrix<f64> {
        &self.kernels
    }
}

impl WavelengthDispersion for VariableTargetDispersion {
    fn wavelength(&self) -> &na::DVector<f64> {
        &self.observed_wl
    }

    fn convolve(&self, flux_arr: na::DVector<f64>) -> Result<na::DVector<f64>> {
        if flux_arr.len() != self.synth_wl.n() {
            return Err(anyhow!(
                "Trying to convolve spectrum with different length than synthetic wavelength grid"
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

fn build_kernel(vsini: f64, synth_wl: WlGrid) -> Vec<f64> {
    let dvelo = match synth_wl {
        WlGrid::Linspace(first, step, N) => step / (first + 0.5 * step * (N as f64)),
        WlGrid::Logspace(_, step, _) => std::f64::consts::LN_10 * step,
    };
    let epsilon = 0.6;
    let vrot = vsini / 299792.0;

    let n = 1.max((2.0 * vrot / dvelo).round() as usize);
    let mut velo_k: Vec<f64> = (0..n).map(|i| i as f64 * dvelo).collect();
    let mid = velo_k[n - 1] / 2.0;
    for v in &mut velo_k {
        *v -= mid;
    }

    let y: Vec<f64> = velo_k.iter().map(|&v| 1.0 - (v / vrot).powi(2)).collect();
    let mut kernel: Vec<f64> = y
        .iter()
        .map(|&y| {
            if y > 0.0 {
                (2.0 * (1.0 - epsilon) * y.sqrt() + std::f64::consts::PI * epsilon / 2.0 * y)
                    / (std::f64::consts::PI * vrot * (1.0 - epsilon / 3.0))
            } else {
                0.0
            }
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= sum;
    }
    kernel
}

fn pad_with_zeros(arr: na::DVectorView<f64>, new_width: usize) -> na::DVector<f64> {
    let mut padded = na::DVector::zeros(new_width);
    padded.rows_mut(0, arr.len()).copy_from(&arr);
    padded
}

pub fn oaconvolve(
    input_array: &na::DVector<f64>,
    vsini: f64,
    synth_wl: WlGrid,
) -> (na::DVector<f64>, usize) {
    if vsini < 1.0 {
        let vec = input_array.into_iter().map(f64::to_owned).collect();
        return (na::DVector::from_vec(vec), 0);
    }

    let n = FFTSIZE;
    let mut planner: RealFftPlanner<f64> = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method
    let kernel = na::DVector::from_vec(build_kernel(vsini, synth_wl));
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
            // dbg!(position);
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
            .add_assign(&re_transformed / n as f64);
    }

    (y, m)
}

pub fn ccf(
    input_wl: &na::DVector<f64>,
    input_spectrum_inverted: &na::DVector<f64>,
    model_spectrum_inverted: &na::DVector<f64>,
    kernel_len: usize,
    synth_wl: WlGrid,
    rvs: &Vec<f64>,
) -> Result<Vec<f64>> {
    let n = input_wl.len();

    rvs.iter()
        .map(|rv| {
            let resampled =
                shift_and_resample(model_spectrum_inverted, kernel_len, synth_wl, input_wl, *rv)?;
            Ok(input_spectrum_inverted.dot(&resampled) / (n as f64))
        })
        .collect()
}

pub fn shift_and_resample(
    input_array: &na::DVector<f64>,
    kernel_len: usize,
    synth_wl: WlGrid,
    observed_wl: &na::DVector<f64>,
    rv: f64,
) -> Result<na::DVector<f64>> {
    let start = kernel_len / 2;
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
        let weight = index - (j as f64);
        output[i] = (1.0 - weight) * input_array[j + start] + weight * input_array[j + start + 1];
    }
    Ok(output)
}

pub fn rot_broad_rv(
    input_array: na::DVector<f64>,
    synth_wl: WlGrid,
    target_dispersion: &impl WavelengthDispersion,
    vsini: f64,
    rv: f64,
) -> Result<na::DVector<f64>> {
    let observed_wl = target_dispersion.wavelength();
    let convolved_for_resolution = target_dispersion.convolve(input_array)?;
    let (output, kernel_len) = oaconvolve(&convolved_for_resolution, vsini, synth_wl);
    shift_and_resample(&output, kernel_len, synth_wl, observed_wl, rv)
}
