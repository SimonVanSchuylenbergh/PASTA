#![allow(non_snake_case, clippy::too_many_arguments, non_upper_case_globals)]
mod continuum_fitting;
mod convolve_rv;
mod cubic;
mod fitting;
mod interpolate;
mod model_fetchers;
mod particleswarm;

use anyhow::Result;
use continuum_fitting::{
    ChunkFitter, ConstantContinuum as RsConstantContinuum, ContinuumFitter,
    FixedContinuum as RsFixedContinuum, LinearModelFitter,
};
use convolve_rv::{
    shift_and_resample, FixedTargetDispersion, NoConvolutionDispersionTarget,
    VariableTargetDispersion, WavelengthDispersion,
};
use cubic::{calculate_interpolation_coefficients, calculate_interpolation_coefficients_flat};
use enum_dispatch::enum_dispatch;
use fitting::{BinaryFitter, ObservedSpectrum, SingleFitter};
use indicatif::ProgressBar;
use interpolate::{FluxFloat, GridBounds, GridInterpolator, Interpolator};
use model_fetchers::{read_npy_file, CachedFetcher, InMemFetcher, OnDiskFetcher};
use nalgebra as na;
use nalgebra::Storage;
use npy::to_file;
use numpy::array::PyArray;
use numpy::{Ix1, Ix2, PyArrayLike};
use pyo3::{prelude::*, pyclass};
use rayon::prelude::*;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Parameters to the PSO algorithm
#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct PSOSettings(fitting::PSOSettings);

#[pymethods]
impl PSOSettings {
    /// Create a new PSOSettings object.
    /// Inertia factor default: 0.7213475204.
    /// Cognitive factor default: 1.1931471806.
    /// Social factor default: 1.1931471806.
    /// Delta default: 1e-7 (forced move threshold).
    #[new]
    pub fn new(
        num_particles: usize,
        max_iters: u64,
        inertia_factor: Option<f64>,
        cognitive_factor: Option<f64>,
        social_factor: Option<f64>,
        delta: Option<f64>,
    ) -> Self {
        PSOSettings(fitting::PSOSettings {
            num_particles,
            max_iters,
            inertia_factor: inertia_factor.unwrap_or(0.7213475204),
            cognitive_factor: cognitive_factor.unwrap_or(1.1931471806),
            social_factor: social_factor.unwrap_or(1.1931471806),
            delta: delta.unwrap_or(1e-7),
        })
    }
}

/// Python to rust bindings.
impl From<PSOSettings> for fitting::PSOSettings {
    fn from(settings: PSOSettings) -> Self {
        settings.0
    }
}

#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "normalization", frozen)]
pub struct Label {
    #[pyo3(get)]
    teff: f64,
    #[pyo3(get)]
    m: f64,
    #[pyo3(get)]
    logg: f64,
    #[pyo3(get)]
    vsini: f64,
    #[pyo3(get)]
    rv: f64,
}

impl From<fitting::Label<f64>> for Label {
    fn from(value: fitting::Label<f64>) -> Self {
        Self {
            teff: value.teff,
            m: value.m,
            logg: value.logg,
            vsini: value.vsini,
            rv: value.rv,
        }
    }
}

impl Into<fitting::Label<f64>> for Label {
    fn into(self) -> fitting::Label<f64> {
        fitting::Label {
            teff: self.teff,
            m: self.m,
            logg: self.logg,
            vsini: self.vsini,
            rv: self.rv,
        }
    }
}

#[pymethods]
impl Label {
    fn as_list(&self) -> [f64; 5] {
        [self.teff, self.m, self.logg, self.vsini, self.rv]
    }
}

#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "normalization", frozen)]
pub struct LabelUncertainties {
    #[pyo3(get)]
    teff: (Option<f64>, Option<f64>),
    #[pyo3(get)]
    m: (Option<f64>, Option<f64>),
    #[pyo3(get)]
    logg: (Option<f64>, Option<f64>),
    #[pyo3(get)]
    vsini: (Option<f64>, Option<f64>),
    #[pyo3(get)]
    rv: (Option<f64>, Option<f64>),
}

impl From<fitting::Label<(Result<f64>, Result<f64>)>> for LabelUncertainties {
    fn from(value: fitting::Label<(Result<f64>, Result<f64>)>) -> Self {
        Self {
            teff: (value.teff.0.ok(), value.teff.1.ok()),
            m: (value.m.0.ok(), value.m.1.ok()),
            logg: (value.logg.0.ok(), value.logg.1.ok()),
            vsini: (value.vsini.0.ok(), value.vsini.1.ok()),
            rv: (value.rv.0.ok(), value.rv.1.ok()),
        }
    }
}

#[pymethods]
impl LabelUncertainties {
    fn to_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }
}

/// Output of the PSO fitting algorithm.
#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "normalization", frozen)]
pub struct OptimizationResult {
    /// Teff, [M/H], logg, vsini, RV
    #[pyo3(get)]
    pub label: Label,
    /// Fitted parameters for the continuum fitting function
    /// (polynomial coefficients)
    #[pyo3(get)]
    pub continuum_params: Vec<FluxFloat>,
    #[pyo3(get)]
    /// Chi2 value
    pub chi2: f64,
    #[pyo3(get)]
    /// Number of iterations used
    pub iterations: u64,
    /// Time taken
    #[pyo3(get)]
    pub time: f64,
}

/// Rust to Python bindings.
impl From<fitting::OptimizationResult> for OptimizationResult {
    fn from(result: fitting::OptimizationResult) -> Self {
        OptimizationResult {
            label: result.label.into(),
            continuum_params: result.continuum_params.data.into(),
            chi2: result.ls,
            iterations: result.iters,
            time: result.time,
        }
    }
}

#[pymethods]
impl OptimizationResult {
    fn to_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }
}

/// Output of the PSO binary fitting algorithm.
#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "normalization", frozen)]
pub struct BinaryOptimizationResult {
    /// (Teff, [M/H], logg, vsini, RV)
    #[pyo3(get)]
    pub label1: Label,
    #[pyo3(get)]
    pub label2: Label,
    #[pyo3(get)]
    pub light_ratio: f64,
    /// Fitted parameters for the continuum fitting function
    /// (polynomial coefficients)
    #[pyo3(get)]
    pub continuum_params: Vec<FluxFloat>,
    /// Chi2 value
    #[pyo3(get)]
    pub chi2: f64,
    /// Number of iterations used
    #[pyo3(get)]
    pub iterations: u64,
    /// Time taken
    #[pyo3(get)]
    pub time: f64,
}

/// Rust to Python bindings.
impl From<fitting::BinaryOptimizationResult> for BinaryOptimizationResult {
    fn from(result: fitting::BinaryOptimizationResult) -> Self {
        BinaryOptimizationResult {
            label1: result.label1.into(),
            label2: result.label2.into(),
            light_ratio: result.light_ratio,
            continuum_params: result.continuum_params.data.into(),
            chi2: result.ls,
            iterations: result.iters,
            time: result.time,
        }
    }
}

impl BinaryOptimizationResult {
    fn to_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }
}

#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct WlGrid(interpolate::WlGrid);

#[pymethods]
impl WlGrid {
    #[new]
    pub fn new(min: f64, step: f64, len: usize, log: Option<bool>) -> Self {
        if log.unwrap_or(false) {
            WlGrid(interpolate::WlGrid::Logspace(min, step, len))
        } else {
            WlGrid(interpolate::WlGrid::Linspace(min, step, len))
        }
    }

    pub fn get_array(&self) -> Vec<f64> {
        self.0.iterate().collect()
    }
}

#[enum_dispatch(WavelengthDispersion)]
#[derive(Clone, Debug)]
enum WavelengthDispersionWrapper {
    NoConvolutionDispersionTarget,
    FixedTargetDispersion,
    VariableTargetDispersion,
}

#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct PyWavelengthDispersion(WavelengthDispersionWrapper);

#[pymethods]
impl PyWavelengthDispersion {
    /// Convolve a spectrum with the dispersion kernel.
    pub fn convolve(&self, flux: Vec<FluxFloat>) -> Vec<FluxFloat> {
        self.0
            .convolve(na::DVector::from_vec(flux))
            .unwrap()
            .data
            .into()
    }

    /// Get the wavelength grid.
    pub fn wavelength(&self) -> Vec<f64> {
        self.0.wavelength().clone().data.into()
    }

    pub fn convolve_and_resample_directory(
        &self,
        input_directory: String,
        output_directory: String,
        includes_factor: bool,
        input_wavelength: WlGrid,
    ) {
        let out_dir = Path::new(output_directory.as_str());
        let files: Vec<PathBuf> = std::fs::read_dir(input_directory)
            .unwrap()
            .map(|x| x.unwrap().path())
            .collect();
        let bar = ProgressBar::new(files.len() as u64);
        files.into_par_iter().for_each(|file| {
            let out_file = out_dir.join(file.file_name().unwrap());
            if out_file.exists() {
                return;
            }
            let arr: na::DVector<u16> = read_npy_file(file).unwrap().into();
            if includes_factor {
                let bytes1 = arr[0].to_le_bytes();
                let bytes2 = arr[1].to_le_bytes();
                let factor = f32::from_le_bytes([bytes1[0], bytes1[1], bytes2[0], bytes2[1]]);

                let spectrum_float = arr
                    .rows(2, arr.len() - 2)
                    .map(|x| (x as FluxFloat) / 65535.0 * factor);
                let convolved = self.0.convolve(spectrum_float).unwrap();
                let resampled =
                    shift_and_resample(&convolved, &input_wavelength.0, self.0.wavelength(), 0.0)
                        .unwrap();

                let max = resampled.max();
                let max_bytes = max.to_le_bytes();
                let first_u16 = u16::from_le_bytes([max_bytes[0], max_bytes[1]]);
                let second_u16 = u16::from_le_bytes([max_bytes[2], max_bytes[3]]);

                let resampled_u16 = [first_u16, second_u16]
                    .into_iter()
                    .chain(
                        resampled
                            .into_iter()
                            .map(|x| (x.max(0.0) / max * 65535.0) as u16),
                    )
                    .collect::<Vec<_>>();
                to_file(out_file, resampled_u16).unwrap();
            } else {
                let spectrum_float = arr.map(|x| (x as FluxFloat) / 65535.0);
                let convolved = self.0.convolve(spectrum_float).unwrap();
                let resampled =
                    shift_and_resample(&convolved, &input_wavelength.0, self.0.wavelength(), 0.0)
                        .unwrap();
                let resampled_u16 = resampled
                    .into_iter()
                    .map(|x| (x.clamp(0.0, 1.0) * 65535.0) as u16)
                    .collect::<Vec<_>>();
                to_file(out_file, resampled_u16).unwrap();
            };

            bar.inc(1);
        });
    }
}

#[pyfunction]
fn VariableResolutionDispersion(
    wl: Vec<f64>,
    disp: Vec<FluxFloat>,
    synth_wl: WlGrid,
) -> PyWavelengthDispersion {
    PyWavelengthDispersion(
        VariableTargetDispersion::new(wl.into(), &disp.into(), synth_wl.0)
            .unwrap()
            .into(),
    )
}

#[pyfunction]
fn FixedResolutionDispersion(
    wl: Vec<f64>,
    resolution: f64,
    synth_wl: WlGrid,
) -> PyWavelengthDispersion {
    PyWavelengthDispersion(
        FixedTargetDispersion::new(wl.into(), resolution, synth_wl.0)
            .unwrap()
            .into(),
    )
}

#[pyfunction]
fn NoConvolutionDispersion(wl: Vec<f64>) -> PyWavelengthDispersion {
    PyWavelengthDispersion(NoConvolutionDispersionTarget(wl.into()).into())
}

#[enum_dispatch(ContinuumFitter)]
#[derive(Clone, Debug)]
enum ContinuumFitterWrapper {
    ChunkFitter,
    LinearModelFitter,
    RsFixedContinuum,
    RsConstantContinuum,
}

#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct PyContinuumFitter(ContinuumFitterWrapper);

#[pymethods]
impl PyContinuumFitter {
    fn fit_continuum(
        &self,
        synth: Vec<FluxFloat>,
        y: Vec<FluxFloat>,
        yerr: Vec<FluxFloat>,
    ) -> PyResult<(Vec<FluxFloat>, f64)> {
        let observed_spectrum = ObservedSpectrum::from_vecs(y, yerr);
        let (params, ls) = self
            .0
            .fit_continuum(&observed_spectrum, &na::DVector::from_vec(synth))
            .unwrap();
        Ok((params.data.into(), ls))
    }

    fn fit_and_return_continuum(
        &self,
        synth: Vec<FluxFloat>,
        y: Vec<FluxFloat>,
        yerr: Vec<FluxFloat>,
    ) -> PyResult<Vec<FluxFloat>> {
        let observed_spectrum = ObservedSpectrum::from_vecs(y, yerr);
        let result = self
            .0
            .fit_continuum_and_return_continuum(&observed_spectrum, &na::DVector::from_vec(synth))
            .unwrap();
        Ok(result.data.into())
    }

    fn fit_continuum_and_return_normalized_spec(
        &self,
        synth: Vec<FluxFloat>,
        y: Vec<FluxFloat>,
        yerr: Vec<FluxFloat>,
    ) -> PyResult<Vec<FluxFloat>> {
        let observed_spectrum = ObservedSpectrum::from_vecs(y, yerr);
        let continuum = self
            .0
            .fit_continuum_and_return_continuum(&observed_spectrum, &na::DVector::from_vec(synth))
            .unwrap();
        let result = observed_spectrum.flux.component_div(&continuum);
        Ok(result.data.into())
    }

    fn fit_and_return_chi2(
        &self,
        synth: Vec<FluxFloat>,
        y: Vec<FluxFloat>,
        yerr: Vec<FluxFloat>,
    ) -> PyResult<f64> {
        let observed_spectrum = ObservedSpectrum::from_vecs(y, yerr);
        let (_, chi2) = self
            .0
            .fit_continuum(&observed_spectrum, &na::DVector::from_vec(synth))
            .unwrap();
        Ok(chi2)
    }

    fn build_continuum(&self, params: Vec<FluxFloat>) -> PyResult<Vec<FluxFloat>> {
        Ok(self
            .0
            .build_continuum(&na::DVector::from_vec(params))
            .unwrap()
            .data
            .into())
    }
}

#[pyfunction]
fn ChunkContinuumFitter(
    wl: Vec<f64>,
    n_chunks: Option<usize>,
    p_order: Option<usize>,
    overlap: Option<f64>,
) -> PyContinuumFitter {
    PyContinuumFitter(
        ChunkFitter::new(
            wl.into(),
            n_chunks.unwrap_or(5),
            p_order.unwrap_or(8),
            overlap.unwrap_or(0.2),
        )
        .into(),
    )
}

#[pyfunction]
fn LinearModelContinuumFitter(design_matrix: PyArrayLike<FluxFloat, Ix2>) -> PyContinuumFitter {
    PyContinuumFitter(LinearModelFitter::new(design_matrix.as_matrix().into()).into())
}

#[pyfunction]
fn FixedContinuum(
    continuum: Vec<FluxFloat>,
    ignore_first_and_last: Option<usize>,
) -> PyContinuumFitter {
    PyContinuumFitter(
        RsFixedContinuum::new(continuum.into(), ignore_first_and_last.unwrap_or(0)).into(),
    )
}

#[pyfunction]
fn ConstantContinuum() -> PyContinuumFitter {
    PyContinuumFitter(continuum_fitting::ConstantContinuum().into())
}

/// Implement methods for all the interpolator classes.
/// We can't use traits with pyo3, so we have to use macros.
macro_rules! implement_methods {
    ($name: ident, $fetcher_type: ty, $PySingleFitter: ident, $PyBinaryFitter: ident) => {
        #[pymethods]
        impl $name {
            /// Produce a model spectrum by interpolating, convolving, shifting by rv and resampling to wl array.
            pub fn produce_model<'a>(
                &self,
                py: Python<'a>,
                dispersion: PyWavelengthDispersion,
                teff: f64,
                m: f64,
                logg: f64,
                vsini: f64,
                rv: f64,
            ) -> Bound<'a, PyArray<FluxFloat, Ix1>> {
                let interpolated = self
                    .0
                    .produce_model(&dispersion.0, teff, m, logg, vsini, rv)
                    .unwrap();
                // let interpolated = [0.0; LEN_C];
                PyArray::from_vec_bound(py, interpolated.iter().copied().collect())
            }

            pub fn produce_binary_model_norm<'a>(
                &self,
                py: Python<'a>,
                continuum_interpolator: &$name,
                dispersion: PyWavelengthDispersion,
                labels1: (f64, f64, f64, f64, f64),
                labels2: (f64, f64, f64, f64, f64),
                light_ratio: f64,
            ) -> Bound<'a, PyArray<FluxFloat, Ix1>> {
                let interpolated = self
                    .0
                    .produce_binary_model_norm(
                        &continuum_interpolator.0,
                        &dispersion.0,
                        &na::Vector5::new(labels1.0, labels1.1, labels1.2, labels1.3, labels1.4),
                        &na::Vector5::new(labels2.0, labels2.1, labels2.2, labels2.3, labels2.4),
                        light_ratio as f32,
                    )
                    .unwrap();
                // let interpolated = [0.0; LEN_C];
                PyArray::from_vec_bound(py, interpolated.iter().copied().collect())
            }

            /// Build the wavelength dispersion kernels (debugging purposes).
            pub fn get_kernels<'a>(
                &self,
                py: Python<'a>,
                wl: Vec<f64>,
                disp: Vec<FluxFloat>,
            ) -> Bound<'a, PyArray<FluxFloat, Ix2>> {
                let target_dispersion = VariableTargetDispersion::new(
                    wl.into(),
                    &disp.into(),
                    self.0.synth_wl().clone(),
                )
                .unwrap();
                let matrix = target_dispersion.kernels;
                let v: Vec<Vec<FluxFloat>> = matrix
                    .row_iter()
                    .map(|x| x.data.into_owned().into())
                    .collect();
                PyArray::from_vec2_bound(py, &v[..]).unwrap()
            }

            /// Produce multiple model spectra using multithreading.
            /// labels: Vec of (Teff, [M/H], logg, vsini, RV) tuples.
            pub fn produce_model_bulk<'a>(
                &self,
                py: Python<'a>,
                dispersion: PyWavelengthDispersion,
                labels: Vec<[f64; 5]>,
                progress: Option<bool>,
            ) -> Bound<'a, PyArray<FluxFloat, Ix2>> {
                let progress_bar = if progress.unwrap_or(false) {
                    ProgressBar::new(labels.len() as u64)
                } else {
                    ProgressBar::hidden()
                };
                let vec: Vec<Vec<FluxFloat>> = labels
                    .into_par_iter()
                    .map(|[teff, m, logg, vsini, rv]| {
                        progress_bar.inc(1);
                        self.0
                            .produce_model(&dispersion.0, teff, m, logg, vsini, rv)
                            .unwrap()
                            .iter()
                            .copied()
                            .collect()
                    })
                    .collect();
                PyArray::from_vec2_bound(py, &vec[..]).unwrap()
            }

            /// Produce a model directly from a grid model (without interpolating).
            /// Throws an error if (teff, m, logg) is not in the grid.
            pub fn produce_model_on_grid(
                &self,
                dispersion: PyWavelengthDispersion,
                teff: f64,
                mh: f64,
                logg: f64,
                vsini: f64,
                rv: f64,
            ) -> PyResult<Vec<FluxFloat>> {
                let out = self
                    .0
                    .produce_model_on_grid(&dispersion.0, teff, mh, logg, vsini, rv)
                    .unwrap();
                Ok(out.data.into())
            }

            /// Interpolate in grid.
            /// Doesn't do convoluton, shifting and resampling.
            pub fn interpolate(&self, teff: f64, m: f64, logg: f64) -> PyResult<Vec<FluxFloat>> {
                let interpolated = self.0.interpolate(teff, m, logg).unwrap();
                Ok(interpolated.iter().copied().collect())
            }

            pub fn get_fitter(
                &self,
                dispersion: PyWavelengthDispersion,
                continuum_fitter: PyContinuumFitter,
                settings: PSOSettings,
                vsini_range: (f64, f64),
                rv_range: (f64, f64),
            ) -> $PySingleFitter {
                $PySingleFitter(SingleFitter::new(
                    dispersion.0,
                    continuum_fitter.0,
                    settings.into(),
                    vsini_range,
                    rv_range,
                ))
            }

            /// Fit a continuum to an observed spectrum and model,
            /// as given by the labels (Teff, [M/H], logg, vsini, RV).
            /// Return the continuum.
            pub fn fit_continuum(
                &self,
                fitter: PyContinuumFitter,
                dispersion: PyWavelengthDispersion,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                label: (f64, f64, f64, f64, f64),
            ) -> PyResult<Vec<FluxFloat>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                let synth = self
                    .0
                    .produce_model(&dispersion.0, label.0, label.1, label.2, label.2, label.4)
                    .unwrap();
                Ok(fitter
                    .0
                    .fit_continuum_and_return_continuum(&observed_spectrum, &synth)
                    .unwrap()
                    .iter()
                    .copied()
                    .collect())
            }

            /// Fit the continuum for an observed spectrum and model,
            /// as given by the labels (Teff, [M/H], logg, vsini, RV).
            /// Return the parameters of the fit function.
            pub fn fit_continuum_and_return_model(
                &self,
                fitter: PyContinuumFitter,
                dispersion: PyWavelengthDispersion,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                label: [f64; 5],
            ) -> PyResult<Vec<FluxFloat>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                let synth = self
                    .0
                    .produce_model(
                        &dispersion.0,
                        label[0],
                        label[1],
                        label[2],
                        label[3],
                        label[4],
                    )
                    .unwrap();
                Ok(fitter
                    .0
                    .fit_continuum_and_return_fit(&observed_spectrum, &synth)
                    .unwrap()
                    .iter()
                    .copied()
                    .collect())
            }

            /// Compute the chi2 value at a given set of labels, continuum is fitted.
            pub fn chi2(
                &self,
                fitter: PyContinuumFitter,
                dispersion: PyWavelengthDispersion,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                labels: Vec<[f64; 5]>,
                allow_nan: Option<bool>,
            ) -> PyResult<Vec<f64>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                Ok(labels
                    .into_par_iter()
                    .map(|[teff, m, logg, vsini, rv]| {
                        let synth_model_result =
                            self.0
                                .produce_model(&dispersion.0, teff, m, logg, vsini, rv);
                        let synth_model = match allow_nan {
                            Some(true) => match synth_model_result {
                                Ok(x) => x,
                                Err(_) => return f64::NAN,
                            },
                            _ => synth_model_result.unwrap(),
                        };
                        let (_, chi2) = fitter
                            .0
                            .fit_continuum(&observed_spectrum, &synth_model)
                            .unwrap();
                        chi2
                        // 0.0
                    })
                    .collect())
            }

            /// Compute the chi2 value at a given set of labels for multiple spectra.
            /// Continuum is fitted.
            pub fn chi2_bulk(
                &self,
                observed_wavelength: Vec<Vec<f64>>,
                observed_flux: Vec<Vec<FluxFloat>>,
                observed_var: Vec<Vec<FluxFloat>>,
                labels: Vec<[f64; 5]>,
                progress: Option<bool>,
                allow_nan: Option<bool>,
            ) -> PyResult<Vec<f64>> {
                let progress_bar = if progress.unwrap_or(false) {
                    ProgressBar::new(labels.len() as u64)
                } else {
                    ProgressBar::hidden()
                };
                Ok(labels
                    .into_par_iter()
                    .zip(observed_wavelength)
                    .zip(observed_flux)
                    .zip(observed_var)
                    .map(|((([teff, m, logg, vsini, rv], wl), flux), var)| {
                        progress_bar.inc(1);
                        let obs = ObservedSpectrum::from_vecs(flux, var);
                        let target_dispersion = NoConvolutionDispersionTarget(wl.into());
                        let synth_model_result =
                            self.0
                                .produce_model(&target_dispersion, teff, m, logg, vsini, rv);
                        let synth_model = match allow_nan {
                            Some(true) => match synth_model_result {
                                Ok(x) => x,
                                Err(_) => return f64::NAN,
                            },
                            _ => synth_model_result.unwrap(),
                        };
                        // let synth_model = synth_model_result.unwrap();
                        let fitter = ChunkFitter::new(target_dispersion.0.clone(), 5, 8, 0.2);
                        let (_, chi2) = fitter.fit_continuum(&obs, &synth_model).unwrap();
                        chi2
                    })
                    .collect())
            }

            /// Clamp a single parameter to the bounds of the grid
            pub fn clamp_1d(&self, teff: f64, m: f64, logg: f64, index: usize) -> f64 {
                self.0
                    .grid_bounds()
                    .clamp_1d(na::Vector3::new(teff, m, logg), index)
                    .unwrap()
            }

            pub fn calculate_interpolation_coefficients_flat(
                &self,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> ([[f64; 4]; 4], [[f64; 4]; 4], [f64; 4]) {
                let local_grid = self.0.grid().get_local_grid(teff, m, logg).unwrap();
                let (x, y, z) = calculate_interpolation_coefficients_flat(&local_grid).unwrap();
                (x.into(), y.into(), z.into())
            }

            pub fn calculate_interpolation_coefficients(
                &self,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> Vec<FluxFloat> {
                let local_grid = self.0.grid().get_local_grid(teff, m, logg).unwrap();
                calculate_interpolation_coefficients(&local_grid)
                    .unwrap()
                    .data
                    .into()
            }

            pub fn get_neighbor_indices(
                &self,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> Vec<(usize, usize, usize)> {
                let local_grid = self.0.grid().get_local_grid(teff, m, logg).unwrap();
                local_grid
                    .teff_logg_indices
                    .iter()
                    .flat_map(|teff_logg_index| {
                        local_grid.m_indices.iter().map(move |m_index| {
                            match (teff_logg_index, m_index) {
                                (Some((i, j)), Some(k)) => (*i, *k, *j),
                                _ => (0, 0, 0),
                            }
                        })
                    })
                    .collect()
            }

            pub fn teffs(&self) -> Vec<f64> {
                self.0.grid().teff.values.clone()
            }

            pub fn ms(&self) -> Vec<f64> {
                self.0.grid().m.values.clone()
            }

            pub fn loggs(&self) -> Vec<f64> {
                self.0.grid().logg.values.clone()
            }

            pub fn logg_limits(&self) -> Vec<(usize, usize)> {
                self.0.grid().logg_limits.clone()
            }

            pub fn cumulative_grid_size(&self) -> Vec<usize> {
                self.0.grid().cumulative_grid_size.clone()
            }

            pub fn list_gridpoints(&self) -> Vec<[f64; 3]> {
                self.0.grid().list_gridpoints()
            }

            pub fn is_teff_logg_between_bounds(&self, teff: f64, logg: f64) -> bool {
                self.0.grid().is_teff_logg_between_bounds(teff, logg)
            }
        }

        #[pyclass]
        pub struct $PySingleFitter(
            SingleFitter<WavelengthDispersionWrapper, ContinuumFitterWrapper>,
        );

        #[pymethods]
        impl $PySingleFitter {
            /// Fit the model and pseudo continuum using PSO.
            /// Using chunk based continuum fitting.
            pub fn fit(
                &self,
                interpolator: &$name,
                observed_flux: PyArrayLike<FluxFloat, Ix1>,
                observed_var: PyArrayLike<FluxFloat, Ix1>,
                trace_directory: Option<String>,
                parallelize: Option<bool>,
            ) -> PyResult<OptimizationResult> {
                let observed_spectrum = ObservedSpectrum {
                    flux: observed_flux.as_matrix().column(0).into_owned(),
                    var: observed_var.as_matrix().column(0).into_owned(),
                };
                let result = self.0.fit(
                    &interpolator.0,
                    &observed_spectrum.into(),
                    trace_directory,
                    parallelize.unwrap_or(true),
                );
                Ok(result?.into())
            }

            pub fn fit_bulk(
                &self,
                interpolator: &$name,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
            ) -> PyResult<Vec<OptimizationResult>> {
                Ok(observed_fluxes
                    .into_par_iter()
                    .zip(observed_vars)
                    .map(|(flux, var)| {
                        let observed_spectrum = ObservedSpectrum::from_vecs(flux, var);
                        let result =
                            self.0
                                .fit(&interpolator.0, &observed_spectrum.into(), None, false);
                        result.unwrap().into()
                    })
                    .collect::<Vec<_>>())
            }

            /// Calculate uncertainties with the chi2 landscape method
            /// parameters: best fit
            /// search_radius: radius in which the intersection point is searched for every parameter
            pub fn compute_uncertainty(
                &self,
                interpolator: &$name,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                spec_res: f64,
                label: Label,
                search_radius: Option<[f64; 5]>,
            ) -> LabelUncertainties {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                self.0
                    .compute_uncertainty(
                        &interpolator.0,
                        &observed_spectrum,
                        spec_res,
                        label.into(),
                        fitting::Label::from_array(search_radius).into(),
                    )
                    .unwrap()
                    .into()
            }

            /// Uncertainties with the chi2 landscape method,
            /// for many spectra with multithreading
            pub fn compute_uncertainty_bulk(
                &self,
                interpolator: &$name,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                spec_res: f64,
                labels: Vec<Label>,
                search_radius: Option<[f64; 5]>,
            ) -> Vec<LabelUncertainties> {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                let bar = ProgressBar::new(observed_fluxes.len() as u64);
                observed_fluxes
                    .into_par_iter()
                    .zip(observed_vars)
                    .zip(labels)
                    .map(|((flux, var), params)| {
                        bar.inc(1);
                        let obs = ObservedSpectrum::from_vecs(flux, var);
                        self.0
                            .compute_uncertainty(
                                &interpolator.0,
                                &obs,
                                spec_res,
                                params.into(),
                                fitting::Label::from_array(search_radius).into(),
                            )
                            .unwrap()
                            .into()
                    })
                    .collect()
            }
        }

        #[pyclass]
        pub struct $PyBinaryFitter(
            BinaryFitter<WavelengthDispersionWrapper, ContinuumFitterWrapper>,
        );

        #[pymethods]
        impl $PyBinaryFitter {
            pub fn fit(
                &self,
                interpolator: &$name,
                continuum_interpolator: &$name,
                observed_flux: PyArrayLike<FluxFloat, Ix1>,
                observed_var: PyArrayLike<FluxFloat, Ix1>,
                trace_directory: Option<String>,
                parallelize: Option<bool>,
            ) -> PyResult<BinaryOptimizationResult> {
                let observed_spectrum = ObservedSpectrum {
                    flux: observed_flux.as_matrix().column(0).into_owned(),
                    var: observed_var.as_matrix().column(0).into_owned(),
                };
                let result = self.0.fit(
                    &interpolator.0,
                    &continuum_interpolator.0,
                    &observed_spectrum.into(),
                    trace_directory,
                    parallelize.unwrap_or(true),
                );
                Ok(result?.into())
            }
        }
    };
}

#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct PyInterpolator();

/// Interpolator that loads every spectrum from disk every time.
#[pyclass]
#[derive(Clone)]
pub struct OnDiskInterpolator(GridInterpolator<OnDiskFetcher>);

#[pymethods]
impl OnDiskInterpolator {
    #[new]
    pub fn new(dir: &str, includes_factor: bool, wavelength: WlGrid) -> Self {
        let fetcher = OnDiskFetcher::new(dir, includes_factor).unwrap();
        Self(GridInterpolator::new(fetcher, wavelength.0))
    }
}

/// Interpolator that loads every grid spectrum into memory in the beginning.
#[pyclass]
pub struct InMemInterpolator(GridInterpolator<InMemFetcher>);

#[pymethods]
impl InMemInterpolator {
    #[new]
    fn new(dir: &str, includes_factor: bool, wavelength: WlGrid) -> InMemInterpolator {
        let fetcher = InMemFetcher::new(&dir, includes_factor).unwrap();
        InMemInterpolator(GridInterpolator::new(fetcher, wavelength.0))
    }
}
/// Interpolator that caches the last _lrucap_ spectra
#[pyclass]
#[derive(Clone)]
pub struct CachedInterpolator(GridInterpolator<CachedFetcher>);

#[pymethods]
impl CachedInterpolator {
    #[new]
    pub fn new(
        dir: &str,
        includes_factor: bool,
        wavelength: WlGrid,
        lrucap: Option<usize>,
        n_shards: Option<usize>,
    ) -> Self {
        let fetcher = CachedFetcher::new(
            dir,
            includes_factor,
            lrucap.unwrap_or(4000),
            n_shards.unwrap_or(4),
        )
        .unwrap();
        Self(GridInterpolator::new(fetcher, wavelength.0))
    }
    pub fn cache_size(&self) -> usize {
        self.0.fetcher.cache_size()
    }
}

implement_methods!(
    OnDiskInterpolator,
    OnDiskFetcher,
    OnDiskSingleFitter,
    OnDiskBinaryFitter
);
implement_methods!(
    InMemInterpolator,
    InMemFetcher,
    InMemSingleFitter,
    InMemBinaryFitter
);
implement_methods!(
    CachedInterpolator,
    CachedFetcher,
    CachedSingleFitter,
    CachedBinaryFitter
);

#[pyfunction]
pub fn get_vsini_kernel(vsini: f64, synth_wl: WlGrid) -> Vec<FluxFloat> {
    let dvelo = match synth_wl.0 {
        interpolate::WlGrid::Linspace(_, _, _) => panic!("Only logspace is supported"),
        interpolate::WlGrid::Logspace(_, step, _) => std::f64::consts::LN_10 * step,
    };
    convolve_rv::build_rotation_kernel(vsini, dvelo).data.into()
}

#[pymodule]
fn pasta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OnDiskInterpolator>()?;
    m.add_class::<InMemInterpolator>()?;
    m.add_class::<CachedInterpolator>()?;
    m.add_class::<WlGrid>()?;
    m.add_class::<PSOSettings>()?;
    m.add_function(wrap_pyfunction!(NoConvolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(FixedResolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(VariableResolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(ChunkContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(FixedContinuum, m)?)?;
    m.add_function(wrap_pyfunction!(ConstantContinuum, m)?)?;
    m.add_function(wrap_pyfunction!(get_vsini_kernel, m)?)?;
    Ok(())
}
