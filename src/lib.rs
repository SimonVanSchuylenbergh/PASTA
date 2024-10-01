mod convolve_rv;
mod cubic;
mod fitting;
mod interpolate;
mod model_fetchers;
mod particleswarm;

use crate::particleswarm::PSOBounds;
use anyhow::Result;
use convolve_rv::{
    FixedTargetDispersion, NoConvolutionDispersionTarget, VariableTargetDispersion,
    WavelengthDispersion,
};
use enum_dispatch::enum_dispatch;
use fitting::{
    fit_pso, uncertainty_chi2, ChunkFitter, FixedContinuum, LinearModelFitter, ObservedSpectrum,
    OptimizationResult, PSOSettings as FittingPSOSettings,
};
use fitting::{ConstantContinuum, ContinuumFitter};
use indicatif::ProgressBar;
use interpolate::{FluxFloat, GridInterpolator, Interpolator, Range};
use model_fetchers::{CachedFetcher, InMemFetcher, OnDiskFetcher};
use nalgebra as na;
use nalgebra::Storage;
use numpy::array::PyArray;
use numpy::{Ix1, Ix2, PyArrayLike};
use pyo3::{prelude::*, pyclass};
use rayon::prelude::*;

/// Parameters to the PSO algorithm
#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct PSOSettings(FittingPSOSettings);

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
        PSOSettings(FittingPSOSettings {
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
impl From<PSOSettings> for FittingPSOSettings {
    fn from(settings: PSOSettings) -> Self {
        settings.0
    }
}

/// Output of the PSO fitting algorithm.
#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct PyOptimizationResult {
    /// (Teff, [M/H], logg, vsini, RV)
    #[pyo3(get)]
    pub labels: (f64, f64, f64, f64, f64),
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
impl From<OptimizationResult> for PyOptimizationResult {
    fn from(result: fitting::OptimizationResult) -> Self {
        PyOptimizationResult {
            labels: result.labels.into(),
            continuum_params: result.continuum_params.data.into(),
            chi2: result.ls,
            iterations: result.iters,
            time: result.time,
        }
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
    FixedContinuum,
    ConstantContinuum,
}

#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct PyContinuumFitter(ContinuumFitterWrapper);

#[pymethods]
impl PyContinuumFitter {
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
fn FixedContinuumFitter(
    continuum: Vec<FluxFloat>,
    ignore_first_and_last: Option<usize>,
) -> PyContinuumFitter {
    PyContinuumFitter(
        FixedContinuum::new(continuum.into(), ignore_first_and_last.unwrap_or(0)).into(),
    )
}

#[pyfunction]
fn ConstantContinuumFitter() -> PyContinuumFitter {
    PyContinuumFitter(ConstantContinuum().into())
}

/// Implement methods for all the interpolator classes.
/// We can't use traits with pyo3, so we have to use macros.
macro_rules! implement_methods {
    ($name: ident, $interpolator_type: ty) => {
        #[pymethods]
        impl $name {
            /// Produce a model spectrum by interpolating, convolving, shifting by rv and resampling to wl array.
            pub fn produce_model<'a>(
                &mut self,
                py: Python<'a>,
                dispersion: PyWavelengthDispersion,
                teff: f64,
                m: f64,
                logg: f64,
                vsini: f64,
                rv: f64,
            ) -> Bound<'a, PyArray<FluxFloat, Ix1>> {
                let interpolated = self
                    .interpolator
                    .produce_model(&dispersion.0, teff, m, logg, vsini, rv)
                    .unwrap();
                // let interpolated = [0.0; LEN_C];
                PyArray::from_vec_bound(py, interpolated.iter().copied().collect())
            }

            /// Build the wavelength dispersion kernels (debugging purposes).
            pub fn get_kernels<'a>(
                &mut self,
                py: Python<'a>,
                wl: Vec<f64>,
                disp: Vec<FluxFloat>,
            ) -> Bound<'a, PyArray<FluxFloat, Ix2>> {
                let target_dispersion = VariableTargetDispersion::new(
                    wl.into(),
                    &disp.into(),
                    self.interpolator.synth_wl().clone(),
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
                &mut self,
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
                        self.interpolator
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
                    .interpolator
                    .produce_model_on_grid(&dispersion.0, teff, mh, logg, vsini, rv)
                    .unwrap();
                Ok(out.data.into())
            }

            /// Interpolate in grid.
            /// Doesn't do convoluton, shifting and resampling.
            pub fn interpolate(
                &mut self,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> PyResult<Vec<FluxFloat>> {
                let interpolated = self.interpolator.interpolate(teff, m, logg).unwrap();
                Ok(interpolated.iter().copied().collect())
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
                    .interpolator
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
                    .interpolator
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
                        let synth_model_result = self.interpolator.produce_model(
                            &dispersion.0,
                            teff,
                            m,
                            logg,
                            vsini,
                            rv,
                        );
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
                        let synth_model_result = self.interpolator.produce_model(
                            &target_dispersion,
                            teff,
                            m,
                            logg,
                            vsini,
                            rv,
                        );
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

            /// Fit the model and pseudo continuum using PSO.
            /// Using chunk based continuum fitting.
            pub fn fit_pso(
                &self,
                fitter: PyContinuumFitter,
                dispersion: PyWavelengthDispersion,
                observed_flux: PyArrayLike<FluxFloat, Ix1>,
                observed_var: PyArrayLike<FluxFloat, Ix1>,
                settings: PSOSettings,
                save_directory: Option<String>,
                parallelize: Option<bool>,
            ) -> PyResult<PyOptimizationResult> {
                let observed_spectrum = ObservedSpectrum {
                    flux: observed_flux.as_matrix().column(0).into_owned(),
                    var: observed_var.as_matrix().column(0).into_owned(),
                };
                let result = fit_pso(
                    &self.interpolator,
                    &dispersion.0,
                    &observed_spectrum.into(),
                    &fitter.0,
                    &settings.into(),
                    save_directory,
                    parallelize.unwrap_or(true),
                );
                Ok(result?.into())
            }

            pub fn fit_pso_bulk(
                &self,
                fitters: Vec<PyContinuumFitter>,
                dispersions: Vec<PyWavelengthDispersion>,
                observed_fluxs: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                settings: PSOSettings,
            ) -> PyResult<Vec<PyOptimizationResult>> {
                let pso_settings: fitting::PSOSettings = settings.into();
                println!("{:?}", fitters.len());
                Ok(fitters
                    .into_par_iter()
                    .zip(dispersions)
                    .zip(observed_fluxs)
                    .zip(observed_vars)
                    .map(|(((fitter, disp), flux), var)| {
                        let observed_spectrum = ObservedSpectrum::from_vecs(flux, var);
                        let result = fit_pso(
                            &self.interpolator,
                            &disp.0,
                            &observed_spectrum.into(),
                            &fitter.0,
                            &pso_settings,
                            None,
                            false,
                        );
                        result.unwrap().into()
                    })
                    .collect::<Vec<_>>())
            }

            /// Calculate uncertainties with the chi2 landscape method
            /// parameters: best fit
            /// search_radius: radius in which the intersection point is searched for every parameter
            pub fn uncertainty_chi2(
                &self,
                fitter: PyContinuumFitter,
                dispersion: PyWavelengthDispersion,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                spec_res: f64,
                parameters: [f64; 5],
                search_radius: Option<[f64; 5]>,
            ) -> [Option<(f64, f64)>; 5] {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                uncertainty_chi2(
                    &self.interpolator,
                    &dispersion.0,
                    &observed_spectrum,
                    &fitter.0,
                    spec_res,
                    parameters.into(),
                    search_radius.into(),
                )
                .unwrap()
                .map(|x| x.ok())
            }

            /// Uncertainties with the chi2 landscape method,
            /// for many spectra with multithreading
            pub fn uncertainty_chi2_bulk(
                &self,
                fitters: Vec<PyContinuumFitter>,
                dispersions: Vec<PyWavelengthDispersion>,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                spec_res: f64,
                parameters: Vec<[f64; 5]>,
                search_radius: Option<[f64; 5]>,
            ) -> Vec<[Option<(f64, f64)>; 5]> {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                let bar = ProgressBar::new(fitters.len() as u64);
                fitters
                    .into_par_iter()
                    .zip(dispersions)
                    .zip(observed_fluxes)
                    .zip(observed_vars)
                    .zip(parameters)
                    .map(|((((fitter, disp), flux), var), params)| {
                        bar.inc(1);
                        let obs = ObservedSpectrum::from_vecs(flux, var);
                        uncertainty_chi2(
                            &self.interpolator,
                            &disp.0,
                            &obs,
                            &fitter.0,
                            spec_res,
                            params.into(),
                            search_radius.into(),
                        )
                        .unwrap()
                        .map(|x| x.ok())
                    })
                    .collect()
            }

            /// Uncertainties with the chi2 landscape method,
            /// for many spectra with multithreading
            /// where every spectrum is on the same wavelength grid.
            /// Only one `dispersion` object must be given.
            pub fn uncertainty_chi2_bulk_fixed_wl(
                &self,
                fitter: PyContinuumFitter,
                dispersion: PyWavelengthDispersion,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                spec_res: f64,
                parameters: Vec<[f64; 5]>,
                search_radius: Option<[f64; 5]>,
            ) -> Vec<[Option<(f64, f64)>; 5]> {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                observed_fluxes
                    .into_par_iter()
                    .zip(observed_vars)
                    .zip(parameters)
                    .map(|((flux, var), params)| {
                        let obs = ObservedSpectrum::from_vecs(flux, var);
                        uncertainty_chi2(
                            &self.interpolator,
                            &dispersion.0,
                            &obs,
                            &fitter.0,
                            spec_res,
                            params.into(),
                            search_radius.into(),
                        )
                        .unwrap()
                        .map(|x| x.ok())
                    })
                    .collect()
            }

            /// Clamp a single parameter to the bounds of the grid
            pub fn clamp_1d(
                &self,
                teff: f64,
                m: f64,
                logg: f64,
                vsini: f64,
                rv: f64,
                index: usize,
            ) -> f64 {
                self.interpolator
                    .bounds_single()
                    .clamp_1d(na::Vector5::new(teff, m, logg, vsini, rv), index)
                    .unwrap()
            }
        }
    };
}

/// Interpolator that loads every spectrum from disk every time.
#[pyclass]
#[derive(Clone)]
pub struct OnDiskInterpolator {
    interpolator: GridInterpolator<OnDiskFetcher>,
}

#[pymethods]
impl OnDiskInterpolator {
    #[new]
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
        vsini_range: Option<(f64, f64)>,
        rv_range: Option<(f64, f64)>,
    ) -> Self {
        let fetcher = OnDiskFetcher::new(
            dir,
            vsini_range.unwrap_or((1.0, 600.0)),
            rv_range.unwrap_or((-150.0, 150.0)),
        )
        .unwrap();
        Self {
            interpolator: GridInterpolator::new(fetcher, wavelength.0),
        }
    }
}

/// Interpolator that loads every grid spectrum into memory in the beginning.
#[pyclass]
#[derive(Clone)]
pub struct InMemInterpolator {
    dir: String,
    wavelength: WlGrid,
    vsini_range: Option<(f64, f64)>,
    rv_range: Option<(f64, f64)>,
}

/// Interpolator where all spectra have been loaded into memory.
#[pyclass]
pub struct LoadedInMemInterpolator {
    interpolator: GridInterpolator<InMemFetcher>,
}

#[pymethods]
impl InMemInterpolator {
    #[new]
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
        vsini_range: Option<(f64, f64)>,
        rv_range: Option<(f64, f64)>,
    ) -> Self {
        Self {
            dir: dir.to_string(),
            wavelength,
            vsini_range,
            rv_range,
        }
    }

    fn load(&self) -> LoadedInMemInterpolator {
        let fetcher = InMemFetcher::new(
            &self.dir,
            self.vsini_range.unwrap_or((1.0, 600.0)),
            self.rv_range.unwrap_or((-150.0, 150.0)),
        )
        .unwrap();
        LoadedInMemInterpolator {
            interpolator: GridInterpolator::new(fetcher, self.wavelength.0),
        }
    }
}
/// Interpolator that caches the last _lrucap_ spectra
#[pyclass]
#[derive(Clone)]
pub struct CachedInterpolator {
    interpolator: GridInterpolator<CachedFetcher>,
}

#[pymethods]
impl CachedInterpolator {
    #[new]
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
        vsini_range: Option<(f64, f64)>,
        rv_range: Option<(f64, f64)>,
        lrucap: Option<usize>,
    ) -> Self {
        let fetcher = CachedFetcher::new(
            dir,
            vsini_range.unwrap_or((1.0, 600.0)),
            rv_range.unwrap_or((-150.0, 150.0)),
            lrucap.unwrap_or(4000),
        )
        .unwrap();
        Self {
            interpolator: GridInterpolator::new(fetcher, wavelength.0),
        }
    }
    pub fn cache_size(&self) -> usize {
        self.interpolator.fetcher.cache_size()
    }
}

implement_methods!(OnDiskInterpolator, interpolators::OnDiskInterpolator);
implement_methods!(LoadedInMemInterpolator, interpolators::InMemInterpolator);
implement_methods!(CachedInterpolator, interpolators::CachedInterpolator);

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
    m.add_class::<CachedInterpolator>()?;
    m.add_class::<OnDiskInterpolator>()?;
    m.add_class::<InMemInterpolator>()?;
    m.add_class::<WlGrid>()?;
    m.add_class::<PSOSettings>()?;
    m.add_function(wrap_pyfunction!(NoConvolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(FixedResolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(VariableResolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(ChunkContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(FixedContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(ConstantContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(get_vsini_kernel, m)?)?;
    Ok(())
}
