mod convolve_rv;
mod cubic;
mod fitting;
mod interpolate;
mod model_fetchers;
mod particleswarm;
use anyhow::Result;
use convolve_rv::{NoDispersionTarget, VariableTargetDispersion, WavelengthDispersion};
use enum_dispatch::enum_dispatch;
use fitting::{
    fit_pso, uncertainty_chi2, ChunkFitter, FixedContinuum, LinearModelFitter, ObservedSpectrum,
    OptimizationResult, PSOSettings as FittingPSOSettings,
};
use fitting::{ConstantContinuum, ContinuumFitter};
use indicatif::ProgressBar;
use interpolate::{
    Bounds, CompoundInterpolator, FluxFloat, Interpolator, Range, SquareGridInterpolator,
};
use model_fetchers::{CachedFetcher, InMemFetcher, OnDiskFetcher};
use nalgebra as na;
use nalgebra::Storage;
use numpy::array::PyArray;
use numpy::{Ix1, Ix2, PyArrayLike};
use pyo3::{prelude::*, pyclass};
use rayon::prelude::*;

/// Fit polynomial to data.
fn polyfit<const N: usize>(
    x_values: &na::SVector<f64, N>,
    b: &na::SVector<f64, N>,
    polynomial_degree: usize,
) -> na::DVector<f64> {
    let number_of_columns = polynomial_degree + 1;
    let number_of_rows = x_values.len();
    let mut a = na::DMatrix::zeros(number_of_rows, number_of_columns);

    for (row, &x) in x_values.iter().enumerate() {
        // First column is always 1
        a[(row, 0)] = 1.0;

        for col in 1..number_of_columns {
            a[(row, col)] = x.powf(col as f64);
        }
    }

    let decomp = na::SVD::new(a, true, true);
    decomp.solve(b, na::convert(1e-18f64)).unwrap()
}

/// Filter a spectrum and keep only the pixels with wavelengths within at least one of the ranges.
/// Ranges are given as a vec of (start, end) tuples (inclusive).
fn cut_spectrum_multi_ranges(
    wl: &na::DVector<f64>,
    flux: &na::DVector<f64>,
    wl_ranges: Vec<(f64, f64)>,
) -> (na::DVector<f64>, na::DVector<f64>) {
    let mut indices = Vec::new();
    for (i, &x) in wl.iter().enumerate() {
        for wl_range in wl_ranges.iter() {
            if x >= wl_range.0 && x <= wl_range.1 {
                indices.push(i);
                break;
            }
        }
    }
    let mut new_wl = na::DVector::zeros(indices.len());
    let mut new_flux = na::DVector::zeros(indices.len());
    for (i, &index) in indices.iter().enumerate() {
        new_wl[i] = wl[index];
        new_flux[i] = flux[index];
    }
    (new_wl, new_flux)
}

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

/// Output from the PSO fitting.
#[derive(Clone, Debug)]
#[pyclass(module = "normalization", frozen)]
pub struct PyOptimizationResult {
    /// (Teff, [M/H], logg, vsini, RV)
    #[pyo3(get)]
    pub labels: (f64, f64, f64, f64, f64),
    /// Fitted parameters for the continuum fitting function
    #[pyo3(get)]
    pub continuum_params: Vec<FluxFloat>,
    #[pyo3(get)]
    /// Chi2 value
    pub ls: f64,
    #[pyo3(get)]
    /// Number of iterations used
    pub iters: u64,
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
            ls: result.ls,
            iters: result.iters,
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
}

#[enum_dispatch(WavelengthDispersion)]
#[derive(Clone, Debug)]
enum WavelengthDispersionWrapper {
    NoDispersionTarget,
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
fn FixedResolutionDispersion(wl: Vec<f64>) -> PyWavelengthDispersion {
    PyWavelengthDispersion(NoDispersionTarget(wl.into()).into())
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
fn FixedContinuumFitter(continuum: Vec<FluxFloat>) -> PyContinuumFitter {
    PyContinuumFitter(FixedContinuum::new(continuum.into()).into())
}

#[pyfunction]
fn ConstantContinuumFitter() -> PyContinuumFitter {
    PyContinuumFitter(ConstantContinuum().into())
}

/// Build LinearModelFitter object from design matrix
fn convert_dm(design_matrix: Vec<FluxFloat>, len: usize) -> LinearModelFitter {
    let n_params = design_matrix.len() / len;
    LinearModelFitter::new(na::DMatrix::<FluxFloat>::from_vec(
        len,
        n_params,
        design_matrix,
    ))
}

macro_rules! implement_methods {
    ($name: ident, $interpolator_type: ty) => {
        #[pymethods]
        impl $name {
            /// Produce a model by interpolating, convolving, shifting by rv and resampling to wl array.
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

            /// Produce multiple models with multithreading.
            /// labels: Vec of (Teff, [M/H], logg, vsini, RV) tuples.
            pub fn produce_model_bulk<'a>(
                &mut self,
                py: Python<'a>,
                observed_wavelength: Vec<f64>,
                labels: Vec<(f64, f64, f64, f64, f64)>,
                progress: Option<bool>,
            ) -> Bound<'a, PyArray<FluxFloat, Ix2>> {
                let target_dispersion = NoDispersionTarget(observed_wavelength.into());
                let progress_bar = if progress.unwrap_or(false) {
                    ProgressBar::new(labels.len() as u64)
                } else {
                    ProgressBar::hidden()
                };
                let vec: Vec<Vec<FluxFloat>> = labels
                    .into_par_iter()
                    .map(|(teff, m, logg, vsini, rv)| {
                        progress_bar.inc(1);
                        self.interpolator
                            .produce_model(&target_dispersion, teff, m, logg, vsini, rv)
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
            ) -> PyResult<Vec<f64>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                Ok(labels
                    .into_par_iter()
                    .map(|[teff, m, logg, vsini, rv]| {
                        let synth_model = self
                            .interpolator
                            .produce_model(&dispersion.0, teff, m, logg, vsini, rv)
                            .unwrap();
                        let (_, chi2) = fitter
                            .0
                            .fit_continuum(&observed_spectrum, &synth_model)
                            .unwrap();
                        chi2
                    })
                    .collect())
            }

            /// Compute the chi2 value at a given set of labels, with a fixed continuum.
            pub fn chi2_fixed_continuum(
                &self,
                dispersion: PyWavelengthDispersion,
                continuum: Vec<FluxFloat>,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                labels: Vec<[f64; 5]>,
            ) -> PyResult<Vec<f64>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                let fitter = FixedContinuum::new(continuum.into());
                Ok(labels
                    .into_par_iter()
                    .map(|[teff, m, logg, vsini, rv]| {
                        let synth_model = self
                            .interpolator
                            .produce_model(&dispersion.0, teff, m, logg, vsini, rv)
                            .unwrap();
                        let (_, chi2) = fitter
                            .fit_continuum(&observed_spectrum, &synth_model)
                            .unwrap();
                        chi2
                    })
                    .collect())
            }

            /// Compute the chi2 value at a given set of labels with multithreading.
            /// Continuum is fitted.
            pub fn chi2_bulk(
                &self,
                observed_wavelength: Vec<Vec<f64>>,
                observed_flux: Vec<Vec<FluxFloat>>,
                observed_var: Vec<Vec<FluxFloat>>,
                labels: Vec<[f64; 5]>,
                progress: Option<bool>,
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
                        let target_dispersion = NoDispersionTarget(wl.into());
                        let synth_model = self
                            .interpolator
                            .produce_model(&target_dispersion, teff, m, logg, vsini, rv)
                            .unwrap();
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
                Ok(result.unwrap().into())
                // Ok(PyOptimizationResult {
                //     labels: (0.0, 0.0, 0.0, 0.0, 0.0),
                //     continuum_params: vec![0.0],
                //     ls: 0.0,
                //     iters: 0,
                //     time: 0.0,
                // })
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
                observed_wavelengths: Vec<Vec<f64>>,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                parameters: Vec<[f64; 5]>,
                search_radius: Option<[f64; 5]>,
            ) -> Vec<[Option<(f64, f64)>; 5]> {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                let bar = ProgressBar::new(observed_wavelengths.len() as u64);
                observed_wavelengths
                    .into_par_iter()
                    .zip(observed_fluxes)
                    .zip(observed_vars)
                    .zip(parameters)
                    .map(|(((wl, flux), var), params)| {
                        bar.inc(1);
                        let target_dispersion = NoDispersionTarget(wl.into());
                        let obs = ObservedSpectrum::from_vecs(flux, var);
                        let fitter = ChunkFitter::new(target_dispersion.0.clone(), 5, 8, 0.2);
                        uncertainty_chi2(
                            &self.interpolator,
                            &target_dispersion,
                            &obs,
                            &fitter,
                            params.into(),
                            search_radius.into(),
                        )
                        .unwrap()
                        .map(|x| x.ok())
                    })
                    .collect()
            }

            /// Uncertainties with the chi2 landscape method,
            /// on multiple spectra using multithreading
            pub fn uncertainty_chi2_fixed_cont_bulk(
                &self,
                observed_wavelengths: Vec<Vec<f64>>,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                parameters: Vec<[f64; 5]>,
                search_radius: Option<[f64; 5]>,
            ) -> Vec<[Option<(f64, f64)>; 5]> {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                let bar = ProgressBar::new(observed_wavelengths.len() as u64);
                observed_wavelengths
                    .into_par_iter()
                    .zip(observed_fluxes)
                    .zip(observed_vars)
                    .zip(parameters)
                    .map(|(((wl, flux), var), params)| {
                        bar.inc(1);
                        let target_dispersion = NoDispersionTarget(wl.into());
                        let obs = ObservedSpectrum::from_vecs(flux, var);
                        let cont_fitter = ChunkFitter::new(target_dispersion.0.clone(), 5, 8, 0.2);
                        let synth = self
                            .interpolator
                            .produce_model(
                                &target_dispersion,
                                params[0],
                                params[1],
                                params[2],
                                params[3],
                                params[4],
                            )
                            .unwrap();
                        let continuum = cont_fitter
                            .fit_continuum_and_return_continuum(&obs, &synth)
                            .unwrap();
                        let fixed_fitter = FixedContinuum::new(continuum);
                        uncertainty_chi2(
                            &self.interpolator,
                            &target_dispersion,
                            &obs,
                            &fixed_fitter,
                            params.into(),
                            search_radius.into(),
                        )
                        .unwrap()
                        .map(|x| x.ok())
                    })
                    .collect()
            }

            /// Check whether a set of labels is within the bounds of the grid
            pub fn is_within_bounds(
                &self,
                teff: f64,
                m: f64,
                logg: f64,
                vsini: f64,
                rv: f64,
            ) -> bool {
                self.interpolator
                    .bounds()
                    .is_within_bounds(na::Vector5::new(teff, m, logg, vsini, rv))
            }

            /// Clamp a set of labels to the bounds of the grid
            pub fn clamp(&self, teff: f64, m: f64, logg: f64, vsini: f64, rv: f64) -> [f64; 5] {
                self.interpolator
                    .bounds()
                    .clamp(na::Vector5::new(teff, m, logg, vsini, rv))
                    .into()
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
                    .bounds()
                    .clamp_1d(na::Vector5::new(teff, m, logg, vsini, rv), index)
            }
        }
    };
}

/// Interpolator that loads every spectrum from disk every time.
#[pyclass]
#[derive(Clone)]
pub struct OnDiskInterpolator {
    interpolator: SquareGridInterpolator<OnDiskFetcher>,
}

#[pymethods]
impl OnDiskInterpolator {
    #[new]
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
        teff_range: Vec<f64>,
        m_range: Vec<f64>,
        logg_range: Vec<f64>,
        vsini_range: Option<(f64, f64)>,
        rv_range: Option<(f64, f64)>,
    ) -> Self {
        let fetcher = OnDiskFetcher::new(
            dir,
            Range::new(teff_range),
            Range::new(m_range),
            Range::new(logg_range),
            vsini_range.unwrap_or((1.0, 600.0)),
            rv_range.unwrap_or((-150.0, 150.0)),
        );
        Self {
            interpolator: SquareGridInterpolator::new(fetcher, wavelength.0),
        }
    }
}

/// Interpolator that loads every grid spectrum into memory in the beginning.
#[pyclass]
#[derive(Clone)]
pub struct InMemInterpolator {
    dir: String,
    wavelength: WlGrid,
    teff_range: Vec<f64>,
    m_range: Vec<f64>,
    logg_range: Vec<f64>,
    vsini_range: Option<(f64, f64)>,
    rv_range: Option<(f64, f64)>,
}

/// Interpolator where all spectra have been loaded into memory.
#[pyclass]
pub struct LoadedInMemInterpolator {
    interpolator: SquareGridInterpolator<InMemFetcher>,
}

#[pymethods]
impl InMemInterpolator {
    #[new]
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
        teff_range: Vec<f64>,
        m_range: Vec<f64>,
        logg_range: Vec<f64>,
        vsini_range: Option<(f64, f64)>,
        rv_range: Option<(f64, f64)>,
    ) -> Self {
        Self {
            dir: dir.to_string(),
            wavelength,
            teff_range,
            m_range,
            logg_range,
            vsini_range,
            rv_range,
        }
    }

    fn load(&self) -> LoadedInMemInterpolator {
        let fetcher = InMemFetcher::new(
            &self.dir,
            Range::new(self.teff_range.clone()),
            Range::new(self.m_range.clone()),
            Range::new(self.logg_range.clone()),
            self.vsini_range.unwrap_or((1.0, 600.0)),
            self.rv_range.unwrap_or((-150.0, 150.0)),
        )
        .unwrap();
        LoadedInMemInterpolator {
            interpolator: SquareGridInterpolator::new(fetcher, self.wavelength.0),
        }
    }
}
/// Interpolator that caches the last _lrucap_ spectra
#[pyclass]
#[derive(Clone)]
pub struct CachedInterpolator {
    interpolator: SquareGridInterpolator<CachedFetcher>,
}

#[pymethods]
impl CachedInterpolator {
    #[new]
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
        teff_range: Vec<f64>,
        m_range: Vec<f64>,
        logg_range: Vec<f64>,
        vsini_range: Option<(f64, f64)>,
        rv_range: Option<(f64, f64)>,
        lrucap: Option<usize>,
    ) -> Self {
        let fetcher = CachedFetcher::new(
            dir,
            Range::new(teff_range),
            Range::new(m_range),
            Range::new(logg_range),
            vsini_range.unwrap_or((1.0, 600.0)),
            rv_range.unwrap_or((-150.0, 150.0)),
            lrucap.unwrap_or(4000),
        );
        Self {
            interpolator: SquareGridInterpolator::new(fetcher, wavelength.0),
        }
    }
}

implement_methods!(OnDiskInterpolator, interpolators::OnDiskInterpolator);
implement_methods!(LoadedInMemInterpolator, interpolators::InMemInterpolator);
implement_methods!(CachedInterpolator, interpolators::CachedInterpolator);

/// Compount Interpolator: uses multiple square grids of OnDiskInterpolator
#[pyclass]
pub struct OnDiskCompound {
    interpolator: CompoundInterpolator<OnDiskFetcher>,
}

#[pymethods]
impl OnDiskCompound {
    #[new]
    pub fn new(
        interpolator1: OnDiskInterpolator,
        interpolator2: OnDiskInterpolator,
        interpolator3: OnDiskInterpolator,
    ) -> Self {
        Self {
            interpolator: CompoundInterpolator::new(
                interpolator1.interpolator,
                interpolator2.interpolator,
                interpolator3.interpolator,
            ),
        }
    }
}

/// Compount Interpolator: uses multiple square grids of InMemInterpolator

#[pyclass]
pub struct InMemCompound {
    interpolator: CompoundInterpolator<InMemFetcher>,
}

#[pymethods]
impl InMemCompound {
    #[new]
    fn new(
        interpolator1: InMemInterpolator,
        interpolator2: InMemInterpolator,
        interpolator3: InMemInterpolator,
    ) -> Self {
        Self {
            interpolator: CompoundInterpolator::new(
                interpolator1.load().interpolator,
                interpolator2.load().interpolator,
                interpolator3.load().interpolator,
            ),
        }
    }
}

/// Compount Interpolator: uses multiple square grids of CachedInterpolator

#[pyclass]
pub struct CachedCompound {
    interpolator: CompoundInterpolator<CachedFetcher>,
}

#[pymethods]
impl CachedCompound {
    #[new]
    pub fn new(
        interpolator1: CachedInterpolator,
        interpolator2: CachedInterpolator,
        interpolator3: CachedInterpolator,
    ) -> Self {
        Self {
            interpolator: CompoundInterpolator::new(
                interpolator1.interpolator,
                interpolator2.interpolator,
                interpolator3.interpolator,
            ),
        }
    }
}

implement_methods!(
    OnDiskCompound,
    interpolate::CompoundInterpolator<interpolators::OnDiskInterpolator>
);
implement_methods!(
    InMemCompound,
    interpolate::CompoundInterpolator<interpolators::InMemInterpolator>
);
implement_methods!(
    CachedCompound,
    interpolate::CompoundInterpolator<interpolators::CachedInterpolator>
);

/// Fit continuum with chunk method.
/// Returns the fit parameters for each chunk.
/// y is the observed spectrum divided by model.
#[pyfunction]
pub fn fit_continuum(wl: Vec<f64>, y: Vec<FluxFloat>) -> PyResult<Vec<Vec<FluxFloat>>> {
    let wl_arr = na::DVector::from_vec(wl);
    let fitter = ChunkFitter::new(wl_arr, 5, 8, 0.2);
    let result = fitter._fit_chunks(&y.into());
    Ok(result.into_iter().map(|x| x.data.into()).collect())
}

/// Build continuum from chunk fit parameters
#[pyfunction]
pub fn build_continuum(wl: Vec<f64>, pfits: Vec<Vec<FluxFloat>>) -> PyResult<Vec<FluxFloat>> {
    let wl_arr = na::DVector::from_vec(wl);
    let fitter = ChunkFitter::new(wl_arr, 5, 8, 0.2);
    let parr = pfits.into_iter().map(na::DVector::from_vec).collect();
    let result = fitter.build_continuum_from_chunks(parr);
    Ok(result.data.into())
}

#[pymodule]
fn pso(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CachedInterpolator>()?;
    m.add_class::<OnDiskInterpolator>()?;
    m.add_class::<InMemInterpolator>()?;
    m.add_class::<OnDiskCompound>()?;
    m.add_class::<InMemCompound>()?;
    m.add_class::<CachedCompound>()?;
    m.add_class::<WlGrid>()?;
    m.add_class::<PSOSettings>()?;
    m.add_function(wrap_pyfunction!(VariableResolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(FixedResolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(ChunkContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(FixedContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(ConstantContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(fit_continuum, m)?)?;
    m.add_function(wrap_pyfunction!(build_continuum, m)?)?;
    Ok(())
}
