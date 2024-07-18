use crate::convolve_rv::WavelengthDispersion;
use crate::interpolate::{Bounds, Interpolator, FluxFloat};
use crate::particleswarm;
use anyhow::{anyhow, Context, Result};
use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{CostFunction, Error, Executor, PopulationState, State, KV};
use argmin::solver::brent::BrentRoot;
use enum_dispatch::enum_dispatch;
use nalgebra as na;
use num_traits::Float;
use rayon::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

/// Scale labels by this amount during fitting
pub const SCALING: na::SVector<f64, 5> =
    na::SVector::<f64, 5>::new(10_000.0, 1.0, 1.0, 100.0, 100.0);

/// Observed specrum with flux and variance
pub struct ObservedSpectrum {
    pub flux: na::DVector<FluxFloat>,
    pub var: na::DVector<FluxFloat>,
}

impl ObservedSpectrum {
    /// Load observed spectrum from vector of flux and vector of variance
    pub fn from_vecs(flux: Vec<FluxFloat>, var: Vec<FluxFloat>) -> Self {
        let flux = na::DVector::from_vec(flux);
        let var = na::DVector::from_vec(var);
        Self { flux, var }
    }
}

/// The function that is used to fit against the pseudo continuum must implement this trait
#[enum_dispatch]
pub trait ContinuumFitter: Send + Sync {
    /// Fit the continuum and return the parameters of the continuum and the chi2 value
    fn fit_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<(na::DVector<FluxFloat>, f64)>;

    /// Fit the continuum and return  its flux values
    fn fit_continuum_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<na::DVector<FluxFloat>>;

    /// Fit the continuum and return the synthetic spectrum (model+pseudocontinuum)
    fn fit_continuum_and_return_fit(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<na::DVector<FluxFloat>> {
        let continuum =
            self.fit_continuum_and_return_continuum(observed_spectrum, synthetic_spectrum)?;
        Ok(continuum.component_mul(synthetic_spectrum))
    }
}

/// Continuum fitting function that is a linear model
#[derive(Clone)]
pub struct LinearModelFitter {
    design_matrix: na::DMatrix<FluxFloat>,
    svd: na::linalg::SVD<FluxFloat, na::Dyn, na::Dyn>,
}

impl LinearModelFitter {
    /// Make linear model fitter from design matrix
    pub fn new(design_matrix: na::DMatrix<FluxFloat>) -> Self {
        let svd = na::linalg::SVD::new(design_matrix.clone(), true, true);
        Self { design_matrix, svd }
    }

    /// Solve the linear model
    fn solve(&self, b: &na::DVector<FluxFloat>, epsilon: FluxFloat) -> Result<na::DVector<FluxFloat>, &'static str> {
        self.svd.solve(b, epsilon)
    }
}

impl ContinuumFitter for LinearModelFitter {
    fn fit_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<(na::DVector<FluxFloat>, f64)> {
        let epsilon = 1e-14;
        let b = observed_spectrum.flux.component_div(synthetic_spectrum);
        let solution = self.solve(&b, epsilon).map_err(|err| anyhow!(err))?;

        // calculate residuals
        let model = &self.design_matrix * &solution;
        let residuals = model - b;

        let residuals_cut = residuals.rows(20, synthetic_spectrum.len() - 40);
        let var = &observed_spectrum.var;
        let var_cut = var.rows(20, var.len() - 40);
        let chi2 = residuals_cut.component_div(&var_cut).dot(&residuals_cut) as f64
            / residuals.len() as f64;
        Ok((solution, chi2))
    }

    fn fit_continuum_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<na::DVector<FluxFloat>> {
        let epsilon = 1e-14;
        let b = observed_spectrum.flux.component_div(synthetic_spectrum);
        let solution = self.solve(&b, epsilon).map_err(|err| anyhow!(err))?;

        // calculate residuals
        Ok(&self.design_matrix * &solution)
    }
}

fn map_range<F: Float>(x: F, from_low: F, from_high: F, to_low: F, to_high: F, clamp: bool) -> F {
    let value = (x - from_low) / (from_high - from_low) * (to_high - to_low) + to_low;
    if clamp {
        value.max(to_low).min(to_high)
    } else {
        value
    }
}

/// Chunk blending function
fn poly_blend(x: FluxFloat) -> FluxFloat {
    let x_c = x.max(0.0).min(1.0);
    let s = FluxFloat::sin(x_c * std::f32::consts::PI / 2.0);
    s * s
}

/// Chunk based polynomial fitter
#[derive(Clone, Debug)]
pub struct ChunkFitter {
    n_chunks: usize,
    p_order: usize,
    wl: na::DVector<f64>,
    chunks_startstop: na::DMatrix<usize>,
    design_matrices: Vec<na::DMatrix<FluxFloat>>,
    svds: Vec<na::linalg::SVD<FluxFloat, na::Dyn, na::Dyn>>,
}

impl ChunkFitter {
    /// Create a new chunk fitter with number of chunks, order of polynomial and overlapping fraction
    pub fn new(wl: na::DVector<f64>, n_chunks: usize, p_order: usize, overlap: f64) -> Self {
        let n = wl.len();
        let mut chunks_startstop = na::DMatrix::<usize>::zeros(n_chunks, 2);
        // Build start-stop indices
        for i in 0..n_chunks {
            let start = map_range(
                i as f64 - overlap,
                0.0,
                n_chunks as f64,
                wl[0],
                wl[n - 1],
                true,
            );
            let stop = map_range(
                i as f64 + 1.0 + overlap,
                0.0,
                n_chunks as f64,
                wl[0],
                wl[n - 1],
                true,
            );
            chunks_startstop[(i, 0)] = wl.iter().position(|&x| x > start).unwrap();
            chunks_startstop[(i, 1)] = wl.iter().rposition(|&x| x < stop).unwrap() + 1;
        }
        chunks_startstop[(0, 0)] = 0;
        chunks_startstop[(n_chunks - 1, 1)] = n;

        let mut dms = Vec::new();
        let mut svds = Vec::new();
        for c in 0..n_chunks {
            let start = chunks_startstop[(c, 0)];
            let stop = chunks_startstop[(c, 1)];
            let wl_remap = wl
                .rows(start, stop - start)
                .map(|x| map_range(x, wl[start], wl[stop - 1], -1.0, 1.0, false));

            let mut a = na::DMatrix::<FluxFloat>::zeros(stop - start, p_order + 1);
            for i in 0..(p_order + 1) {
                a.set_column(i, &wl_remap.map(|x| x.powi(i as i32) as FluxFloat));
            }
            dms.push(a.clone());
            svds.push(na::linalg::SVD::new(a, true, true));
        }

        Self {
            n_chunks,
            p_order,
            wl,
            chunks_startstop,
            design_matrices: dms,
            svds,
        }
    }

    pub fn _fit_chunks(&self, y: &na::DVector<FluxFloat>) -> Vec<na::DVector<FluxFloat>> {
        let mut pfits = Vec::new();
        for c in 0..self.n_chunks {
            let start = self.chunks_startstop[(c, 0)];
            let stop = self.chunks_startstop[(c, 1)];
            let y_cut = y.rows(start, stop - start);

            let pfit = self.svds[c].solve(&y_cut, 1e-14).unwrap();
            pfits.push(pfit);
        }

        pfits
    }

    pub fn build_continuum_from_chunks(&self, pfits: Vec<na::DVector<FluxFloat>>) -> na::DVector<FluxFloat> {
        let polynomials: Vec<na::DVector<FluxFloat>> = self
            .design_matrices
            .iter()
            .zip(pfits.iter())
            .map(|(dm, p)| dm * p)
            .collect();

        let mut continuum = na::DVector::<FluxFloat>::zeros(self.wl.len());

        // First chunk
        let start = self.chunks_startstop[(0, 0)];
        let stop = self.chunks_startstop[(0, 1)];
        continuum
            .rows_mut(start, stop - start)
            .copy_from(&polynomials[0]);

        for (c, p) in polynomials.into_iter().enumerate().skip(1) {
            let start = self.chunks_startstop[(c, 0)];
            let stop = self.chunks_startstop[(c, 1)];
            let stop_prev = self.chunks_startstop[(c - 1, 1)];
            let fac = self.wl.rows(start, stop - start).map(|x| {
                poly_blend(map_range(
                    x as FluxFloat,
                    self.wl[stop_prev - 1] as FluxFloat,
                    self.wl[start] as FluxFloat,
                    0.0,
                    1.0,
                    false,
                ))
            });

            let new = continuum.rows(start, stop - start).component_mul(&fac)
                + p.component_mul(&fac.map(|x| 1.0 - x));
            continuum.rows_mut(start, stop - start).copy_from(&new);
        }
        continuum
    }
    pub fn fit_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> (Vec<na::DVector<FluxFloat>>, na::DVector<FluxFloat>) {
        let flux = &observed_spectrum.flux;
        let y = flux.component_div(synthetic_spectrum);
        let pfits = self._fit_chunks(&y);
        let continuum = self.build_continuum_from_chunks(pfits.clone());
        (pfits, continuum)
    }
}

impl ContinuumFitter for ChunkFitter {
    fn fit_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<(na::DVector<FluxFloat>, f64)> {
        let flux = &observed_spectrum.flux;
        let var = &observed_spectrum.var;
        let (pfits, cont) = self.fit_and_return_continuum(observed_spectrum, synthetic_spectrum);
        let fit = synthetic_spectrum.component_mul(&cont);
        let params = na::DVector::from_iterator(
            pfits.len() * (self.p_order + 1),
            pfits.iter().flat_map(|x| x.iter()).cloned(),
        );
        let residuals = flux - fit;
        let var_cut = var.rows(20, var.len() - 40);
        let residuals_cut = residuals.rows(20, flux.len() - 40);
        let chi2 = residuals_cut.component_div(&var_cut).dot(&residuals_cut) as f64
            / residuals.len() as f64;
        Ok((params, chi2))
    }

    fn fit_continuum_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<na::DVector<FluxFloat>> {
        let (_pfits, fit) = self.fit_and_return_continuum(observed_spectrum, synthetic_spectrum);
        Ok(fit)
    }
}

/// Used to test fitting with a fixed continuum
#[derive(Clone, Debug)]
pub struct FixedContinuum {
    continuum: na::DVector<FluxFloat>,
}

impl FixedContinuum {
    pub fn new(continuum: na::DVector<FluxFloat>) -> Self {
        Self { continuum }
    }
}

impl ContinuumFitter for FixedContinuum {
    fn fit_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<(na::DVector<FluxFloat>, f64)> {
        // u8 is a dummy type
        let residuals = &observed_spectrum.flux - synthetic_spectrum.component_mul(&self.continuum);
        let chi2 = residuals
            .component_div(&observed_spectrum.var)
            .dot(&residuals) as f64
            / residuals.len() as f64;
        Ok((na::DVector::zeros(1), chi2))
    }

    fn fit_continuum_and_return_continuum(
        &self,
        _observed_spectrum: &ObservedSpectrum,
        _synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<na::DVector<FluxFloat>> {
        Ok(self.continuum.clone())
    }
}

#[derive(Clone, Debug)]
pub struct ConstantContinuum();

impl ContinuumFitter for ConstantContinuum {
    fn fit_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<(na::DVector<FluxFloat>, f64)> {
        let n = synthetic_spectrum.len();
        let target = observed_spectrum.flux.component_div(synthetic_spectrum);
        let mean = target.iter().sum::<FluxFloat>() / n as FluxFloat;
        let continuum = nalgebra::DVector::from_element(n, mean);
        let residuals = &observed_spectrum.flux - synthetic_spectrum.component_mul(&continuum);
        let chi2 = residuals
            .component_div(&observed_spectrum.var)
            .dot(&residuals) as f64
            / n as f64;
        Ok((continuum, chi2))
    }

    fn fit_continuum_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &nalgebra::DVector<FluxFloat>,
    ) -> Result<nalgebra::DVector<FluxFloat>> {
        let n = synthetic_spectrum.len();
        let target = observed_spectrum.flux.component_div(synthetic_spectrum);
        let mean = target.iter().sum::<FluxFloat>() / n as FluxFloat;
        Ok(nalgebra::DVector::from_element(n, mean))
    }
}

/// Cost function used in the PSO fitting
struct FitCostFunction<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> {
    interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    parallelize: bool,
}

impl<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> CostFunction
    for FitCostFunction<'a, I, T, F>
{
    type Param = na::SVector<f64, 5>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let rescaled = params.component_mul(&SCALING);
        let teff = rescaled[0];
        let m = rescaled[1];
        let logg = rescaled[2];
        let vsini = rescaled[3];
        let rv = rescaled[4];
        let synth_spec =
            self.interpolator
                .produce_model(self.target_dispersion, teff, m, logg, vsini, rv)?;
        let (_, ls) = self
            .continuum_fitter
            .fit_continuum(self.observed_spectrum, &synth_spec)?;
        Ok(ls)
    }

    fn parallelize(&self) -> bool {
        self.parallelize
    }
}

#[derive(Debug)]
pub struct OptimizationResult {
    pub labels: [f64; 5],
    pub continuum_params: na::DVector<FluxFloat>,
    pub ls: f64,
    pub iters: u64,
    pub time: f64,
}

#[derive(Clone, Debug)]
pub struct PSOSettings {
    pub num_particles: usize,
    pub max_iters: u64,
    pub inertia_factor: f64,
    pub cognitive_factor: f64,
    pub social_factor: f64,
    pub delta: f64,
}

fn get_pso_fitter<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter>(
    interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    settings: &PSOSettings,
    parallelize: bool,
) -> Executor<
    FitCostFunction<'a, I, T, F>,
    particleswarm::ParticleSwarm<I::B, f64, rand::rngs::StdRng>,
    PopulationState<particleswarm::Particle<na::SVector<f64, 5>, f64>, f64>,
> {
    let cost_function = FitCostFunction {
        interpolator,
        target_dispersion,
        observed_spectrum,
        continuum_fitter,
        parallelize,
    };
    let bounds = interpolator.bounds().clone();
    let solver = particleswarm::ParticleSwarm::new(bounds, settings.num_particles)
        .with_inertia_factor(settings.inertia_factor)
        .unwrap()
        .with_cognitive_factor(settings.cognitive_factor)
        .unwrap()
        .with_social_factor(settings.social_factor)
        .unwrap()
        .with_delta(settings.delta)
        .unwrap();
    Executor::new(cost_function, solver).configure(|state| state.max_iters(settings.max_iters))
}

struct PSObserver {
    dir: PathBuf,
    file_prefix: String,
}
impl PSObserver {
    fn new(directory: &str, file_prefix: &str) -> Self {
        Self {
            dir: PathBuf::from(directory),
            file_prefix: file_prefix.to_string(),
        }
    }
}

#[derive(Serialize)]
struct ParticleInfo {
    teff: f64,
    m: f64,
    logg: f64,
    vsini: f64,
    rv: f64,
    cost: f64,
}

impl From<(na::SVector<f64, 5>, f64)> for ParticleInfo {
    fn from(p: (na::SVector<f64, 5>, f64)) -> Self {
        Self {
            teff: p.0[0],
            m: p.0[1],
            logg: p.0[2],
            vsini: p.0[3],
            rv: p.0[4],
            cost: p.1,
        }
    }
}

impl Observe<PopulationState<particleswarm::Particle<na::SVector<f64, 5>, f64>, f64>>
    for PSObserver
{
    fn observe_iter(
        &mut self,
        state: &PopulationState<particleswarm::Particle<na::SVector<f64, 5>, f64>, f64>,
        _kv: &KV,
    ) -> Result<()> {
        let iter = state.get_iter();
        let particles = state.get_population().ok_or(Error::msg("No particles"))?;
        let values: Vec<ParticleInfo> = particles
            .iter()
            .map(|p| (p.position.component_mul(&SCALING), p.cost).into())
            .collect();
        let filename = self.dir.join(format!("{}_{}.json", self.file_prefix, iter));
        let f = BufWriter::new(File::create(filename)?);
        serde_json::to_writer(f, &values)?;
        Ok(())
    }
}

pub fn fit_pso<I: Interpolator>(
    interpolator: &I,
    target_dispersion: &impl WavelengthDispersion,
    observed_spectrum: &ObservedSpectrum,
    continuum_fitter: &impl ContinuumFitter,
    settings: &PSOSettings,
    save_directory: Option<String>,
    parallelize: bool,
) -> Result<OptimizationResult> {
    let fitter = get_pso_fitter(
        interpolator,
        target_dispersion,
        observed_spectrum,
        continuum_fitter,
        settings,
        parallelize,
    );
    let result = if let Some(dir) = save_directory {
        let observer = PSObserver::new(&dir, "iteration");
        fitter.add_observer(observer, ObserverMode::Always).run()?
    } else {
        fitter.run()?
    };

    let best_param = result
        .state
        .best_individual
        .ok_or(anyhow!("No best parameter found"))?
        .position;

    let best_rescaled = best_param.component_mul(&SCALING);
    let best_teff = best_rescaled[0];
    let best_m = best_rescaled[1];
    let best_logg = best_rescaled[2];
    let best_vsini = best_rescaled[3];
    let best_rv = best_rescaled[4];

    let synth_spec = interpolator.produce_model(
        target_dispersion,
        best_teff,
        best_m,
        best_logg,
        best_vsini,
        best_rv,
    )?;
    let (continuum_params, _) = continuum_fitter.fit_continuum(observed_spectrum, &synth_spec)?;
    let time = match result.state.time {
        Some(t) => t.as_secs_f64(),
        None => 0.0,
    };
    Ok(OptimizationResult {
        labels: [best_teff, best_m, best_logg, best_vsini, best_rv],
        continuum_params,
        ls: result.state.best_cost,
        time,
        iters: result.state.iter,
    })
}

struct ModelFitCost<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> {
    interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    label: &'a na::SVector<f64, 5>,
    label_index: usize,
    target_value: f64,
    search_radius: f64,
}

impl<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> CostFunction
    for ModelFitCost<'a, I, T, F>
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let mut new_label = *self.label;
        new_label[self.label_index] += param * self.search_radius;
        let synth_spec = self
            .interpolator
            .produce_model(
                self.target_dispersion,
                new_label[0],
                new_label[1],
                new_label[2],
                new_label[3],
                new_label[4],
            )
            .context(format!(
                "Error computing cost with param={:?}, new_label={:?}",
                param, new_label
            ))?;
        let (_, chi2) = self
            .continuum_fitter
            .fit_continuum(self.observed_spectrum, &synth_spec)?;
        Ok(chi2 - self.target_value)
    }
}

/// Compute uncertainty in every label
/// by finding the intersection points of the chi2 function with a target value.
pub fn uncertainty_chi2<I: Interpolator>(
    interpolator: &I,
    target_dispersion: &impl WavelengthDispersion,
    observed_spectrum: &ObservedSpectrum,
    continuum_fitter: &impl ContinuumFitter,
    label: na::SVector<f64, 5>,
    search_radius: na::SVector<f64, 5>,
) -> Result<[Result<(f64, f64)>; 5]> {
    let best_synth_spec = interpolator.produce_model(
        target_dispersion,
        label[0],
        label[1],
        label[2],
        label[3],
        label[4],
    )?;
    let (_, best_chi2) = continuum_fitter.fit_continuum(observed_spectrum, &best_synth_spec)?;
    let observed_wavelength = target_dispersion.wavelength();
    let n_p = observed_wavelength.len();
    let first_wl = observed_wavelength[0];
    let last_wl = observed_wavelength[n_p - 1];
    let n = 4.0 * 85_000.0 * (last_wl - first_wl) / (first_wl + last_wl);
    let target_chi = best_chi2 * (1.0 + (2.0 / n).sqrt());
    Ok(core::array::from_fn(|i| {
        let bounds = interpolator.bounds().clone();
        let costfunction = ModelFitCost {
            interpolator,
            target_dispersion,
            observed_spectrum,
            continuum_fitter,
            label: &label,
            label_index: i,
            target_value: target_chi,
            search_radius: search_radius[i],
        };
        let mut left_bound_label = label;
        left_bound_label[i] -= search_radius[i];
        let left_bound = bounds.clamp_1d(left_bound_label, i);
        let left_bound_rescaled = (left_bound - label[i]) / search_radius[i];
        let solver_left = BrentRoot::new(left_bound_rescaled, 0.0, 1e-3);
        let executor_left = Executor::new(costfunction, solver_left);
        let result_left = executor_left.run().context(format!(
            "Error while running uncertainty_chi2 on {:?} (left), bound={}",
            label, left_bound
        ))?;
        let left_sol = result_left
            .state
            .get_best_param()
            .context("result_left.state.get_best_param()")?;
        let left = left_sol * search_radius[i];

        let costfunction = ModelFitCost {
            interpolator,
            target_dispersion,
            observed_spectrum,
            continuum_fitter,
            label: &label,
            label_index: i,
            target_value: target_chi,
            search_radius: search_radius[i],
        };
        let mut right_bound_label = label;
        right_bound_label[i] += search_radius[i];
        let right_bound = bounds.clamp_1d(right_bound_label, i);
        let right_bound_rescaled = (right_bound - label[i]) / search_radius[i];
        let solver_right = BrentRoot::new(0.0, right_bound_rescaled.min(1.0), 1e-3);
        let executor_right = Executor::new(costfunction, solver_right);
        let result_right = executor_right.run().context(format!(
            "Error while running uncertainty_chi2 on {:?} (right), bound={}",
            label, right_bound
        ))?;
        let right_sol = result_right
            .state
            .get_best_param()
            .context("result_right.state.get_best_param()")?;
        let right = right_sol * search_radius[i];
        Ok((left, right))
    }))
}
