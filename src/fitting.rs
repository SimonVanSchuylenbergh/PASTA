use crate::bounds::{BinaryBounds, BinaryRVBounds, BoundsConstraint, PSOBounds, SingleBounds};
use crate::continuum_fitting::ContinuumFitter;
use crate::convolve_rv::{shift_and_resample, WavelengthDispersion};
use crate::interpolate::{FluxFloat, GridBounds, Interpolator, WlGrid};
use crate::particleswarm::{self};
use anyhow::{anyhow, Context, Result};
use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{CostFunction as _, Executor, PopulationState, State, KV};
use argmin::solver::brent::BrentRoot;
use nalgebra as na;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::iter::once;
use std::path::PathBuf;

/// Observed specrum with flux and variance
#[derive(Clone, Debug)]
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

// impl PSOBounds<2> for BinaryRVBounds {

// }

/// Cost function used in the PSO fitting
struct CostFunction<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> {
    interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    parallelize: bool,
}

impl<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> argmin::core::CostFunction
    for CostFunction<'a, I, T, F>
{
    type Param = na::SVector<f64, 5>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let teff = params[0];
        let m = params[1];
        let logg = params[2];
        let vsini = params[3];
        let rv = params[4];
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

#[derive(Clone, Debug)]
pub struct Label<T> {
    pub teff: T,
    pub m: T,
    pub logg: T,
    pub vsini: T,
    pub rv: T,
}

impl<T> Label<T> {
    pub fn as_array(self) -> [T; 5] {
        [self.teff, self.m, self.logg, self.vsini, self.rv]
    }

    pub fn from_array(value: [T; 5]) -> Self {
        let [teff, m, logg, vsini, rv] = value;
        Self {
            teff,
            m,
            logg,
            vsini,
            rv,
        }
    }
}
impl<T: na::Scalar> Label<T> {
    fn as_vector(self) -> na::SVector<T, 5> {
        na::SVector::from_row_slice(self.as_array().as_slice())
    }
}

impl<S: na::Storage<f64, na::Const<5>, na::Const<1>>> From<na::Vector<f64, na::Const<5>, S>>
    for Label<f64>
{
    fn from(value: na::Vector<f64, na::Const<5>, S>) -> Self {
        Self {
            teff: value[0],
            m: value[1],
            logg: value[2],
            vsini: value[3],
            rv: value[4],
        }
    }
}

#[derive(Debug)]
pub struct RvLessLabel {
    pub teff: f64,
    pub m: f64,
    pub logg: f64,
    pub vsini: f64,
}
#[derive(Debug)]
pub struct OptimizationResult {
    pub label: Label<f64>,
    pub continuum_params: na::DVector<FluxFloat>,
    pub ls: f64,
    pub iters: u64,
    pub time: f64,
}
struct BinaryCostFunction<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> {
    interpolator: &'a I,
    continuum_interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    parallelize: bool,
}

impl<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> argmin::core::CostFunction
    for BinaryCostFunction<'a, I, T, F>
{
    type Param = na::SVector<f64, 11>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let star1_parameters = params.fixed_rows::<5>(0).into_owned();
        let star2_parameters = params.fixed_rows::<5>(5).into_owned();
        let light_ratio = params[10];

        let synth_spec = self.interpolator.produce_binary_model_norm(
            self.continuum_interpolator,
            self.target_dispersion,
            &star1_parameters,
            &star2_parameters,
            light_ratio as f32,
        )?;
        let (_, ls) = self
            .continuum_fitter
            .fit_continuum(self.observed_spectrum, &synth_spec)?;
        Ok(ls)
    }

    fn parallelize(&self) -> bool {
        self.parallelize
    }
}

pub struct BinaryOptimizationResult {
    pub label1: Label<f64>,
    pub label2: Label<f64>,
    pub light_ratio: f64,
    pub continuum_params: na::DVector<FluxFloat>,
    pub ls: f64,
    pub iters: u64,
    pub time: f64,
}

struct RVCostFunction<'a, F: ContinuumFitter> {
    continuum_fitter: &'a F,
    observed_wl: &'a na::DVector<f64>,
    observed_spectrum: &'a ObservedSpectrum,
    synth_wl: &'a WlGrid,
    model1: &'a na::DVector<FluxFloat>,
    model2: &'a na::DVector<FluxFloat>,
    continuum1: &'a na::DVector<FluxFloat>,
    continuum2: &'a na::DVector<FluxFloat>,
    light_ratio: f64,
}

impl<'a, F: ContinuumFitter> argmin::core::CostFunction for RVCostFunction<'a, F> {
    type Param = na::SVector<f64, 2>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let rv1 = param[0];
        let rv2 = param[1];

        let shifted_synth1 =
            shift_and_resample(&self.model1, self.synth_wl, self.observed_wl, rv1)?;
        let shifted_synth2 =
            shift_and_resample(&self.model2, self.synth_wl, self.observed_wl, rv2)?;
        let shifted_continuum1 =
            shift_and_resample(&self.continuum1, self.synth_wl, self.observed_wl, rv1)?;
        let shifted_continuum2 =
            shift_and_resample(&self.continuum2, self.synth_wl, self.observed_wl, rv2)?;

        let synth_spec = shifted_synth1.component_mul(&shifted_continuum1)
            * self.light_ratio as FluxFloat
            + shifted_synth2.component_mul(&shifted_continuum2)
                * (1.0 - self.light_ratio as FluxFloat);

        let (_, ls) = self
            .continuum_fitter
            .fit_continuum(self.observed_spectrum, &synth_spec)?;
        Ok(ls)
    }
}

pub struct BinaryRVOptimizationResult {
    pub rv1: f64,
    pub rv2: f64,
    pub chi2: f64,
    pub iters: u64,
    pub time: f64,
}

struct BinaryTimeseriesCostFunction<
    'a,
    I: Interpolator,
    T: WavelengthDispersion,
    F: ContinuumFitter,
    B: PSOBounds<2>,
> {
    interpolator: &'a I,
    continuum_interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectra: &'a Vec<ObservedSpectrum>,
    continuum_fitter: &'a F,
    synth_wl: &'a WlGrid,
    rv_fit_settings: PSOSettings,
    rv_bounds: B,
    parallelize: bool,
}

impl<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter, B: PSOBounds<2>>
    argmin::core::CostFunction for BinaryTimeseriesCostFunction<'a, I, T, F, B>
{
    type Param = na::SVector<f64, 9>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let star1_parameters = params.fixed_rows::<4>(0).into_owned();
        let star2_parameters = params.fixed_rows::<4>(5).into_owned();
        let light_ratio = params[8];

        let synth_spec1 = self.interpolator.interpolate_and_convolve(
            self.target_dispersion,
            star1_parameters[0],
            star1_parameters[1],
            star1_parameters[2],
            star1_parameters[3],
        )?;
        let continuum1 = self.continuum_interpolator.interpolate_and_convolve(
            self.target_dispersion,
            star1_parameters[0],
            star1_parameters[1],
            star1_parameters[2],
            star1_parameters[3],
        )?;
        let synth_spec2 = self.interpolator.interpolate_and_convolve(
            self.target_dispersion,
            star2_parameters[0],
            star2_parameters[1],
            star2_parameters[2],
            star2_parameters[3],
        )?;
        let continuum2 = self.continuum_interpolator.interpolate_and_convolve(
            self.target_dispersion,
            star2_parameters[0],
            star2_parameters[1],
            star2_parameters[2],
            star2_parameters[3],
        )?;

        let chi2 = self
            .observed_spectra
            .iter()
            .map(|observed_spectrum| {
                let inner_cost_function = RVCostFunction {
                    continuum_fitter: self.continuum_fitter,
                    observed_wl: self.target_dispersion.wavelength(),
                    observed_spectrum: &observed_spectrum,
                    synth_wl: self.synth_wl,
                    model1: &synth_spec1,
                    model2: &synth_spec2,
                    continuum1: &continuum1,
                    continuum2: &continuum2,
                    light_ratio,
                };
                let solver = setup_pso(self.rv_bounds.clone(), self.rv_fit_settings.clone());
                let fitter = Executor::new(inner_cost_function, solver)
                    .configure(|state| state.max_iters(self.rv_fit_settings.max_iters));
                let result = fitter.run()?;
                Ok::<f64, anyhow::Error>(result.state.best_cost)
            })
            .sum::<Result<f64>>()?;

        Ok(chi2)
    }

    fn parallelize(&self) -> bool {
        self.parallelize
    }
}

pub struct BinaryTimeseriesOptimizationResult {
    pub label1: RvLessLabel,
    pub label2: RvLessLabel,
    pub light_ratio: f64,
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

fn setup_pso<const N: usize, B: PSOBounds<N>>(
    bounds: B,
    settings: PSOSettings,
) -> particleswarm::ParticleSwarm<N, B, f64> {
    particleswarm::ParticleSwarm::new(bounds, settings.num_particles)
        .with_inertia_factor(settings.inertia_factor)
        .unwrap()
        .with_cognitive_factor(settings.cognitive_factor)
        .unwrap()
        .with_social_factor(settings.social_factor)
        .unwrap()
        .with_delta(settings.delta)
        .unwrap()
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
        let particles = state
            .get_population()
            .ok_or(argmin::core::Error::msg("No particles"))?;
        let values: Vec<ParticleInfo> = particles
            .iter()
            .map(|p| (p.position, p.cost).into())
            .collect();
        let filename = self.dir.join(format!("{}_{}.json", self.file_prefix, iter));
        let f = BufWriter::new(File::create(filename)?);
        serde_json::to_writer(f, &values)?;
        Ok(())
    }
}

#[derive(Serialize, Debug)]
struct BinaryParticleInfo {
    teff1: f64,
    m1: f64,
    logg1: f64,
    vsini1: f64,
    rv1: f64,
    teff2: f64,
    m2: f64,
    logg2: f64,
    vsini2: f64,
    rv2: f64,
    light_ratio: f64,
    cost: f64,
}

impl From<(na::SVector<f64, 11>, f64)> for BinaryParticleInfo {
    fn from(p: (na::SVector<f64, 11>, f64)) -> Self {
        Self {
            teff1: p.0[0],
            m1: p.0[1],
            logg1: p.0[2],
            vsini1: p.0[3],
            rv1: p.0[4],
            teff2: p.0[5],
            m2: p.0[6],
            logg2: p.0[7],
            vsini2: p.0[8],
            rv2: p.0[9],
            light_ratio: p.0[10],
            cost: p.1,
        }
    }
}

impl Observe<PopulationState<particleswarm::Particle<na::SVector<f64, 11>, f64>, f64>>
    for PSObserver
{
    fn observe_iter(
        &mut self,
        state: &PopulationState<particleswarm::Particle<na::SVector<f64, 11>, f64>, f64>,
        _kv: &KV,
    ) -> Result<()> {
        let iter = state.get_iter();
        let particles = state
            .get_population()
            .ok_or(argmin::core::Error::msg("No particles"))?;
        let values: Vec<BinaryParticleInfo> = particles
            .iter()
            .map(|p| (p.position, p.cost).into())
            .collect();
        let filename = self.dir.join(format!("{}_{}.json", self.file_prefix, iter));
        let f = BufWriter::new(File::create(filename)?);
        serde_json::to_writer(f, &values)?;
        Ok(())
    }
}

#[derive(Serialize, Debug)]
struct RVParticleInfo {
    rv1: f64,
    rv2: f64,
    cost: f64,
}

impl From<(na::SVector<f64, 2>, f64)> for RVParticleInfo {
    fn from(p: (na::SVector<f64, 2>, f64)) -> Self {
        Self {
            rv1: p.0[0],
            rv2: p.0[1],
            cost: p.1,
        }
    }
}

impl Observe<PopulationState<particleswarm::Particle<na::SVector<f64, 2>, f64>, f64>>
    for PSObserver
{
    fn observe_iter(
        &mut self,
        state: &PopulationState<particleswarm::Particle<na::SVector<f64, 2>, f64>, f64>,
        _kv: &KV,
    ) -> Result<()> {
        let iter = state.get_iter();
        let particles = state
            .get_population()
            .ok_or(argmin::core::Error::msg("No particles"))?;
        let values: Vec<RVParticleInfo> = particles
            .iter()
            .map(|p| (p.position, p.cost).into())
            .collect();
        let filename = self.dir.join(format!("{}_{}.json", self.file_prefix, iter));
        let f = BufWriter::new(File::create(filename)?);
        serde_json::to_writer(f, &values)?;
        Ok(())
    }
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

impl<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> argmin::core::CostFunction
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

pub struct SingleFitter<T: WavelengthDispersion, F: ContinuumFitter> {
    target_dispersion: T,
    continuum_fitter: F,
    settings: PSOSettings,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
}

impl<T: WavelengthDispersion, F: ContinuumFitter> SingleFitter<T, F> {
    pub fn new(
        target_dispersion: T,
        continuum_fitter: F,
        settings: PSOSettings,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
    ) -> Self {
        Self {
            target_dispersion,
            continuum_fitter,
            settings,
            vsini_range,
            rv_range,
        }
    }

    pub fn fit(
        &self,
        interpolator: &impl Interpolator,
        observed_spectrum: &ObservedSpectrum,
        trace_directory: Option<String>,
        parallelize: bool,
    ) -> Result<OptimizationResult> {
        let cost_function = CostFunction {
            interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize,
        };
        let bounds = SingleBounds::new(interpolator.grid_bounds(), self.vsini_range, self.rv_range);
        let solver = setup_pso(bounds, self.settings.clone());
        let fitter = Executor::new(cost_function, solver)
            .configure(|state| state.max_iters(self.settings.max_iters));
        let result = if let Some(dir) = trace_directory {
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

        let teff = best_param[0];
        let m = best_param[1];
        let logg = best_param[2];
        let vsini = best_param[3];
        let rv = best_param[4];

        let synth_spec =
            interpolator.produce_model(&self.target_dispersion, teff, m, logg, vsini, rv)?;
        let (continuum_params, _) = self
            .continuum_fitter
            .fit_continuum(observed_spectrum, &synth_spec)?;
        let time = match result.state.time {
            Some(t) => t.as_secs_f64(),
            None => 0.0,
        };
        Ok(OptimizationResult {
            label: Label {
                teff,
                m,
                logg,
                vsini,
                rv,
            },
            continuum_params,
            ls: result.state.best_cost,
            time,
            iters: result.state.iter,
        })
    }

    pub fn chi2(
        &self,
        interpolator: &impl Interpolator,
        observed_spectrum: &ObservedSpectrum,
        labels: na::SVector<f64, 5>,
    ) -> Result<f64> {
        let cost_function = CostFunction {
            interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize: false,
        };

        cost_function.cost(&labels)
    }

    /// Compute uncertainty in every label
    /// by finding the intersection points of the chi2 function with a target value.
    pub fn compute_uncertainty(
        &self,
        interpolator: &impl Interpolator,
        observed_spectrum: &ObservedSpectrum,
        spec_res: f64,
        label: Label<f64>,
        search_radius: Label<f64>,
    ) -> Result<Label<(Result<f64>, Result<f64>)>> {
        let mut search_radius = search_radius.as_vector();
        let label = label.as_vector();
        search_radius[0] *= label[0];
        search_radius[3] = (label[3] * search_radius[3]).max(100.0);
        let best_synth_spec = interpolator.produce_model(
            &self.target_dispersion,
            label[0],
            label[1],
            label[2],
            label[3],
            label[4],
        )?;
        let (_, best_chi2) = self
            .continuum_fitter
            .fit_continuum(observed_spectrum, &best_synth_spec)?;
        let observed_wavelength = self.target_dispersion.wavelength();
        let n_p = observed_wavelength.len();
        let first_wl = observed_wavelength[0];
        let last_wl = observed_wavelength[n_p - 1];
        let n = 4.0 * spec_res * (last_wl - first_wl) / (first_wl + last_wl);
        let target_chi = best_chi2 * (1.0 + (2.0 / n).sqrt());

        let computer = |i| {
            // Maybe don't hardcode here
            let bounds = SingleBounds::new(interpolator.grid_bounds(), (0.0, 1e4), (-1e3, 1e3));

            let get_bound = |right: bool| {
                let costfunction = ModelFitCost {
                    interpolator,
                    target_dispersion: &self.target_dispersion,
                    observed_spectrum,
                    continuum_fitter: &self.continuum_fitter,
                    label: &label,
                    label_index: i,
                    target_value: target_chi,
                    search_radius: search_radius[i],
                };
                let mut bound_label = label;
                if right {
                    bound_label[i] += search_radius[i];
                } else {
                    bound_label[i] -= search_radius[i];
                }
                let bound = bounds.clamp_1d(bound_label, i)?;
                let bound_rescaled = (bound - label[i]) / search_radius[i];
                let solver = if right {
                    BrentRoot::new(0.0, bound_rescaled.min(1.0), 1e-3)
                } else {
                    BrentRoot::new(bound_rescaled, 0.0, 1e-3)
                };
                let executor = Executor::new(costfunction, solver);
                let result = executor.run().context(format!(
                    "Error while running uncertainty_chi2 on {:?}, bound={}",
                    label, bound
                ))?;
                let sol = result
                    .state
                    .get_best_param()
                    .context("result.state.get_best_param()")?;
                if right {
                    Ok(sol * search_radius[i])
                } else {
                    Ok(-sol * search_radius[i]) // Minus because we are looking for the negative of the solution
                }
            };

            (get_bound(false), get_bound(true))
        };
        Ok(Label {
            teff: computer(0),
            m: computer(1),
            logg: computer(2),
            vsini: computer(3),
            rv: computer(4),
        })
    }
}

pub struct BinaryFitter<T: WavelengthDispersion, F: ContinuumFitter> {
    target_dispersion: T,
    continuum_fitter: F,
    settings: PSOSettings,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
}

impl<T: WavelengthDispersion, F: ContinuumFitter> BinaryFitter<T, F> {
    pub fn new(
        target_dispersion: T,
        continuum_fitter: F,
        settings: PSOSettings,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
    ) -> Self {
        Self {
            target_dispersion,
            continuum_fitter,
            settings,
            vsini_range,
            rv_range,
        }
    }

    pub fn fit<I: Interpolator>(
        &self,
        interpolator: &I,
        continuum_interpolator: &I,
        observed_spectrum: &ObservedSpectrum,
        trace_directory: Option<String>,
        parallelize: bool,
        constraints: Vec<BoundsConstraint>,
    ) -> Result<BinaryOptimizationResult> {
        let cost_function = BinaryCostFunction {
            interpolator,
            continuum_interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize,
        };
        let bounds = BinaryBounds::new(
            interpolator.grid_bounds(),
            (0.0, 0.5),
            self.vsini_range,
            self.rv_range,
        )
        .with_constraints(constraints);
        let solver = setup_pso(bounds, self.settings.clone());
        let fitter = Executor::new(cost_function, solver)
            .configure(|state| state.max_iters(self.settings.max_iters));

        let result = if let Some(dir) = trace_directory {
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

        let star1_parameters = best_param.fixed_rows::<5>(0).into_owned();
        let star2_parameters = best_param.fixed_rows::<5>(5).into_owned();
        let light_ratio = best_param[10];

        let synth_spec = interpolator.produce_binary_model_norm(
            continuum_interpolator,
            &self.target_dispersion,
            &star1_parameters,
            &star2_parameters,
            light_ratio as f32,
        )?;
        let (continuum_params, _) = self
            .continuum_fitter
            .fit_continuum(observed_spectrum, &synth_spec)?;
        let time = match result.state.time {
            Some(t) => t.as_secs_f64(),
            None => 0.0,
        };
        Ok(BinaryOptimizationResult {
            label1: best_param.fixed_rows::<5>(0).into(),
            label2: best_param.fixed_rows::<5>(5).into(),
            light_ratio: best_param[10],
            continuum_params,
            ls: result.state.best_cost,
            time,
            iters: result.state.iter,
        })
    }

    pub fn chi2<I: Interpolator>(
        &self,
        interpolator: &I,
        continuum_interpolator: &I,
        observed_spectrum: &ObservedSpectrum,
        labels: na::SVector<f64, 11>,
    ) -> Result<f64> {
        let cost_function = BinaryCostFunction {
            interpolator,
            continuum_interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize: false,
        };

        cost_function.cost(&labels)
    }
}

pub struct BinaryRVFitter<F: ContinuumFitter> {
    observed_wl: na::DVector<f64>,
    continuum_fitter: F,
    settings: PSOSettings,
    rv_range: (f64, f64),
}

impl<F: ContinuumFitter> BinaryRVFitter<F> {
    pub fn new(
        observed_wl: na::DVector<f64>,
        continuum_fitter: F,
        settings: PSOSettings,
        rv_range: (f64, f64),
    ) -> Self {
        Self {
            observed_wl,
            continuum_fitter,
            settings,
            rv_range,
        }
    }

    pub fn fit(
        &self,
        model1: &na::DVector<FluxFloat>,
        model2: &na::DVector<FluxFloat>,
        continuum1: &na::DVector<FluxFloat>,
        continuum2: &na::DVector<FluxFloat>,
        light_ratio: f64,
        synth_wl: &WlGrid,
        observed_spectrum: &ObservedSpectrum,
        trace_directory: Option<String>,
        constraints: Vec<BoundsConstraint>,
    ) -> Result<BinaryRVOptimizationResult> {
        let cost_function = RVCostFunction {
            observed_wl: &self.observed_wl,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            model1: &model1,
            model2: &model2,
            continuum1: &continuum1,
            continuum2: &continuum2,
            light_ratio,
            synth_wl,
        };
        let bounds = BinaryRVBounds::new(self.rv_range).with_constraints(constraints);
        let solver = setup_pso(bounds, self.settings.clone());
        let fitter = Executor::new(cost_function, solver)
            .configure(|state| state.max_iters(self.settings.max_iters));

        let result = if let Some(dir) = trace_directory {
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

        let rv1 = best_param[0];
        let rv2 = best_param[1];

        let time = match result.state.time {
            Some(t) => t.as_secs_f64(),
            None => 0.0,
        };
        Ok(BinaryRVOptimizationResult {
            rv1,
            rv2,
            chi2: result.state.best_cost,
            time,
            iters: result.state.iter,
        })
    }
}
