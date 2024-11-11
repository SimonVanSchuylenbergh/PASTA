use crate::continuum_fitting::ContinuumFitter;
use crate::convolve_rv::WavelengthDispersion;
use crate::interpolate::{FluxFloat, GridBounds, Interpolator};
use crate::particleswarm::{self, PSOBounds};
use anyhow::{anyhow, Context, Result};
use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{Executor, PopulationState, State, KV};
use argmin::solver::brent::BrentRoot;
use argmin_math::ArgminRandom;
use nalgebra as na;
use num_traits::Float;
use rand::Rng;
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

#[derive(Clone)]
pub struct GridBoundsSingle<B: GridBounds> {
    grid: B,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
}

impl<B: GridBounds> GridBoundsSingle<B> {
    fn new(grid: B, vsini_range: (f64, f64), rv_range: (f64, f64)) -> Self {
        Self {
            grid,
            vsini_range,
            rv_range,
        }
    }
}

impl<B: GridBounds> PSOBounds<5> for GridBoundsSingle<B> {
    fn limits(&self) -> (nalgebra::SVector<f64, 5>, nalgebra::SVector<f64, 5>) {
        let (min, max) = self.grid.limits();
        (
            nalgebra::SVector::from_iterator(
                min.iter()
                    .copied()
                    .chain(once(self.vsini_range.0))
                    .chain(once(self.rv_range.0)),
            ),
            nalgebra::SVector::from_iterator(
                max.iter()
                    .copied()
                    .chain(once(self.vsini_range.1))
                    .chain(once(self.rv_range.1)),
            ),
        )
    }

    fn clamp_1d(&self, param: nalgebra::SVector<f64, 5>, index: usize) -> Result<f64> {
        match index {
            3 => Ok(param[3].clamp(self.vsini_range.0, self.vsini_range.1)),
            4 => Ok(param[4].clamp(self.rv_range.0, self.rv_range.1)),
            i => self.grid.clamp_1d(param.fixed_rows::<3>(0).into_owned(), i),
        }
    }

    fn generate_random_within_bounds(
        &self,
        rng: &mut impl Rng,
        num_particles: usize,
    ) -> Vec<nalgebra::SVector<f64, 5>> {
        let mut particles = Vec::with_capacity(num_particles);
        let (min, max) = self.limits();
        while particles.len() < num_particles {
            let param = na::SVector::rand_from_range(&min, &max, rng);
            if self
                .grid
                .is_within_bounds(param.fixed_rows::<3>(0).into_owned())
            {
                particles.push(param);
            }
        }
        particles
    }
}

#[derive(Clone)]
pub struct GridBoundsBinary<B: GridBounds> {
    grid: B,
    light_ratio: (f64, f64),
    vsini_range: (f64, f64),
    rv_range1: (f64, f64),
    rv_range2: (f64, f64),
}

impl<B: GridBounds> GridBoundsBinary<B> {
    fn new(
        grid: B,
        light_ratio: (f64, f64),
        vsini_range: (f64, f64),
        rv_range1: (f64, f64),
        rv_range2: (f64, f64),
    ) -> Self {
        Self {
            grid,
            light_ratio,
            vsini_range,
            rv_range1,
            rv_range2,
        }
    }

    fn get_first_grid(&self) -> GridBoundsSingle<B> {
        GridBoundsSingle::new(self.grid.clone(), self.vsini_range, self.rv_range1)
    }

    fn get_second_grid(&self) -> GridBoundsSingle<B> {
        GridBoundsSingle::new(self.grid.clone(), self.vsini_range, self.rv_range2)
    }
}

impl<B: GridBounds> PSOBounds<11> for GridBoundsBinary<B> {
    fn limits(&self) -> (na::SVector<f64, 11>, na::SVector<f64, 11>) {
        let (min, max) = self.grid.limits();
        (
            na::SVector::from_iterator(
                min.iter()
                    .copied()
                    .chain(once(self.vsini_range.0))
                    .chain(once(self.rv_range1.0))
                    .chain(min.iter().copied())
                    .chain(once(self.vsini_range.0))
                    .chain(once(self.rv_range2.0))
                    .chain(once(self.light_ratio.0)),
            ),
            na::SVector::from_iterator(
                max.iter()
                    .copied()
                    .chain(once(self.vsini_range.1))
                    .chain(once(self.rv_range1.1))
                    .chain(max.iter().copied())
                    .chain(once(self.vsini_range.1))
                    .chain(once(self.rv_range2.1))
                    .chain(once(self.light_ratio.1)),
            ),
        )
    }

    fn clamp_1d(&self, param: na::SVector<f64, 11>, index: usize) -> Result<f64> {
        let grid1 = self.get_first_grid();
        let grid2 = self.get_second_grid();
        if index < 5 {
            grid1.clamp_1d(param.fixed_rows::<5>(0).into_owned(), index)
        } else if index < 10 {
            grid2.clamp_1d(param.fixed_rows::<5>(5).into_owned(), index - 5)
        } else {
            Ok(param[10].max(self.light_ratio.0).min(self.light_ratio.1))
        }
    }

    fn generate_random_within_bounds(
        &self,
        rng: &mut impl Rng,
        num_particles: usize,
    ) -> Vec<na::SVector<f64, 11>> {
        let grid1 = self.get_first_grid();
        let grid2 = self.get_second_grid();
        let first = grid1.generate_random_within_bounds(rng, num_particles);
        let second = grid2.generate_random_within_bounds(rng, num_particles);
        let light_ratios = (0..num_particles)
            .map(|_| rng.gen_range(self.light_ratio.0..self.light_ratio.1))
            .collect::<Vec<f64>>();
        first
            .into_iter()
            .zip(second)
            .zip(light_ratios)
            .map(|((first, second), light_ratio)| {
                na::SVector::from_iterator(
                    first
                        .iter()
                        .copied()
                        .chain(second.iter().copied())
                        .chain(once(light_ratio)),
                )
            })
            .collect()
    }
}

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

#[derive(Debug)]
pub struct Label {
    pub teff: f64,
    pub m: f64,
    pub logg: f64,
    pub vsini: f64,
    pub rv: f64,
}


impl<S: na::Storage<f64, na::Const<5>, na::Const<1>>> From<na::Vector<f64, na::Const<5>, S>>
    for Label
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
pub struct OptimizationResult {
    pub label: Label,
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
    pub label1: Label,
    pub label2: Label,
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

#[derive(Serialize)]
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

pub struct PSOFitter<T: WavelengthDispersion, F: ContinuumFitter> {
    target_dispersion: T,
    continuum_fitter: F,
    settings: PSOSettings,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
}

impl<T: WavelengthDispersion, F: ContinuumFitter> PSOFitter<T, F> {
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
        let bounds =
            GridBoundsSingle::new(interpolator.grid_bounds(), self.vsini_range, self.rv_range);
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

    /// Compute uncertainty in every label
    /// by finding the intersection points of the chi2 function with a target value.
    pub fn compute_uncertainty(
        &self,
        interpolator: &impl Interpolator,
        observed_spectrum: &ObservedSpectrum,
        spec_res: f64,
        label: na::SVector<f64, 5>,
        search_radius: na::SVector<f64, 5>,
    ) -> Result<[(Result<f64>, Result<f64>); 5]> {
        let mut search_radius = search_radius;
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

        Ok(core::array::from_fn(|i| {
            // Maybe don't hardcode here
            let bounds = GridBoundsSingle::new(interpolator.grid_bounds(), (0.0, 1e4), (-1e3, 1e3));

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
        }))
    }

    pub fn fit_binary_normalized<I: Interpolator>(
        &self,
        interpolator: &I,
        continuum_interpolator: &I,
        observed_spectrum: &ObservedSpectrum,
        trace_directory: Option<String>,
        parallelize: bool,
    ) -> Result<BinaryOptimizationResult> {
        let cost_function = BinaryCostFunction {
            interpolator,
            continuum_interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize,
        };
        let bounds = GridBoundsBinary::new(
            interpolator.grid_bounds(),
            (0.0, 1.0),
            (0.0, 1e4),
            (-1e3, 1e3),
            (-1e3, 1e3),
        );
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
}
