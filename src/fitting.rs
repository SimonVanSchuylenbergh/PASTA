use crate::convolve_rv::WavelengthDispersion;
use crate::interpolate::{FluxFloat, GridBoundsBinary, GridBoundsSingle, Interpolator};
use crate::particleswarm::{self, PSOBounds};
use anyhow::{anyhow, Context, Result};
use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{CostFunction, Error, Executor, PopulationState, State, KV};
use argmin::solver::brent::BrentRoot;
use enum_dispatch::enum_dispatch;
use nalgebra as na;
use num_traits::Float;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
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

/// The function that is used to fit against the pseudo continuum must implement this trait
#[enum_dispatch]
pub trait ContinuumFitter: Send + Sync {
    /// Fit the continuum and return the parameters of the continuum and the chi2 value
    fn fit_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<(na::DVector<FluxFloat>, f64)>;

    /// Build the continuum from the parameters
    fn build_continuum(&self, params: &na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>>;

    /// Fit the continuum and return  its flux values
    fn fit_continuum_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<na::DVector<FluxFloat>> {
        let (fit, _) = self.fit_continuum(observed_spectrum, synthetic_spectrum)?;
        self.build_continuum(&fit)
    }

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
#[derive(Clone, Debug)]
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
    fn solve(
        &self,
        b: &na::DVector<FluxFloat>,
        epsilon: FluxFloat,
    ) -> Result<na::DVector<FluxFloat>, &'static str> {
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

        let residuals_cut = residuals.rows(50, synthetic_spectrum.len() - 100);
        let var = &observed_spectrum.var;
        let var_cut = var.rows(50, var.len() - 100);
        let chi2 = residuals_cut.component_div(&var_cut).dot(&residuals_cut) as f64
            / residuals.len() as f64;
        Ok((solution, chi2))
    }

    fn build_continuum(&self, params: &na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
        if params.len() != self.design_matrix.ncols() {
            return Err(anyhow!(
                "Incorrect number of parameters {:?}, expected {:?}",
                params.len(),
                self.design_matrix.ncols()
            ));
        }
        Ok(&self.design_matrix * params)
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
    let x_c = x.clamp(0.0, 1.0);
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
        (0..self.n_chunks)
            .map(|c| {
                let start = self.chunks_startstop[(c, 0)];
                let stop = self.chunks_startstop[(c, 1)];
                let y_cut = y.rows(start, stop - start);

                self.svds[c].solve(&y_cut, 1e-14).unwrap()
            })
            .collect()
    }

    pub fn build_continuum_from_chunks(
        &self,
        pfits: Vec<na::DVector<FluxFloat>>,
    ) -> na::DVector<FluxFloat> {
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

        // Rest of the chunks
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
        let y = &observed_spectrum.flux.component_div(synthetic_spectrum);
        let pfits = self._fit_chunks(y);
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
        // Throw away the first and last 20 pixels in chi2 calculation
        let var_cut = var.rows(50, var.len() - 100);
        let residuals_cut = residuals.rows(50, flux.len() - 100);
        let chi2 = residuals_cut.component_div(&var_cut).dot(&residuals_cut) as f64
            / residuals_cut.len() as f64;
        Ok((params, chi2))
    }

    fn build_continuum(&self, params: &na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
        let pfits: Vec<na::DVector<FluxFloat>> = params
            .data
            .as_vec()
            .chunks(self.p_order + 1)
            .map(na::DVector::from_row_slice)
            .collect();
        Ok(self.build_continuum_from_chunks(pfits))
    }

    fn fit_continuum_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<na::DVector<FluxFloat>> {
        let (_, fit) = self.fit_and_return_continuum(observed_spectrum, synthetic_spectrum);
        Ok(fit)
    }
}

/// Used to test fitting with a fixed continuum
#[derive(Clone, Debug)]
pub struct FixedContinuum {
    continuum: na::DVector<FluxFloat>,
    ignore_first_and_last: usize,
}

impl FixedContinuum {
    pub fn new(continuum: na::DVector<FluxFloat>, ignore_first_and_last: usize) -> Self {
        Self {
            continuum,
            ignore_first_and_last,
        }
    }
}

impl ContinuumFitter for FixedContinuum {
    fn fit_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<(na::DVector<FluxFloat>, f64)> {
        if observed_spectrum.flux.len() != synthetic_spectrum.len() {
            return Err(anyhow!(
                "Length of observed and synthetic spectrum are different {:?}, {:?}",
                observed_spectrum.flux.len(),
                synthetic_spectrum.len()
            ));
        }
        if observed_spectrum.flux.len() != self.continuum.len() {
            return Err(anyhow!(
                "Length of observed and continuum spectrum are different {:?}, {:?}",
                observed_spectrum.flux.len(),
                self.continuum.len()
            ));
        }
        let residuals = &observed_spectrum.flux - synthetic_spectrum.component_mul(&self.continuum);
        let residuals_cut = residuals.rows(
            self.ignore_first_and_last,
            observed_spectrum.flux.len() - self.ignore_first_and_last * 2,
        );
        let var_cut = observed_spectrum.var.rows(
            self.ignore_first_and_last,
            observed_spectrum.flux.len() - self.ignore_first_and_last * 2,
        );
        let chi2 = residuals_cut.component_div(&var_cut).dot(&residuals_cut) as f64
            / residuals_cut.len() as f64;
        // Return a dummy vec of zero as this is a fixed continuum
        Ok((na::DVector::zeros(1), chi2))
    }

    fn build_continuum(&self, _params: &na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
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
        let continuum = na::DVector::from_element(n, mean);
        let residuals = &observed_spectrum.flux - synthetic_spectrum.component_mul(&continuum);
        let chi2 = residuals
            .component_div(&observed_spectrum.var)
            .dot(&residuals) as f64
            / n as f64;
        Ok((vec![mean].into(), chi2))
    }

    fn build_continuum(&self, params: &na::DVector<FluxFloat>) -> Result<na::DVector<FluxFloat>> {
        Ok(na::DVector::from_element(1, params[0]))
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
pub struct OptimizationResult {
    pub labels: [f64; 5],
    pub continuum_params: na::DVector<FluxFloat>,
    pub ls: f64,
    pub iters: u64,
    pub time: f64,
}
struct BinaryFitCostFunction<
    'a,
    I1: Interpolator,
    I2: Interpolator,
    T: WavelengthDispersion,
    F: ContinuumFitter,
> {
    interpolator: &'a I1,
    continuum_interpolator: &'a I2,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    parallelize: bool,
}

impl<'a, I1: Interpolator, I2: Interpolator, T: WavelengthDispersion, F: ContinuumFitter>
    CostFunction for BinaryFitCostFunction<'a, I1, I2, T, F>
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
    pub labels1: [f64; 5],
    pub labels2: [f64; 5],
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

fn get_pso_fitter<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter>(
    interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    settings: &PSOSettings,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
    parallelize: bool,
) -> Executor<
    FitCostFunction<'a, I, T, F>,
    particleswarm::ParticleSwarm<5, GridBoundsSingle<I::GB>, f64>,
    PopulationState<particleswarm::Particle<na::SVector<f64, 5>, f64>, f64>,
> {
    let cost_function = FitCostFunction {
        interpolator,
        target_dispersion,
        observed_spectrum,
        continuum_fitter,
        parallelize,
    };
    let bounds = interpolator.bounds_single(vsini_range, rv_range).clone();
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

fn get_binary_pso_fitter<
    'a,
    I1: Interpolator,
    I2: Interpolator,
    T: WavelengthDispersion,
    F: ContinuumFitter,
>(
    interpolator: &'a I1,
    continuum_interpolator: &'a I2,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    settings: &PSOSettings,
    vsini_range: (f64, f64),
    rv_range1: (f64, f64),
    rv_range2: (f64, f64),
    parallelize: bool,
) -> Executor<
    BinaryFitCostFunction<'a, I1, I2, T, F>,
    particleswarm::ParticleSwarm<11, GridBoundsBinary<I1::GB>, f64>,
    PopulationState<particleswarm::Particle<na::SVector<f64, 11>, f64>, f64>,
> {
    let cost_function = BinaryFitCostFunction {
        interpolator,
        continuum_interpolator,
        target_dispersion,
        observed_spectrum,
        continuum_fitter,
        parallelize,
    };
    let bounds = interpolator
        .bounds_binary(vsini_range, rv_range1, rv_range2)
        .clone();
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
        let particles = state.get_population().ok_or(Error::msg("No particles"))?;
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

pub fn fit_pso(
    interpolator: &impl Interpolator,
    target_dispersion: &impl WavelengthDispersion,
    observed_spectrum: &ObservedSpectrum,
    continuum_fitter: &impl ContinuumFitter,
    settings: &PSOSettings,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
    trace_directory: Option<String>,
    parallelize: bool,
) -> Result<OptimizationResult> {
    let fitter = get_pso_fitter(
        interpolator,
        target_dispersion,
        observed_spectrum,
        continuum_fitter,
        settings,
        vsini_range,
        rv_range,
        parallelize,
    );
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

    let best_teff = best_param[0];
    let best_m = best_param[1];
    let best_logg = best_param[2];
    let best_vsini = best_param[3];
    let best_rv = best_param[4];

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

pub fn fit_pso_binary_norm(
    interpolator: &impl Interpolator,
    continuum_interpolator: &impl Interpolator,
    target_dispersion: &impl WavelengthDispersion,
    observed_spectrum: &ObservedSpectrum,
    continuum_fitter: &impl ContinuumFitter,
    settings: &PSOSettings,
    vsini_range: (f64, f64),
    rv_range1: (f64, f64),
    rv_range2: (f64, f64),
    trace_directory: Option<String>,
    parallelize: bool,
) -> Result<BinaryOptimizationResult> {
    let fitter = get_binary_pso_fitter(
        interpolator,
        continuum_interpolator,
        target_dispersion,
        observed_spectrum,
        continuum_fitter,
        settings,
        vsini_range,
        rv_range1,
        rv_range2,
        parallelize,
    );
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
        target_dispersion,
        &star1_parameters,
        &star2_parameters,
        light_ratio as f32,
    )?;
    let (continuum_params, _) = continuum_fitter.fit_continuum(observed_spectrum, &synth_spec)?;
    let time = match result.state.time {
        Some(t) => t.as_secs_f64(),
        None => 0.0,
    };
    Ok(BinaryOptimizationResult {
        labels1: [
            best_param[0],
            best_param[1],
            best_param[2],
            best_param[3],
            best_param[4],
        ],
        labels2: [
            best_param[5],
            best_param[6],
            best_param[7],
            best_param[8],
            best_param[9],
        ],
        light_ratio: best_param[10],
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
    spec_res: f64,
    label: na::SVector<f64, 5>,
    search_radius: na::SVector<f64, 5>,
) -> Result<[(Result<f64>, Result<f64>); 5]> {
    let mut search_radius = search_radius;
    search_radius[0] *= label[0];
    search_radius[3] = (label[3] * search_radius[3]).max(100.0);
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
    let n = 4.0 * spec_res * (last_wl - first_wl) / (first_wl + last_wl);
    let target_chi = best_chi2 * (1.0 + (2.0 / n).sqrt());

    Ok(core::array::from_fn(|i| {
        // Maybe don't hardcode here
        let bounds = interpolator.bounds_single((0.0, 1e4), (-1e3, 1e3)).clone();

        let get_bound = |right: bool| {
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
