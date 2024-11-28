use crate::fitting::ObservedSpectrum;
use crate::interpolate::FluxFloat;
use anyhow::{anyhow, bail, Result};
use enum_dispatch::enum_dispatch;
use nalgebra as na;
use num_traits::Float;

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
        if observed_spectrum.flux.len() != synthetic_spectrum.len() {
            return Err(anyhow!(
                "Length of observed and synthetic spectrum are different {:?}, {:?}",
                observed_spectrum.flux.len(),
                synthetic_spectrum.len()
            ));
        }
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
    ) -> Result<na::DVector<FluxFloat>> {
        if (pfits.len() != self.n_chunks) || pfits.iter().any(|p| p.len() != self.p_order + 1) {
            bail!(
                "Incorrect number of parameters {:?}, {:?}. Expected: {:?}, {:?}",
                pfits.len(),
                pfits.iter().map(|x| x.len()).collect::<Vec<_>>(),
                self.n_chunks,
                self.p_order + 1
            );
        }
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
        Ok(continuum)
    }
    pub fn fit_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<(Vec<na::DVector<FluxFloat>>, na::DVector<FluxFloat>)> {
        if !observed_spectrum.flux.len() == synthetic_spectrum.len() {
            bail!(
                "Length of observed and synthetic spectrum are different {:?}, {:?}",
                observed_spectrum.flux.len(),
                synthetic_spectrum.len()
            );
        }
        let y = &observed_spectrum.flux.component_div(synthetic_spectrum);
        let pfits = self._fit_chunks(y);
        let continuum = self.build_continuum_from_chunks(pfits.clone())?;
        Ok((pfits, continuum))
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
        let (pfits, cont) = self.fit_and_return_continuum(observed_spectrum, synthetic_spectrum)?;
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
        if params.len() != self.n_chunks * (self.p_order + 1) {
            bail!(
                "Incorrect number of continuum parameters: {}, expected: {}",
                params.len(),
                self.n_chunks * (self.p_order + 1)
            );
        }
        let pfits: Vec<na::DVector<FluxFloat>> = params
            .data
            .as_vec()
            .chunks(self.p_order + 1)
            .map(na::DVector::from_row_slice)
            .collect();
        self.build_continuum_from_chunks(pfits)
    }

    fn fit_continuum_and_return_continuum(
        &self,
        observed_spectrum: &ObservedSpectrum,
        synthetic_spectrum: &na::DVector<FluxFloat>,
    ) -> Result<na::DVector<FluxFloat>> {
        let (_, fit) = self.fit_and_return_continuum(observed_spectrum, synthetic_spectrum)?;
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
