// #![allow(unused_imports)]
// #![allow(dead_code)]

mod convolve_rv;
mod cubic;
mod fitting;
mod interpolate;
mod model_fetchers;
mod particleswarm;
use crate::fitting::ObservedSpectrum;
use crate::interpolate::{Grid, GridInterpolator};
use anyhow::Result;
use convolve_rv::{
    oa_convolve, rot_broad_rv, NoConvolutionDispersionTarget, VariableTargetDispersion,
    WavelengthDispersion,
};
use cubic::{calculate_interpolation_coefficients, calculate_interpolation_coefficients_flat};
use fitting::{fit_pso, uncertainty_chi2, ChunkFitter, ContinuumFitter, PSOSettings};
use interpolate::{Interpolator, Range, WlGrid};
use iter_num_tools::arange;
use itertools::Itertools;
use model_fetchers::{InMemFetcher, OnDiskFetcher};
use nalgebra as na;
use particleswarm::PSOBounds;
use rayon::prelude::*;
use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;

const teff: f64 = 27000.0;
const m: f64 = 0.05;
const logg: f64 = 4.5;
const vsini: f64 = 100.0;
const rv: f64 = 0.0;

pub fn main() -> Result<()> {
    let folder = "/Users/ragnar/Documents/hermesnet/boss_grid_without_continuum_32";
    let wl_grid = WlGrid::Logspace(3.5440680443502757, 5.428681023790647e-06, 87508);
    let interpolator1 = GridInterpolator::new(
        OnDiskFetcher::new(folder, (1.0, 300.0), (-150.0, 150.0))?,
        wl_grid,
    );

    println!(
        "{}",
        interpolator1
            .bounds_single()
            .clamp_1d(na::Vector5::new(35_000.0, 0.0, 2.99, 5.0, 0.0), 0)?
    );

    // let dispersion = NoConvolutionDispersionTarget();
    // let start = Instant::now();
    // (0..500).into_par_iter().for_each(|_| {
    //     let _ = interpolator1
    //         .produce_model(&dispersion, teff, m, logg, vsini, rv)
    //         .unwrap();
    // });
    Ok(())
}
