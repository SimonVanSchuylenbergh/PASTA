#![allow(unused_imports, dead_code, non_upper_case_globals, unused)]

mod convolve_rv;
mod cubic;
mod continuum_fitting;
mod fitting;
mod interpolate;
mod model_fetchers;
mod particleswarm;
use crate::fitting::ObservedSpectrum;
use crate::interpolate::{Grid, GridInterpolator};
use anyhow::Result;
use convolve_rv::{
    oa_convolve, NoConvolutionDispersionTarget, VariableTargetDispersion,
    WavelengthDispersion,
};
use cubic::{calculate_interpolation_coefficients, calculate_interpolation_coefficients_flat};
use fitting::{PSOSettings};
use interpolate::{GridBounds, Interpolator, Range, WlGrid};
use continuum_fitting::ChunkFitter;
use iter_num_tools::arange;
use itertools::Itertools;
use model_fetchers::{CachedFetcher, InMemFetcher, OnDiskFetcher};
use nalgebra as na;
use npy::NpyData;
use particleswarm::PSOBounds;
use rayon::prelude::*;
use std::borrow::Cow;
use std::hint::black_box;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

const teff: f64 = 27000.0;
const m: f64 = 0.05;
const logg: f64 = 3.5;
const vsini: f64 = 100.0;
const rv: f64 = 0.0;

pub fn read_npy_file(file_path: PathBuf) -> Result<na::DVector<f64>> {
    let mut file = std::fs::File::open(file_path.clone())?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let data: NpyData<f64> = NpyData::from_bytes(&buf)?;
    Ok(na::DVector::from_iterator(data.len(), data))
}

pub fn main() -> Result<()> {
    let folder = "/STER/hermesnet/hermes_norm_convolved_u16";
    let wl_grid = WlGrid::Logspace(3.6020599913, 2e-6, 76_145);
    let interpolator1 = GridInterpolator::new(
        // OnDiskFetcher::new(folder, (1.0, 600.0), (-150.0, 150.0))?,
        CachedFetcher::new(folder, false, 3000, 1)?,
        wl_grid,
    );

    println!(
        "{}",
        interpolator1
            .grid_bounds()
            .clamp_1d(na::Vector3::new(35_000.0, 0.0, 2.99), 0)?
    );

    let wl = read_npy_file("wl_hermes.npy".into())?;
    let flux = read_npy_file("flux_hermes.npy".into())?.map(|x| x as f32);
    let var = read_npy_file("var_hermes.npy".into())?.map(|x| x as f32);
    let spec = ObservedSpectrum::from_vecs(flux.data.as_vec().clone(), var.data.as_vec().clone());
    let fitter = ChunkFitter::new(wl.clone(), 10, 5, 0.2);
    let settings = PSOSettings {
        num_particles: 44,
        max_iters: 100,
        inertia_factor: 0.7213475204,
        cognitive_factor: 1.1931471806,
        social_factor: 0.5,
        delta: 1e-7,
    };
    let dispersion = NoConvolutionDispersionTarget(wl);
    let start = Instant::now();
    let (min_teff, max_teff) = (10_000.0, 30_000.0);
    (0..4400).into_par_iter().for_each(|i| {
        let new_teff = min_teff + (max_teff - min_teff) * (i as f64) / 4400.0;
        let model = interpolator1
            .produce_model(&dispersion, teff, m, logg, vsini, rv)
            .unwrap();
        // let (_, chi) = fitter.fit_continuum(&spec, &model).unwrap();
        // let _ = interpolator1.grid().clamp_1d(na::Vector5::new(8000.0, 0.0, 2.9, 100.0, 0.0), 3).unwrap();
    });
    // (0..48).into_par_iter().for_each(|_| {
    //     let _ = fit_pso(&interpolator1, &dispersion, &spec, &fitter, &settings, None, false).unwrap();
    // });
    println!("Time: {:?}", start.elapsed());
    Ok(())
}
