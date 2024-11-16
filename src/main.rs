#![allow(unused_imports, dead_code, non_upper_case_globals, unused)]

mod bounds;
mod continuum_fitting;
mod convolve_rv;
mod cubic;
mod fitting;
mod interpolate;
mod model_fetchers;
mod particleswarm;
use crate::fitting::ObservedSpectrum;
use crate::interpolate::{Grid, GridInterpolator};
use anyhow::Result;
use bounds::PSOBounds;
use bounds::{BoundsConstraint, Constraint};
use continuum_fitting::ChunkFitter;
use convolve_rv::{
    oa_convolve, NoConvolutionDispersionTarget, VariableTargetDispersion, WavelengthDispersion,
};
use cubic::{calculate_interpolation_coefficients, calculate_interpolation_coefficients_flat};
use fitting::{BinaryFitter, PSOSettings, SingleFitter};
use interpolate::{GridBounds, Interpolator, Range, WlGrid};
use iter_num_tools::arange;
use itertools::Itertools;
use model_fetchers::{CachedFetcher, InMemFetcher, OnDiskFetcher};
use nalgebra::{self as na, constraint};
use npy::NpyData;
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
    let wl_grid = WlGrid::Logspace(3.6020599913, 2e-6, 76_145);
    let interpolator = GridInterpolator::new(
        CachedFetcher::new("/STER/hermesnet/hermes_norm_convolved_u16", false, 3000, 1)?,
        wl_grid,
    );
    let continuum_interpolator = GridInterpolator::new(
        CachedFetcher::new(
            "/STER/hermesnet/hermes_continuum_convolved_u16",
            true,
            50_000,
            4,
        )?,
        wl_grid,
    );

    let wl = read_npy_file("wl_hermes.npy".into())?;
    let flux = read_npy_file("flux.npy".into())?.map(|x| x as f32);
    let var = read_npy_file("var.npy".into())?.map(|x| x as f32);
    let spec = ObservedSpectrum::from_vecs(flux.data.as_vec().clone(), var.data.as_vec().clone());
    let continuum_fitter = ChunkFitter::new(wl.clone(), 10, 5, 0.2);
    let settings = PSOSettings {
        num_particles: 50,
        max_iters: 10,
        inertia_factor: -0.3085,
        cognitive_factor: 0.0,
        social_factor: 2.0273,
        delta: 1e-7,
    };
    let dispersion = NoConvolutionDispersionTarget(wl);
    // let fitter = BinaryFitter::new(
    //     dispersion,
    //     continuum_fitter,
    //     settings,
    //     (0.0, 600.0),
    //     (-150.0, 150.0),
    // );
    let fitter = BinaryFitter::new(
        dispersion,
        continuum_fitter,
        settings,
        (1.0, 600.0),
        (-150.0, 150.0),
    );

    let constraints = vec![
        BoundsConstraint {
            parameter: 4,
            constraint: Constraint::Fixed(0.0),
        },
        BoundsConstraint {
            parameter: 9,
            constraint: Constraint::Fixed(50.0),
        },
    ];

    let start = Instant::now();
    // fitter.fit(&interpolator, &continuum_interpolator, &spec, None, true);
    fitter.fit(
        &interpolator,
        &continuum_interpolator,
        &spec,
        None,
        false,
        constraints,
    )?;
    println!("Time: {:?}", start.elapsed());
    Ok(())
}
