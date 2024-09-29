#![allow(unused_imports)]
#![allow(dead_code)]

mod convolve_rv;
mod cubic;
mod fitting;
mod interpolate;
mod model_fetchers;
mod particleswarm;
use crate::fitting::ObservedSpectrum;
use crate::interpolate::{GridBounds, CompoundInterpolator, SquareBounds};
use anyhow::Result;
use convolve_rv::{
    oa_convolve, rot_broad_rv, NoConvolutionDispersionTarget, VariableTargetDispersion,
    WavelengthDispersion,
};
use cubic::{calculate_interpolation_coefficients, calculate_interpolation_coefficients_flat};
use fitting::{fit_pso, uncertainty_chi2, ChunkFitter, ContinuumFitter, PSOSettings};
use interpolate::{read_npy_file, Interpolator, Range, SquareGridInterpolator, WlGrid};
use iter_num_tools::arange;
use itertools::Itertools;
use model_fetchers::InMemFetcher;
use nalgebra as na;
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
    let start = Instant::now();
    let folder = "/STER/hermesnet/boss_grid_with_continuum_32";
    let wl_grid = WlGrid::Logspace(3.5440680443502757, 5.428681023790647e-06, 87508);
    // let fetcher = InMemFetcher::new(
    //     folder,
    //     Range::new(arange(25_250.0..30_250.0, 250.0).collect()),
    //     Range::new(arange(-0.2..0.2, 0.1).collect()),
    //     Range::new(arange(4.0..5.0, 0.1).collect()),
    //     (1.0, 300.0),
    //     (-150.0, 150.0),
    // )?;
    // println!("Instantiated in: {:?}", start.elapsed());
    // let interpolator = SquareGridInterpolator::new(fetcher, wl_grid);

    let interpolator1 = SquareGridInterpolator::new(
        InMemFetcher::new(
            folder,
            Range::new(arange(25_250.0..30_250.0, 250.0).collect()),
            Range::new(arange(-0.8..0.9, 0.1).collect()),
            Range::new(arange(3.3..5.1, 0.1).collect()),
            (1.0, 300.0),
            (-150.0, 150.0),
        )?,
        wl_grid,
    );

    let interpolator2 = SquareGridInterpolator::new(
        InMemFetcher::new(
            folder,
            Range::new(
                arange(9800.0..10_000.0, 100.0)
                    .chain(arange(10_000.0..26_000.0, 250.0))
                    .collect(),
            ),
            Range::new(arange(-0.8..0.9, 0.1).collect()),
            Range::new(arange(3.0..5.1, 0.1).collect()),
            (1.0, 300.0),
            (-150.0, 150.0),
        )?,
        wl_grid,
    );

    let interpolator3 = SquareGridInterpolator::new(
        InMemFetcher::new(
            folder,
            Range::new(arange(6000.0..10_100.0, 100.0).collect()),
            Range::new(arange(-0.8..0.9, 0.1).collect()),
            Range::new(arange(2.5..5.1, 0.1).collect()),
            (1.0, 300.0),
            (-150.0, 150.0),
        )?,
        wl_grid,
    );
    let interpolator = CompoundInterpolator::new(interpolator1, interpolator2, interpolator3);

    let wl = read_npy_file::<f64>("wl.npy".into())?;
    let disp = read_npy_file::<f64>("disp.npy".into())?
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>();
    let target_dispersion =
        VariableTargetDispersion::new(wl.clone().into(), &disp.into(), wl_grid)?;
    // let target_dispersion = NoDispersionTarget(wl.clone().into());
    let flux = read_npy_file::<f64>("flux.npy".into())?
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>();
    let var = read_npy_file::<f64>("var.npy".into())?
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>();
    let observed_spectrum = ObservedSpectrum::from_vecs(flux, var);
    let observed_spectra = (0..44)
        .map(|_| observed_spectrum.clone())
        .collect::<Vec<_>>();
    let continuum_fitter = ChunkFitter::new(wl.into(), 5, 8, 0.2);
    let settings = PSOSettings {
        num_particles: 44,
        max_iters: 100,
        inertia_factor: 0.7213475204,
        cognitive_factor: 1.193180596,
        social_factor: 0.5,
        delta: 1e-7,
    };

    let is = (0..44 * 100).collect::<Vec<usize>>();
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = interpolator.interpolate(teff, m, logg).unwrap();
    // });
    // println!("interpolate: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = interpolator.interpolate(teff, m, logg).unwrap();
    // });
    // println!("interpolate: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = interpolator.interpolate(teff, m, logg).unwrap();
    // });
    // println!("interpolate: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = interpolator.interpolate(teff, m, logg).unwrap();
    // });
    // println!("interpolate: {:?}", start.elapsed());

    // let interpolated = interpolator.interpolate(teff, m, logg)?;

    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = target_dispersion.convolve(interpolated.clone()).unwrap();
    // });
    // println!("convolve res: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = target_dispersion.convolve(interpolated.clone()).unwrap();
    // });
    // println!("convolve res: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = target_dispersion.convolve(interpolated.clone()).unwrap();
    // });
    // println!("convolve res: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = target_dispersion.convolve(interpolated.clone()).unwrap();
    // });
    // println!("convolve res: {:?}", start.elapsed());

    // let convolved_for_resolution = target_dispersion.convolve(interpolated.clone())?;

    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = oaconvolve(&convolved_for_resolution, vsini, wl_grid);
    // });
    // println!("convolve vsini: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = oaconvolve(&convolved_for_resolution, vsini, wl_grid);
    // });
    // println!("convolve vsini: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = oaconvolve(&convolved_for_resolution, vsini, wl_grid);
    // });
    // println!("convolve vsini: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = oaconvolve(&convolved_for_resolution, vsini, wl_grid);
    // });
    // println!("convolve vsini: {:?}", start.elapsed());

    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = interpolator
    //         .produce_model(&target_dispersion, teff, m, logg, vsini, rv)
    //         .unwrap();
    // });
    // println!("produce_model: {:?}", start.elapsed());
    // let start = Instant::now();
    // is.par_iter().for_each(|_| {
    //     let y = interpolator
    //         .produce_model(&target_dispersion, teff, m, logg, vsini, rv)
    //         .unwrap();
    // });
    // println!("produce_model: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        let y = interpolator
            .produce_model(&target_dispersion, teff, m, logg, vsini, rv)
            .unwrap();
    });
    println!("produce_model: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        let y = interpolator
            .produce_model(&target_dispersion, teff, m, logg, vsini, rv)
            .unwrap();
    });
    println!("produce_model: {:?}", start.elapsed());

    let model = interpolator.produce_model(&target_dispersion, teff, m, logg, vsini, rv)?;

    let start = Instant::now();
    is.par_iter().for_each(|_| {
        let y = continuum_fitter
            .fit_continuum(&observed_spectrum, &model)
            .unwrap();
    });
    println!("fit_continuum: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        let y = continuum_fitter
            .fit_continuum(&observed_spectrum, &model)
            .unwrap();
    });
    println!("fit_continuum: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        let y = continuum_fitter
            .fit_continuum(&observed_spectrum, &model)
            .unwrap();
    });
    println!("fit_continuum: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        let y = continuum_fitter
            .fit_continuum(&observed_spectrum, &model)
            .unwrap();
    });
    println!("fit_continuum: {:?}", start.elapsed());

    // let start = Instant::now();
    // let y = fit_pso(
    //     &interpolator,
    //     &target_dispersion,
    //     &observed_spectrum,
    //     &continuum_fitter,
    //     &settings,
    //     None,
    //     true,
    // )
    // .unwrap();
    // println!("fit_pso: {:?}", start.elapsed());
    // let start = Instant::now();
    // let y = fit_pso(
    //     &interpolator,
    //     &target_dispersion,
    //     &observed_spectrum,
    //     &continuum_fitter,
    //     &settings,
    //     None,
    //     true,
    // )
    // .unwrap();
    // println!("fit_pso: {:?}", start.elapsed());
    let start = Instant::now();
    let y = fit_pso(
        &interpolator,
        &target_dispersion,
        &observed_spectrum,
        &continuum_fitter,
        &settings,
        None,
        true,
    )
    .unwrap();
    println!("fit_pso: {:?}", start.elapsed());
    let start = Instant::now();
    let y = fit_pso(
        &interpolator,
        &target_dispersion,
        &observed_spectrum,
        &continuum_fitter,
        &settings,
        None,
        true,
    )
    .unwrap();
    println!("fit_pso: {:?}", start.elapsed());
    let start = Instant::now();
    let y = fit_pso(
        &interpolator,
        &target_dispersion,
        &observed_spectrum,
        &continuum_fitter,
        &settings,
        None,
        true,
    )
    .unwrap();
    println!("fit_pso: {:?}", start.elapsed());

    // let start = Instant::now();
    // observed_spectra.par_iter().for_each(|spec| {
    //     fit_pso(
    //         &interpolator,
    //         &target_dispersion,
    //         spec,
    //         &continuum_fitter,
    //         &settings,
    //         None,
    //         false,
    //     ).unwrap();
    // });
    // println!("fit_pso: {:?}", start.elapsed());
    Ok(())
}
