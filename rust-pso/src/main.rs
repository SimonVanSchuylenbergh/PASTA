#![allow(unused_imports)]
#![allow(dead_code)]

mod convolve_rv;
mod cubic;
mod fitting;
mod interpolate;
mod interpolators;
mod particleswarm;
use crate::fitting::ObservedSpectrum;
use crate::interpolate::{Bounds, CompoundInterpolator};
use fitting::{fit_pso, uncertainty_chi2, ChunkFitter, ContinuumFitter};
use interpolate::{read_npy_file, Interpolator, Range, SquareGridInterpolator, WlGrid};
use interpolators::{CachedInterpolator, InMemInterpolator, OnDiskInterpolator};
use iter_num_tools::arange;
use itertools::Itertools;
use nalgebra as na;
use rayon::prelude::*;
use std::time::Instant;
use anyhow::Result;

pub fn main() -> Result<()> {
    let start = Instant::now();
    // "/Users/ragnar/Documents/hermesnet/fine_grid",
    let folder = "/STER/hermesnet/fine_grid";
    let interpolator1 = CachedInterpolator::new(
        folder,
        WlGrid::Linspace(4000.0, 0.02, 84001),
        Range::new(arange(25_250.0..30_250.0, 250.0).collect()),
        Range::new(arange(-0.8..0.9, 0.1).collect()),
        Range::new(arange(3.3..5.1, 0.1).collect()),
        (1.0, 300.0),
        (-150.0, 150.0),
        8000,
    );

    let interpolator2 = CachedInterpolator::new(
        folder,
        WlGrid::Linspace(4000.0, 0.02, 84001),
        Range::new(
            arange(9800.0..10_000.0, 100.0)
                .chain(arange(10_000.0..26_000.0, 250.0))
                .collect(),
        ),
        Range::new(arange(-0.8..0.9, 0.1).collect()),
        Range::new(arange(3.0..5.1, 0.1).collect()),
        (1.0, 300.0),
        (-150.0, 150.0),
        8000,
    );

    let interpolator3 = CachedInterpolator::new(
        folder,
        WlGrid::Linspace(4000.0, 0.02, 84001),
        Range::new(arange(6000.0..10_100.0, 100.0).collect()),
        Range::new(arange(-0.8..0.9, 0.1).collect()),
        Range::new(arange(2.5..5.1, 0.1).collect()),
        (1.0, 300.0),
        (-150.0, 150.0),
        8000,
    );
    let interpolator = CompoundInterpolator::new(interpolator1, interpolator2, interpolator3);
    println!("Instantiated in: {:?}", start.elapsed());

    let wl = read_npy_file("wl.npy".into())?;
    let disp = read_npy_file("disp.npy".into())?;

    let target_dispersion = convolve_rv::NoDispersionTarget(wl.clone());
    let start = Instant::now();
    for _ in 0..100 {
        let _ = interpolator.produce_model(&target_dispersion, 8000.0, 0.0, 3.7, 5.0, 0.0);
    }
    println!("Base: {:?}", start.elapsed());

    let start = Instant::now();
    let target_dispersion = convolve_rv::VariableTargetDispersion::new(wl, &disp, WlGrid::Linspace(4_000.0, 0.02, 84001))?;
    println!("Constructing kernels: {:?}", start.elapsed());
    let start = Instant::now();
    for _ in 0..100 {
        let _ = interpolator.produce_model(&target_dispersion, 8000.0, 0.0, 3.7, 5.0, 0.0);
    }
    println!("Variable: {:?}", start.elapsed());

    // let result = uncertainty_chi2(&interpolator, &obs_wl, &obs, &fitter, label, radius)?;
    // for p in result.into_iter() {
    //     println!("Result: {:?}", p?);
    // }


    // let interpolator = InMemInterpolator::new(
    //         "/STER/hermesnet/fine_grid",
    //         // "/Users/ragnar/Documents/hermesnet/fine_grid",
    //         (4000.0, 0.02),
    //         Range::new(arange(25_250.0..30_250.0, 250.0).collect()),
    //         Range::new(arange(-0.2..0.2, 0.1).collect()),
    //         Range::new(arange(3.8..4.2, 0.1).collect()),
    //         (1.0, 300.0),
    //         (-150.0, 150.0),
    //     );

    // let start = Instant::now();
    // for _ in 0..200 {
    //     let _ = interpolator.interpolate(28_000.0, 0.0, 4.0).unwrap();
    // }
    // println!("Elapsed time: {:?}", start.elapsed());

    // let start = Instant::now();
    // let result = fit_pso_best(
    //     &interpolator,
    //     &obs_wl,
    //     &obs,
    //     &fitter,
    //     32,
    //     Some(500),
    //     None,
    //     None,
    //     None,
    //     Some(1.0),
    //     None,
    //     true,
    // )
    // .unwrap();
    // (14..40).for_each(|t| {
    //     (-14..14).for_each(|m| {
    //         let teff = t as f64 * 500.0;
    //         let m = m as f64 * 0.05;
    //         interpolator.produce_model(&obs_wl, teff, m, 4.0, 100.0, 0.0);
    //     });
    // });
    // println!("Elapsed time: {:?}", start.elapsed());
    // println!("Result: {:?}", result.labels);

    // let result = fit_pso_best(&interpolator, &obs_wl, &obs, &fitter, 16, Some(100), None, true).unwrap();

    // // let fit = fitter._fit_continuum(&cont);
    // // let cont = fitter.build_continuum(fit);

    // let fitter = SimonFitter::new(&obs_wl, 5, 8, 0.2);

    // // interpolator.interpolate(15_000.0, 0.0, 4.0).unwrap();
    // let vsini = 300.0;
    // // let start = Instant::now();
    // // for _ in 0..300 {
    // //     convolve_rv::convolve(&i, vsini);
    // // }
    // // let t1 = start.elapsed();
    // // println!("Regular: {:?}", t1);
    // let start = Instant::now();
    // (0..2000).into_par_iter().for_each(|_| {
    //     let i = interpolator.interpolate(15_000.0, 0.0, 4.0).unwrap();
    //     // convolve_rv::oaconvolve(&i, vsini);
    //     let s = convolve_rv::rot_broad_rv(&i, interpolator.synth_wl, &obs_wl, vsini, 0.0);
    //     let _ = fitter.fit_continuum(&obs, &s);
    // });
    // let t2 = start.elapsed();
    // println!("oa: {:?}", t2);

    // let fitter = SimonFitter::new(&obs_wl, 5, 8, 0.2);
    // let start = Instant::now();
    // let params: Vec<_> = (7..20).cartesian_product(-7..7).collect();
    // params.into_iter().for_each(|(t, m)| {
    //     let teff = t as f64 * 1000.0;
    //     let m = m as f64 * 0.1;
    //     let synth = interpolator
    //         .produce_model(&obs_wl, teff, m, 4.0, 100.0, 0.0)
    //         .unwrap();
    //     // let synth = interpolator
    //     //     .interpolate(teff, m, 4.0)
    //     //     .unwrap();
    //     // let _ = fitter.fit_continuum(&obs, &synth);
    // });
    // println!("Elapsed time: {:?}", start.elapsed());

    // println!("Result: {:?}", result.labels);
    Ok(())
}
