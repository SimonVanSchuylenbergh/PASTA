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
use anyhow::Result;
use burn::tensor::Tensor;
use convolve_rv::{oaconvolve, rot_broad_rv, WavelengthDispersion};
use cubic::cubic_3d;
use fitting::{fit_pso, uncertainty_chi2, ChunkFitter, ContinuumFitter};
use interpolate::{
    nalgebra_to_tensor, prepare_interpolate, read_npy_file, tensor_to_nalgebra, InterpolInput,
    Interpolator, Range, SquareGridInterpolator, WlGrid,
};
use interpolators::{CachedInterpolator, InMemInterpolator, OnDiskInterpolator};
use iter_num_tools::arange;
use itertools::Itertools;
use nalgebra as na;
use rayon::prelude::*;
use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;

pub fn main() -> Result<()> {
    type Backend = burn::backend::LibTorch;

    let start = Instant::now();
    let folder = "/Users/ragnar/Documents/hermesnet/boss_grid_with_continuum";
    let wlGrid = WlGrid::Logspace(3.5440680443502757, 5.428_681_023_790_647e-6, 87508);
    let interpolator = InMemInterpolator::new(
        folder,
        wlGrid,
        Range::new(arange(25_250.0..30_250.0, 250.0).collect()),
        Range::new(arange(-0.2..0.2, 0.1).collect()),
        Range::new(arange(4.0..5.0, 0.1).collect()),
        (1.0, 300.0),
        (-150.0, 150.0),
    );
    println!("Instantiated in: {:?}", start.elapsed());

    let wl = read_npy_file("wl.npy".into())?;
    let disp = read_npy_file("disp.npy".into())?;
    let target_dispersion = convolve_rv::VariableTargetDispersion::new(wl.clone(), &disp, wlGrid)?;
    let flux = read_npy_file("flux.npy".into())?;
    let var = read_npy_file("var.npy".into())?;
    let observed_spectrum = ObservedSpectrum { flux, var };
    let continuum_fitter = ChunkFitter::new(wl, 10, 8, 0.2);

    let teff = 27000.0;
    let m = 0.0;
    let logg = 4.5;
    let vsini = 100.0;
    let rv = 0.0;

    let start = Instant::now();
    for _ in 0..100 {
        let _ = Interpolator::<Backend>::produce_model(
            &interpolator,
            &target_dispersion,
            teff,
            m,
            logg,
            vsini,
            rv,
        );
    }
    println!("produce_model: {:?}", start.elapsed() / 100);

    let start = Instant::now();
    for _ in 0..100 {
        let _ = Interpolator::<Backend>::interpolate(&interpolator, teff, m, logg);
    }
    println!("  interpolate: {:?}", start.elapsed() / 100);

    let start = Instant::now();
    for _ in 0..100 {
        let InterpolInput {
            coord,
            xp,
            local_4x4x4_indices,
            shape,
        } = prepare_interpolate(interpolator.ranges(), teff, m, logg)?;
        let vec_of_tensors = local_4x4x4_indices
            .into_iter()
            .map(|[i, j, k]| {
                nalgebra_to_tensor(
                    interpolator
                        .find_spectrum(i as usize, j as usize, k as usize)
                        .unwrap()
                        .into_owned(),
                )
            })
            .collect::<Vec<Tensor<Backend, 1>>>();
        // (4, 4, 4, N)
        let local_4x4x4 = Tensor::stack::<2>(vec_of_tensors, 0).reshape([4, 4, 4, -1]);
    }
    println!("      prepare: {:?}", start.elapsed() / 100);

    let InterpolInput {
        coord,
        xp,
        local_4x4x4_indices,
        shape,
    } = prepare_interpolate(interpolator.ranges(), teff, m, logg)?;
    let vec_of_tensors = local_4x4x4_indices
        .into_iter()
        .map(|[i, j, k]| {
            nalgebra_to_tensor(
                interpolator
                    .find_spectrum(i as usize, j as usize, k as usize)
                    .unwrap()
                    .into_owned(),
            )
        })
        .collect::<Vec<Tensor<Backend, 1>>>();
    // (4, 4, 4, N)
    let local_4x4x4 = Tensor::stack::<2>(vec_of_tensors, 0).reshape([4, 4, 4, -1]);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = cubic_3d(coord, &xp, local_4x4x4.clone(), shape);
    }
    println!("      interpolate: {:?}", start.elapsed() / 100);

    let interpolated = tensor_to_nalgebra::<Backend, f64>(interpolator.interpolate(teff, m, logg)?);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = rot_broad_rv(interpolated.clone(), wlGrid, &target_dispersion, vsini, rv);
    }
    println!("  rot_broad_rv: {:?}", start.elapsed() / 100);

    let start = Instant::now();
    for _ in 0..100 {
        let _ = target_dispersion.convolve(interpolated.clone());
    }
    println!("convolve resolution: {:?}", start.elapsed() / 100);

    let convolved_for_resolution = target_dispersion.convolve(interpolated.clone())?;
    let start = Instant::now();
    for _ in 0..100 {
        let _ = oaconvolve(&convolved_for_resolution, vsini, wlGrid);
    }
    println!("convolve vsini: {:?}", start.elapsed() / 100);

    let synth_spec = tensor_to_nalgebra::<Backend, f64>(interpolator.produce_model(
        &target_dispersion,
        teff,
        m,
        logg,
        vsini,
        rv,
    )?);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = continuum_fitter.fit_continuum(&observed_spectrum, &synth_spec);
    }
    println!("fit continuum: {:?}", start.elapsed() / 100);

    Ok(())
}
