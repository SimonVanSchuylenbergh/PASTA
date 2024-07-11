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
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Tensor;
use convolve_rv::{oaconvolve, rot_broad_rv, WavelengthDispersion};
use cubic::{cubic_3d, prepare_interpolate, InterpolInput};
use fitting::{fit_pso, uncertainty_chi2, ChunkFitter, ContinuumFitter};
use interpolate::{
    nalgebra_to_tensor, read_npy_file, tensor_to_nalgebra, Interpolator, Range,
    SquareGridInterpolator, WlGrid,
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
    tch::set_num_threads(1);
    type Backend = burn::backend::LibTorch;
    let device = <Backend as burn::prelude::Backend>::Device::Cpu;
    // type Backend = burn::backend::Wgpu;
    // let device = burn::backend::wgpu::WgpuDevice::BestAvailable;
    let n = 1;

    let start = Instant::now();
    let folder = "/Users/ragnar/Documents/hermesnet/boss_grid_with_continuum";
    let wl_grid = WlGrid::Logspace(3.5440680443502757, 5.428_681_023_790_647e-6, 87508);
    let interpolator = InMemInterpolator::new(
        folder,
        wl_grid,
        Range::new(arange(25_250.0..30_250.0, 250.0).collect()),
        Range::new(arange(-0.2..0.2, 0.1).collect()),
        Range::new(arange(4.0..5.0, 0.1).collect()),
        (1.0, 300.0),
        (-150.0, 150.0),
        device,
    );
    println!("Instantiated in: {:?}", start.elapsed());

    let wl = read_npy_file("wl.npy".into())?;
    let disp = read_npy_file("disp.npy".into())?;
    let target_dispersion =
        convolve_rv::VariableTargetDispersion::new(wl.clone().into(), &disp.into(), wl_grid)?;
    let flux = read_npy_file("flux.npy".into())?;
    let var = read_npy_file("var.npy".into())?;
    let observed_spectrum = ObservedSpectrum {
        flux: flux.into(),
        var: var.into(),
    };
    let continuum_fitter = ChunkFitter::new(wl.into(), 10, 8, 0.2);

    let teff = 27000.0;
    let m = 0.0;
    let logg = 4.5;
    let vsini = 100.0;
    let rv = 0.0;

    let start = Instant::now();
    for _ in 0..n {
        let _ = interpolator.produce_model(&target_dispersion, teff, m, logg, vsini, rv);
    }
    println!("produce_model: {:?}", start.elapsed() / n);

    let start = Instant::now();
    for _ in 0..n {
        let _ = interpolator.interpolate(teff, m, logg);
    }
    println!("  interpolate: {:?}", start.elapsed() / n);

    let start = Instant::now();
    let InterpolInput {
        factors,
        local_4x4x4_indices,
        shape,
    } = prepare_interpolate::<Backend>(
        interpolator.ranges(),
        teff,
        m,
        logg,
        Interpolator::device(&interpolator),
    )?;
    let vec_of_tensors = local_4x4x4_indices
        .into_iter()
        .map(|[i, j, k]| {
            interpolator
                .find_spectrum(i as usize, j as usize, k as usize)
                .unwrap()
                .into_owned()
        })
        .collect::<Vec<Tensor<Backend, 1>>>();
    for _ in 0..n {
        // (4, 4, 4, N)
        let local_4x4x4 = Tensor::stack::<2>(vec_of_tensors.clone(), 0).reshape([4, 4, 4, -1]);
    }
    println!("      stacking: {:?}", start.elapsed() / n);

    let InterpolInput {
        factors,
        local_4x4x4_indices,
        shape,
    } = prepare_interpolate(
        interpolator.ranges(),
        teff,
        m,
        logg,
        Interpolator::device(&interpolator),
    )?;
    let vec_of_tensors = local_4x4x4_indices
        .into_iter()
        .map(|[i, j, k]| {
            interpolator
                .find_spectrum(i as usize, j as usize, k as usize)
                .unwrap()
                .into_owned()
        })
        .collect::<Vec<Tensor<Backend, 1>>>();
    // (4, 4, 4, N)
    let local_4x4x4 = Tensor::stack::<2>(vec_of_tensors, 0).reshape([4, 4, 4, -1]);
    let start = Instant::now();
    for _ in 0..n {
        let _ = cubic_3d(factors.clone(), shape, local_4x4x4.clone());
    }
    println!("      interpolate: {:?}", start.elapsed() / n);

    let interpolated = tensor_to_nalgebra::<Backend, f64>(interpolator.interpolate(teff, m, logg)?);
    let start = Instant::now();
    for _ in 0..n {
        let _ = rot_broad_rv(interpolated.clone(), wl_grid, &target_dispersion, vsini, rv);
    }
    println!("  rot_broad_rv: {:?}", start.elapsed() / n);

    let start = Instant::now();
    for _ in 0..n {
        let _ = target_dispersion.convolve(interpolated.clone());
    }
    println!("      convolve resolution: {:?}", start.elapsed() / n);

    let convolved_for_resolution = target_dispersion.convolve(interpolated.clone())?;
    let start = Instant::now();
    for _ in 0..n {
        let _ = oaconvolve(&convolved_for_resolution, vsini, wl_grid);
    }
    println!("      convolve vsini: {:?}", start.elapsed() / n);

    let synth_spec = tensor_to_nalgebra::<Backend, f64>(interpolator.produce_model(
        &target_dispersion,
        teff,
        m,
        logg,
        vsini,
        rv,
    )?);
    let start = Instant::now();
    for _ in 0..n {
        let _ = continuum_fitter.fit_continuum(&observed_spectrum, &synth_spec);
    }
    println!("fit continuum: {:?}", start.elapsed() / n);

    Ok(())
}
