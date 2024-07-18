#![allow(unused_imports)]
#![allow(dead_code)]

mod convolve_rv;
mod cubic;
mod fitting;
mod interpolate;
mod model_fetchers;
mod particleswarm;
mod tensor;
use crate::fitting::ObservedSpectrum;
use crate::interpolate::{Bounds, CompoundInterpolator};
use crate::tensor::Tensor;
use anyhow::Result;
use convolve_rv::{oaconvolve, rot_broad_rv, WavelengthDispersion};
use cubic::{calculate_interpolation_coefficients, calculate_interpolation_coefficients_flat};
use fitting::{fit_pso, uncertainty_chi2, ChunkFitter, ContinuumFitter};
use interpolate::{
    nalgebra_to_tensor, read_npy_file, tensor_to_nalgebra, Interpolator, ModelFetcher, Range,
    SquareBounds, SquareGridInterpolator, WlGrid,
};
use iter_num_tools::arange;
use itertools::Itertools;
use model_fetchers::{CachedFetcher, InMemFetcher};
use nalgebra as na;
use rand::distributions::Standard;
use rayon::prelude::*;
use std::borrow::Cow;
use std::hint::black_box;
use std::iter::repeat;
use std::time::Instant;

const teff: f64 = 27000.0;
const m: f64 = 0.05;
const logg: f64 = 4.5;

fn convolve1(flux_arr: na::DVector<f32>, kernels: &na::DMatrix<f32>, n: usize) -> na::DVector<f32> {
    na::DVector::from_fn(flux_arr.len(), |i, _| {
        let kernel = kernels.column(i);
        if (i < n) || (i >= kernels.ncols() - n) {
            return flux_arr[i];
        }
        let fluxs = flux_arr.rows(i - n, 2 * n + 1);
        kernel.dot(&fluxs)
    })
}

fn convolve2(flux_arr: &Tensor, kernels: &Tensor, n: i64) -> Tensor {
    let padded = flux_arr.tensor.pad([n, n], "constant", Some(0_f64));
    let unfolded = padded.unfold(0, 2 * n + 1, 1);

    let out = (unfolded * kernels.clone().tensor).sum_dim_intlist(&[0_i64][..], false, None);
    Tensor { tensor: out }
}

const N: usize = 500;

fn convolve3(flux_arr: na::DVector<f32>, kernels: &na::DMatrix<f32>, n: usize) -> na::DVector<f32> {
    // Herschrijf naar batches
    let mut out = na::DVector::zeros(flux_arr.len());
    let mut mat = na::DMatrix::<f32>::zeros(N, 2 * n + 1);
    for i in 0..(flux_arr.len() / N) {
        let start = i * N;
        for j in 0..N {
            mat.row_mut(j)
                .copy_from(&flux_arr.rows(start + j - n, 2 * n + 1));
        }
        mat.mul_to(&kernels.columns(start, N), &mut out.rows_mut(start, N))
    }
    let start = (flux_arr.len() / N) * N;
    let remaining = flux_arr.len() - start;
    let mut mat = na::DMatrix::<f32>::zeros(remaining, 2 * n + 1);
    for j in 0..remaining {
        mat.row_mut(j)
            .copy_from(&flux_arr.rows(start + j - n, 2 * n + 1));
    }
    mat.mul_to(
        &kernels.columns(start, remaining),
        &mut out.rows_mut(start, remaining),
    );
    out
}

fn convolve4(
    flux_arr: na::DVector<f32>,
    kernels_tr: &na::DMatrix<f32>,
    n: usize,
) -> na::DVector<f32> {
    let n = flux_arr.len();
    let flux_arr_padded = na::DVector::from_iterator(
        flux_arr.len() + 2 * n,
        repeat(0.0)
            .take(n)
            .chain(flux_arr.iter().cloned())
            .chain(repeat(0.0).take(n)),
    );
    kernels_tr.column_iter().enumerate().fold(
        na::DVector::zeros(flux_arr.len()),
        |mut acc, (i, kernel)| {
            let fluxs = flux_arr_padded.rows(i, n);
            acc + (kernel.component_mul(&fluxs))
        },
    )
}

pub fn main() -> Result<()> {
    tch::set_num_threads(1);
    let device = tch::Device::Cpu;

    let folder = "/STER/hermesnet/boss_grid_with_continuum";
    let wl_grid = WlGrid::Logspace(3.5440680443502757, 5.428_681_023_790_647e-6, 87508);
    let fetcher = InMemFetcher::new(
        folder,
        Range::new(arange(25_250.0..30_250.0, 250.0).collect()),
        Range::new(arange(-0.2..0.2, 0.1).collect()),
        Range::new(arange(4.0..5.0, 0.1).collect()),
        (1.0, 300.0),
        (-150.0, 150.0),
        device,
    )?;
    let interpolator = SquareGridInterpolator::new(fetcher, wl_grid);
    let device = interpolator.fetcher.device().clone();
    let interpolated_tensor = interpolator.interpolate(teff, m, logg)?;
    let interpolated_na = tensor_to_nalgebra(interpolated_tensor.clone());

    let wl = read_npy_file("wl.npy".into())?;
    let disp = read_npy_file("disp.npy".into())?
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    let target_dispersion =
        convolve_rv::VariableTargetDispersion::new(wl.clone().into(), &disp.into(), wl_grid)?;
    let kernels_na = target_dispersion.kernels;
    let kernels_na_tr = kernels_na.transpose();
    let kernels_flat = na::DVector::from_iterator(
        kernels_na.nrows() * kernels_na.ncols(),
        kernels_na.iter().cloned(),
    );
    let kernels_tensor = nalgebra_to_tensor(kernels_flat, &device)
        .reshape([kernels_na.nrows() as i64, kernels_na.ncols() as i64])?
        .transpose(0, 1);
    let n = target_dispersion.n;
    println!("n: {}", n);

    let is = (0..44 * 8).collect::<Vec<usize>>();

    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve1: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve1: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve1: {:?}", start.elapsed());
    println!("convolve1: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve1: {:?}", start.elapsed());
    println!("convolve1: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve1: {:?}", start.elapsed());
    println!("");

    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve2(&interpolated_tensor, &kernels_tensor, n as i64);
    });
    println!("convolve2: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve2(&interpolated_tensor, &kernels_tensor, n as i64);
    });
    println!("convolve2: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve2(&interpolated_tensor, &kernels_tensor, n as i64);
    });
    println!("convolve2: {:?}", start.elapsed());
    println!("");

    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve3: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve3: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve3: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve3: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve1(interpolated_na.clone(), &kernels_na, n);
    });
    println!("convolve3: {:?}", start.elapsed());
    println!("");

    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve4(interpolated_na.clone(), &kernels_na_tr, n);
    });
    println!("convolve4: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve4(interpolated_na.clone(), &kernels_na_tr, n);
    });
    println!("convolve4: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve4(interpolated_na.clone(), &kernels_na_tr, n);
    });
    println!("convolve4: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve4(interpolated_na.clone(), &kernels_na_tr, n);
    });
    println!("convolve4: {:?}", start.elapsed());
    let start = Instant::now();
    is.par_iter().for_each(|_| {
        convolve4(interpolated_na.clone(), &kernels_na_tr, n);
    });
    println!("convolve4: {:?}", start.elapsed());
    println!("");
    Ok(())
}
