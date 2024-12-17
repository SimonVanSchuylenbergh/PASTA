#![allow(unused)]
use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra as na;
use npy::NpyData;
use pasta::convolve_rv::{
    convolve_rotation, shift_and_resample, NoConvolutionDispersionTarget, WavelengthDispersion,
};
use pasta::interpolate::{GridInterpolator, Interpolator, WlGrid};
use pasta::model_fetchers::InMemFetcher;
use rayon::prelude::*;
use std::io::Read;
use std::path::PathBuf;

const SMALL_GRID_PATH: &str = "/Users/ragnar/Documents/hermesnet/hermes_norm_convolved_u16_small";

pub fn read_npy_file(file_path: PathBuf) -> Result<na::DVector<f64>> {
    let mut file = std::fs::File::open(file_path.clone())?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let data: NpyData<f64> = NpyData::from_bytes(&buf)?;
    Ok(na::DVector::from_iterator(data.len(), data))
}

pub fn benchmark(c: &mut Criterion) {
    let wl_grid = WlGrid::Logspace(3.6020599913, 2e-6, 76_145);
    let interpolator = GridInterpolator::new(
        InMemFetcher::new(
            "/Users/ragnar/Documents/hermesnet/hermes_norm_convolved_u16_small",
            false,
        )
        .unwrap(),
        wl_grid.clone(),
    );
    let wl = read_npy_file("wl_hermes.npy".into()).unwrap();
    let dispersion = NoConvolutionDispersionTarget::new(wl.clone(), &wl_grid);
    let interpolated = interpolator.interpolate(8000.0, 0.0, 3.5).unwrap();
    let convolved_for_rotation = convolve_rotation(&wl_grid, &interpolated, 20.0).unwrap();
    let model = dispersion.convolve(convolved_for_rotation.clone()).unwrap();
    let output = shift_and_resample(&wl_grid, &model, &dispersion, 1.0).unwrap();
    c.bench_function("produce_model", |b| {
        b.iter(|| {
            interpolator
                .produce_model(&dispersion, 8000.0, 0.0, 3.5, 20.0, 1.0)
                .unwrap()
        })
    });
    c.bench_function("interpolate", |b| {
        b.iter(|| interpolator.interpolate(8000.0, 0.0, 3.5).unwrap())
    });
    c.bench_function("convolve_rotation", |b| {
        b.iter(|| convolve_rotation(&wl_grid, &interpolated, 20.0).unwrap())
    });
    c.bench_function("convolve resolution", |b| {
        b.iter(|| dispersion.convolve(convolved_for_rotation.clone()).unwrap())
    });
    c.bench_function("resample", |b| {
        b.iter(|| shift_and_resample(&wl_grid, &model, &dispersion, 1.0).unwrap())
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
