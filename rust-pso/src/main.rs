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
use convolve_rv::{oaconvolve, rot_broad_rv, WavelengthDispersion};
use fitting::{fit_pso, uncertainty_chi2, ChunkFitter, ContinuumFitter};
use interpolate::{
    read_npy_file, tensor_to_nalgebra, Interpolator, Range, SquareGridInterpolator, WlGrid,
};
use interpolators::{CachedInterpolator, InMemInterpolator, OnDiskInterpolator};
use iter_num_tools::arange;
use itertools::Itertools;
use nalgebra as na;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

pub fn main() -> Result<()> {


    Ok(())
}
