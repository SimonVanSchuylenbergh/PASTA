use crate::convolve_rv::{rot_broad_rv, WavelengthDispersion};
use crate::cubic::{cubic_3d, prepare_interpolate, InterpolInput};
use anyhow::{Context, Result};
use argmin_math::ArgminRandom;
use burn::prelude::Backend;
use burn::tensor::{Data, Element, Tensor};
use nalgebra::{self as na, Scalar};
use npy::NpyData;
use rand::Rng;
use std::borrow::Cow;
use std::io::Read;
use std::path::{Path, PathBuf};

pub fn read_npy_file(file_path: PathBuf) -> Result<Vec<f64>> {
    let mut file = std::fs::File::open(file_path.clone())
        .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let data: NpyData<f64> = NpyData::from_bytes(&buf)?;
    Ok(data.to_vec())
}

pub fn read_spectrum(dir: &Path, teff: f64, m: f64, logg: f64) -> Result<Vec<f64>> {
    // e.g. 00027_lm0050_07000_0350_0020_0000_Vsini_0000.npy
    let _teff = teff.round() as i32;
    let _m = (m * 100.0).round() as i32;
    let _logg = (logg * 100.0).round() as i32;
    let sign = if _m < 0 { "m" } else { "p" };
    let filename = format!("l{}{:04}_{:05}_{:04}", sign, _m.abs(), _teff, _logg);
    let file_path = dir.join(format!("{}.npy", filename));
    read_npy_file(file_path.clone())
}

#[derive(Clone)]
pub struct Range {
    pub values: Vec<f64>,
}

impl Range {
    pub fn new(values: Vec<f64>) -> Self {
        Range { values }
    }

    pub fn get_left_index(&self, x: f64) -> usize {
        match self.values.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            }
            Err(i) => i - 1,
        }
    }

    pub fn try_get_index(&self, x: f64) -> Option<usize> {
        let precision = 0.001;
        let i = self.get_left_index(x);
        let j = i + 1;
        if (self.values[i] - x).abs() < precision {
            Some(i)
        } else if (self.values[j] - x).abs() < precision {
            Some(j)
        } else {
            None
        }
    }

    pub fn between_bounds(&self, x: f64) -> bool {
        x >= self.values[0] && x <= *self.values.last().unwrap()
    }

    pub fn get_value(&self, i: usize) -> f64 {
        self.values[i]
    }

    pub fn n(&self) -> usize {
        self.values.len()
    }

    pub fn get_first_and_last(&self) -> (f64, f64) {
        (self.values[0], *self.values.last().unwrap())
    }
}

#[derive(Clone)]
pub struct SquareBounds {
    pub teff: Range,
    pub m: Range,
    pub logg: Range,
    pub vsini: (f64, f64),
    pub rv: (f64, f64),
}

pub trait Bounds: Clone {
    fn is_within_bounds(&self, param: na::SVector<f64, 5>) -> bool;
    fn clamp(&self, param: na::SVector<f64, 5>) -> na::SVector<f64, 5>;
    fn clamp_1d(&self, param: na::SVector<f64, 5>, index: usize) -> f64;
    fn minmax(&self) -> (na::SVector<f64, 5>, na::SVector<f64, 5>);
    fn generate_random_within_bounds(
        &self,
        rng: &mut impl Rng,
        num_particles: usize,
    ) -> Vec<na::SVector<f64, 5>>;
}

#[derive(Clone, Copy, Debug)]
pub enum WlGrid {
    Linspace(f64, f64, usize), // first, step, total
    Logspace(f64, f64, usize), // log10(first), step log10, total
}

impl WlGrid {
    pub fn n(&self) -> usize {
        match self {
            WlGrid::Linspace(_, _, n) => *n,
            WlGrid::Logspace(_, _, n) => *n,
        }
    }

    pub fn get_first_and_last(&self) -> (f64, f64) {
        match self {
            WlGrid::Linspace(first, step, total) => (*first, *first + step * *total as f64),
            WlGrid::Logspace(first, step, total) => (
                10_f64.powf(*first),
                10_f64.powf(*first + step * *total as f64),
            ),
        }
    }

    pub fn get_float_index_of_wl(&self, wl: f64) -> f64 {
        match self {
            WlGrid::Linspace(first, step, _) => (wl - first) / step,
            WlGrid::Logspace(first, step, _) => (wl.log10() - first) / step,
        }
    }

    pub fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            WlGrid::Linspace(first, step, total) => {
                Box::new((0..*total).map(move |i| first + step * i as f64))
            }
            WlGrid::Logspace(first, step, total) => {
                Box::new((0..*total).map(move |i| 10_f64.powf(first + step * i as f64)))
            }
        }
    }
}

pub trait Interpolator: Send + Sync {
    type B: Bounds;
    type E: Backend;

    fn synth_wl(&self) -> WlGrid;
    fn bounds(&self) -> &Self::B;
    fn device(&self) -> &<Self::E as Backend>::Device;
    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<Tensor<Self::E, 1>>;
    fn produce_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<Tensor<Self::E, 1>>;
    fn produce_model_on_grid(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<Tensor<Self::E, 1>>;
}

impl Bounds for SquareBounds {
    fn is_within_bounds(&self, param: na::SVector<f64, 5>) -> bool {
        self.teff.between_bounds(param[0])
            && self.m.between_bounds(param[1])
            && self.logg.between_bounds(param[2])
            && param[3] >= self.vsini.0
            && param[3] <= self.vsini.1
            && param[4] >= self.rv.0
            && param[4] <= self.rv.1
    }
    fn clamp(&self, param: na::SVector<f64, 5>) -> na::SVector<f64, 5> {
        let mut new_param = param;

        let teff_bounds = self.teff.get_first_and_last();
        let m_bounds = self.m.get_first_and_last();
        let logg_bounds = self.logg.get_first_and_last();

        new_param[0] = new_param[0].max(teff_bounds.0).min(teff_bounds.1);
        new_param[1] = new_param[1].max(m_bounds.0).min(m_bounds.1);
        new_param[2] = new_param[2].max(logg_bounds.0).min(logg_bounds.1);
        new_param[3] = new_param[3].max(self.vsini.0).min(self.vsini.1);
        new_param[4] = new_param[4].max(self.rv.0).min(self.rv.1);

        new_param
    }

    fn clamp_1d(&self, param: na::SVector<f64, 5>, index: usize) -> f64 {
        let bound = match index {
            0 => self.teff.get_first_and_last(),
            1 => self.m.get_first_and_last(),
            2 => self.logg.get_first_and_last(),
            3 => self.vsini,
            4 => self.rv,
            _ => panic!("Index out of bounds"),
        };
        param[index].max(bound.0).min(bound.1)
    }

    fn minmax(&self) -> (na::SVector<f64, 5>, na::SVector<f64, 5>) {
        (
            na::Vector5::new(
                *self.teff.values.first().unwrap(),
                *self.m.values.first().unwrap(),
                *self.logg.values.first().unwrap(),
                self.vsini.0,
                self.rv.0,
            ),
            na::Vector5::new(
                *self.teff.values.last().unwrap(),
                *self.m.values.last().unwrap(),
                *self.logg.values.last().unwrap(),
                self.vsini.1,
                self.rv.1,
            ),
        )
    }

    fn generate_random_within_bounds(
        &self,
        rng: &mut impl Rng,
        num_particles: usize,
    ) -> Vec<na::SVector<f64, 5>> {
        let (min, max) = self.minmax();
        (0..num_particles)
            .map(|_| na::SVector::rand_from_range(&min, &max, rng))
            .collect()
    }
}

pub trait ModelFetcher: Send + Sync {
    type E: Backend;
    fn ranges(&self) -> &SquareBounds;
    fn device(&self) -> &<Self::E as Backend>::Device;
    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<Tensor<Self::E, 1>>>;
}

#[derive(Clone)]
pub struct SquareGridInterpolator<F: ModelFetcher> {
    pub fetcher: F,
    pub synth_wl: WlGrid,
}

impl<F: ModelFetcher> SquareGridInterpolator<F> {
    pub fn new(model_fetcher: F, synth_wl: WlGrid) -> Self {
        Self {
            fetcher: model_fetcher,
            synth_wl,
        }
    }

    pub fn ranges(&self) -> &SquareBounds {
        self.fetcher.ranges()
    }

    pub fn device(&self) -> &<F::E as Backend>::Device {
        self.fetcher.device()
    }
}

pub fn nalgebra_to_tensor<E: Backend>(x: na::DVector<f64>, device: &E::Device) -> Tensor<E, 1> {
    Tensor::<E, 1>::from_data(Data::from(x.data.as_slice()).convert(), device)
}

pub fn tensor_to_nalgebra<E: Backend, T: Element + Scalar>(x: Tensor<E, 1>) -> na::DVector<T> {
    let vec: Vec<T> = x.into_data().convert().value;
    na::DVector::from_vec(vec)
}

impl<F: ModelFetcher> Interpolator for SquareGridInterpolator<F> {
    type B = SquareBounds;
    type E = F::E;
    fn synth_wl(&self) -> WlGrid {
        self.synth_wl
    }
    fn bounds(&self) -> &SquareBounds {
        &self.ranges()
    }
    fn device(&self) -> &<Self::E as Backend>::Device {
        &self.device()
    }
    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<Tensor<Self::E, 1>> {
        let InterpolInput {
            factors,
            local_4x4x4_indices,
        } = prepare_interpolate(self.ranges(), teff, m, logg, self.device())
            .context("prepare_interpolate error")?;

        // TODO: avoid unwrap
        let vec_of_tensors = local_4x4x4_indices
            .into_iter()
            .map(|[i, j, k]| {
                self.fetcher
                    .find_spectrum(i as usize, j as usize, k as usize)
                    .unwrap()
                    .into_owned()
            })
            .collect::<Vec<Tensor<Self::E, 1>>>();
        // (4, 4, 4, N)
        let local_4x4x4 = Tensor::stack::<2>(vec_of_tensors, 0).reshape([4, 4, 4, -1]);

        Ok(cubic_3d(factors, local_4x4x4))
    }

    fn produce_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<Tensor<Self::E, 1>> {
        let interpolated = tensor_to_nalgebra::<Self::E, f64>(
            self.interpolate(teff, m, logg)
                .context("Error during interpolation step.")?,
        );
        let broadened = rot_broad_rv(interpolated, self.synth_wl(), target_dispersion, vsini, rv)
            .context("Error during convolution/resampling step.")?;
        Ok(nalgebra_to_tensor(broadened, self.device()))
    }

    fn produce_model_on_grid(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<Tensor<Self::E, 1>> {
        let i = self
            .fetcher
            .ranges()
            .teff
            .try_get_index(teff)
            .context("Teff not in grid")?;
        let j = self
            .fetcher
            .ranges()
            .m
            .try_get_index(m)
            .context("m not in grid")?;
        let k = self
            .fetcher
            .ranges()
            .logg
            .try_get_index(logg)
            .context("logg not in grid")?;
        let spec = self.fetcher.find_spectrum(i, j, k)?;
        let broadened = rot_broad_rv(
            tensor_to_nalgebra(spec.into_owned()),
            self.synth_wl(),
            target_dispersion,
            vsini,
            rv,
        )?;
        Ok(nalgebra_to_tensor(broadened, self.device()))
    }
}

#[derive(Clone)]
pub struct CompoundBounds {
    bounds: [SquareBounds; 3],
}

impl CompoundBounds {
    /// Teff edges that determine in which grid will be interpolated
    pub fn edges(&self) -> (f64, f64) {
        (self.bounds[1].teff.values[1], self.bounds[0].teff.values[1])
    }

    /// Lowest Teff of grid 2, highest of grid 2, highest of grid 1, highest of grid 0
    pub fn teff_bounds(&self) -> (f64, f64, f64, f64) {
        let (min, lower_edge) = self.bounds[2].teff.get_first_and_last();
        let upper_edge = self.bounds[1].teff.get_first_and_last().1;
        let max = self.bounds[0].teff.get_first_and_last().1;
        (min, lower_edge, upper_edge, max)
    }

    pub fn m_bounds(&self) -> (f64, f64) {
        self.bounds[0].m.get_first_and_last()
    }

    pub fn logg_bounds(&self) -> (f64, f64, f64, f64) {
        let (min, max) = self.bounds[2].logg.get_first_and_last();
        let lower_edge = self.bounds[1].logg.get_first_and_last().0;
        let upper_edge = self.bounds[0].logg.get_first_and_last().0;
        (min, lower_edge, upper_edge, max)
    }

    pub fn vsini_bounds(&self) -> (f64, f64) {
        self.bounds[0].vsini
    }

    pub fn rv_bounds(&self) -> (f64, f64) {
        self.bounds[0].rv
    }
}

impl Bounds for CompoundBounds {
    fn is_within_bounds(&self, param: na::SVector<f64, 5>) -> bool {
        self.bounds[0].is_within_bounds(param)
            || self.bounds[1].is_within_bounds(param)
            || self.bounds[2].is_within_bounds(param)
    }

    /// Clamp a point outside the grid to the closest point inside the grid
    fn clamp(&self, param: na::SVector<f64, 5>) -> na::SVector<f64, 5> {
        if self.is_within_bounds(param) {
            return param;
        };

        let teff = param[0];
        let m = param[1];
        let logg = param[2];
        let vsini = param[3];
        let rv = param[4];

        let m_bounds = self.bounds[0].m.get_first_and_last();
        let vsini_bounds = self.bounds[0].vsini;
        let rv_bounds = self.bounds[0].rv;

        let new_m = m.max(m_bounds.0).min(m_bounds.1);
        let new_vsini = vsini.max(vsini_bounds.0).min(vsini_bounds.1);
        let new_rv = rv.max(rv_bounds.0).min(rv_bounds.1);
        let new_param = na::Vector5::new(teff, new_m, logg, new_vsini, new_rv);

        if self.is_within_bounds(new_param) {
            return new_param;
        };

        let (lower_bound_teff, lower_edge_teff, upper_edge_teff, upper_bound_teff) =
            self.teff_bounds();
        let (lower_bound_logg, lower_edge_logg, upper_edge_logg, upper_bound_logg) =
            self.logg_bounds();
        let teff_distance = |other_teff: f64| (teff - other_teff) * 40.0 / teff;
        let logg_distance = |other_logg: f64| (other_logg - logg) * 15.0;
        let point_distance = |other_teff: f64, other_logg: f64| {
            teff_distance(other_teff).hypot(logg_distance(other_logg))
        };
        let (new_teff, new_logg) = if logg < lower_bound_logg {
            if teff < lower_edge_teff {
                (teff.max(lower_bound_teff), lower_bound_logg)
            } else if teff < upper_edge_teff {
                let logg_distance = logg_distance(lower_edge_logg);
                let corner_distance = point_distance(lower_edge_teff, lower_bound_logg);
                if logg_distance < corner_distance {
                    (teff, lower_edge_logg)
                } else {
                    (lower_edge_teff, lower_bound_logg)
                }
            } else if teff < upper_bound_teff {
                let distance_corner1 = point_distance(lower_edge_teff, lower_bound_logg);
                let distance_corner2 = point_distance(upper_edge_teff, lower_edge_logg);
                let distance_logg = logg_distance(upper_edge_logg);
                if distance_corner1 < distance_corner2 && distance_corner1 < distance_logg {
                    (lower_edge_teff, lower_bound_logg)
                } else if distance_corner2 < distance_corner1 && distance_corner2 < distance_logg {
                    (upper_edge_teff, lower_edge_logg)
                } else {
                    (teff, upper_edge_logg)
                }
            } else {
                let distance_corner1 = point_distance(lower_edge_teff, lower_bound_logg);
                let distance_corner2 = point_distance(upper_edge_teff, lower_edge_logg);
                let distance_corner3 = point_distance(upper_bound_teff, upper_edge_logg);
                if distance_corner1 < distance_corner2 && distance_corner1 < distance_corner3 {
                    (lower_edge_teff, lower_bound_logg)
                } else if distance_corner2 < distance_corner1 && distance_corner2 < distance_corner3
                {
                    (upper_edge_teff, lower_edge_logg)
                } else {
                    (upper_bound_teff, upper_edge_logg)
                }
            }
        } else if logg < lower_edge_logg {
            if teff < lower_bound_teff {
                (lower_bound_teff, logg)
            } else if teff < lower_edge_teff {
                unreachable!()
            } else if teff < upper_edge_teff {
                let teff_distance = teff_distance(lower_edge_teff);
                let logg_distance = logg_distance(lower_edge_logg);
                if teff_distance < logg_distance {
                    (lower_edge_teff, logg)
                } else {
                    (teff, lower_edge_logg)
                }
            } else if teff < upper_bound_teff {
                let teff_distance = teff_distance(lower_edge_teff);
                let logg_distance = logg_distance(upper_edge_logg);
                let corner_distance = point_distance(upper_edge_teff, lower_edge_logg);

                if corner_distance < teff_distance && corner_distance < logg_distance {
                    (upper_edge_teff, lower_edge_logg)
                } else if teff_distance < corner_distance && teff_distance < logg_distance {
                    (lower_edge_teff, logg)
                } else {
                    (teff, upper_edge_logg)
                }
            } else {
                let teff_distance = teff_distance(lower_edge_teff);
                let corner_distance1 = point_distance(upper_edge_teff, lower_edge_logg);
                let corner_distance2 = point_distance(upper_bound_teff, upper_edge_logg);
                if corner_distance1 < corner_distance2 && corner_distance1 < teff_distance {
                    (upper_edge_teff, lower_edge_logg)
                } else if corner_distance2 < corner_distance1 && corner_distance2 < teff_distance {
                    (upper_bound_teff, upper_edge_logg)
                } else {
                    (lower_edge_teff, logg)
                }
            }
        } else if logg < upper_edge_logg {
            if teff < lower_bound_teff {
                (lower_bound_teff, logg)
            } else if teff < upper_edge_teff {
                unreachable!()
            } else if teff < upper_bound_teff {
                let teff_distance = teff_distance(upper_edge_teff);
                let logg_distance = logg_distance(upper_edge_logg);
                if teff_distance < logg_distance {
                    (upper_edge_teff, logg)
                } else {
                    (teff, upper_edge_logg)
                }
            } else {
                let teff_distance = teff_distance(upper_edge_teff);
                let corner_distance = point_distance(upper_edge_teff, upper_bound_logg);
                if teff_distance < corner_distance {
                    (upper_edge_teff, logg)
                } else {
                    (upper_bound_teff, upper_bound_logg)
                }
            }
        } else if logg < upper_bound_logg {
            (teff.max(lower_bound_teff).min(upper_bound_teff), logg)
        } else {
            (
                teff.max(lower_bound_teff).min(upper_bound_teff),
                upper_bound_logg,
            )
        };

        na::Vector5::new(new_teff, new_m, new_logg, new_vsini, new_rv)
    }

    fn clamp_1d(&self, param: na::SVector<f64, 5>, index: usize) -> f64 {
        let bound = match index {
            0 => {
                let lower_bound = self.bounds[2].teff.get_first_and_last().0;
                let (_, lower, upper, _) = self.logg_bounds();
                let grid_index = if param[2] <= lower {
                    2
                } else if param[2] <= upper {
                    1
                } else {
                    0
                };
                let upper_bound = self.bounds[grid_index].teff.get_first_and_last().1;
                (lower_bound, upper_bound)
            }
            1 => self.bounds[0].m.get_first_and_last(),
            2 => {
                let (_, lower, upper, _) = self.teff_bounds();
                let i = if param[0] <= lower {
                    2
                } else if param[0] <= upper {
                    1
                } else {
                    0
                };
                self.bounds[i].logg.get_first_and_last()
            }
            3 => self.bounds[0].vsini,
            4 => self.bounds[0].rv,
            _ => panic!("Index out of bounds"),
        };
        param[index].max(bound.0).min(bound.1)
    }

    fn minmax(&self) -> (na::SVector<f64, 5>, na::SVector<f64, 5>) {
        let (min1, max1) = self.bounds[0].minmax();
        let (min2, max2) = self.bounds[1].minmax();
        let (min3, max3) = self.bounds[2].minmax();
        (
            na::Vector5::new(
                min1[0].min(min2[0]).min(min3[0]),
                min1[1].min(min2[1]).min(min3[1]),
                min1[2].min(min2[2]).min(min3[2]),
                min1[3].min(min2[3]).min(min3[3]),
                min1[4].min(min2[4]).min(min3[4]),
            ),
            na::Vector5::new(
                max1[0].max(max2[0]).max(max3[0]),
                max1[1].max(max2[1]).max(max3[1]),
                max1[2].max(max2[2]).max(max3[2]),
                max1[3].max(max2[3]).max(max3[3]),
                max1[4].max(max2[4]).max(max3[4]),
            ),
        )
    }

    fn generate_random_within_bounds(
        &self,
        rng: &mut impl Rng,
        num_particles: usize,
    ) -> Vec<na::SVector<f64, 5>> {
        let mut particles = Vec::with_capacity(num_particles);
        let (min, max) = self.minmax();
        while particles.len() < num_particles {
            let param = na::SVector::rand_from_range(&min, &max, rng);
            if self.is_within_bounds(param) {
                particles.push(param);
            }
        }
        particles
    }
}

pub struct CompoundInterpolator<F: ModelFetcher> {
    interpolators: [SquareGridInterpolator<F>; 3],
    bounds: CompoundBounds,
}

impl<F: ModelFetcher> CompoundInterpolator<F> {
    pub fn new(
        first: SquareGridInterpolator<F>,
        second: SquareGridInterpolator<F>,
        third: SquareGridInterpolator<F>,
    ) -> Self {
        let bounds = CompoundBounds {
            bounds: [
                first.ranges().clone(),
                second.ranges().clone(),
                third.ranges().clone(),
            ],
        };
        CompoundInterpolator {
            interpolators: [first, second, third],
            bounds,
        }
    }

    fn choose_grid(&self, teff: f64, logg: f64) -> usize {
        let (lower_edge, upper_edge) = self.bounds.edges();
        let (_, lower_logg, upper_logg, _) = self.bounds.logg_bounds();
        if teff <= lower_edge || logg < lower_logg {
            2
        } else if teff <= upper_edge || logg < upper_logg {
            1
        } else {
            0
        }
    }
}

impl<'a, F: ModelFetcher> Interpolator for CompoundInterpolator<F> {
    type B = CompoundBounds;
    type E = F::E;

    fn synth_wl(&self) -> WlGrid {
        self.interpolators[0].synth_wl
    }
    fn bounds(&self) -> &CompoundBounds {
        &self.bounds
    }
    fn device(&self) -> &<Self::E as Backend>::Device {
        &self.interpolators[0].device()
    }

    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<Tensor<Self::E, 1>> {
        let i = self.choose_grid(teff, logg);
        self.interpolators[i].interpolate(teff, m, logg)
    }

    fn produce_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<Tensor<Self::E, 1>> {
        let i = self.choose_grid(teff, logg);
        self.interpolators[i].produce_model(target_dispersion, teff, m, logg, vsini, rv)
    }
    fn produce_model_on_grid(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<Tensor<Self::E, 1>> {
        let i = self.choose_grid(teff, logg);
        self.interpolators[i].produce_model_on_grid(target_dispersion, teff, m, logg, vsini, rv)
    }
}
