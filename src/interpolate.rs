use crate::convolve_rv::{rot_broad_rv, WavelengthDispersion};
use crate::cubic::{calculate_interpolation_coefficients, LocalGrid};
use crate::particleswarm::PSOBounds;
use anyhow::{anyhow, bail, Context, Result};
use argmin_math::ArgminRandom;
use itertools::Itertools;
use nalgebra as na;
use npy::NpyData;
use rand::Rng;
use std::io::Read;
use std::path::{Path, PathBuf};

pub type FluxFloat = f32; // Float type used for spectra

pub enum CowVector<'a> {
    Borrowed(na::DVectorView<'a, FluxFloat>),
    Owned(na::DVector<FluxFloat>),
}

impl<'a> CowVector<'a> {
    pub fn into_owned(self) -> na::DVector<FluxFloat> {
        match self {
            CowVector::Borrowed(view) => view.into_owned(),
            CowVector::Owned(vec) => vec,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CowVector::Borrowed(view) => view.len(),
            CowVector::Owned(vec) => vec.len(),
        }
    }

    pub fn rows(&self, start: usize, len: usize) -> na::DVectorView<FluxFloat> {
        match self {
            CowVector::Borrowed(view) => view.rows(start, len),
            CowVector::Owned(vec) => vec.rows(start, len),
        }
    }

    pub fn fixed_rows<const N: usize>(
        &'a self,
        start: usize,
    ) -> na::Matrix<
        FluxFloat,
        na::Const<N>,
        na::Const<1>,
        na::ViewStorage<'a, FluxFloat, na::Const<N>, na::Const<1>, na::Const<1>, na::Dyn>,
    > {
        match self {
            CowVector::Borrowed(view) => view.fixed_rows::<N>(start),
            CowVector::Owned(vec) => vec.fixed_rows::<N>(start),
        }
    }
}

pub fn read_npy_file<F: npy::Serializable>(file_path: PathBuf) -> Result<Vec<F>> {
    let mut file = std::fs::File::open(file_path.clone())
        .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let data: NpyData<F> = NpyData::from_bytes(&buf)
        .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
    Ok(data.to_vec())
}

pub fn read_spectrum(dir: &Path, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>> {
    // e.g. 00027_lm0050_07000_0350_0020_0000_Vsini_0000.npy
    let _teff = teff.round() as i32;
    let _m = (m * 100.0).round() as i32;
    let _logg = (logg * 100.0).round() as i32;
    let sign = if _m < 0 { "m" } else { "p" };
    let filename = format!("l{}{:04}_{:05}_{:04}", sign, _m.abs(), _teff, _logg);
    let file_path = dir.join(format!("{}.npy", filename));
    let result = read_npy_file::<FluxFloat>(file_path.clone())?;
    Ok(na::DVector::from_iterator(
        result.len(),
        result.into_iter().map(|x| x as FluxFloat),
    ))
}

#[derive(Clone)]
pub struct Range {
    pub values: Vec<f64>,
}

impl Range {
    pub fn get_right_index(&self, x: f64) -> Result<usize, usize> {
        self.values.binary_search_by(|v| v.partial_cmp(&x).unwrap())
    }

    pub fn find_left_neighbor_index(&self, x: f64, limits: (usize, usize)) -> Option<usize> {
        match self.get_right_index(x) {
            Ok(i) => {
                if i == limits.0 {
                    Some(limits.0)
                } else {
                    Some(i - 1)
                }
            }
            Err(i) => {
                if i == limits.0 {
                    None
                } else {
                    Some(i - 1)
                }
            }
        }
    }

    pub fn find_neighbors(&self, x: f64, limits: (usize, usize)) -> Result<[Option<usize>; 4]> {
        let (left, right) = limits;
        match self.get_right_index(x) {
            Ok(i) => {
                if i == left {
                    Ok([None, Some(i), Some(i + 1), Some(i + 2)])
                } else if i == right {
                    Ok([Some(i - 2), Some(i - 1), Some(i), None])
                } else if i == right - 1 {
                    Ok([Some(i - 2), Some(i - 1), Some(i), Some(i + 1)])
                } else {
                    Ok([Some(i - 1), Some(i), Some(i + 1), Some(i + 2)])
                }
            }
            Err(i) => {
                if i == left {
                    Err(anyhow!("Index out of left bound {}", i))
                } else if i == right + 1 {
                    Err(anyhow!("Index out of right bound {}", i))
                } else if i == left + 1 {
                    Ok([None, Some(i - 1), Some(i), Some(i + 1)])
                } else if i == right {
                    Ok([Some(i - 2), Some(i - 1), Some(i), None])
                } else {
                    Ok([Some(i - 2), Some(i - 1), Some(i), Some(i + 1)])
                }
            }
        }
    }

    pub fn try_get_index(&self, x: f64, limits: (usize, usize)) -> Option<usize> {
        let precision = 0.001;
        let i = match self.find_left_neighbor_index(x, limits) {
            Some(i) => i,
            None => return None,
        };
        let j = i + 1;
        if (self.values[i] - x).abs() < precision {
            Some(i)
        } else if (self.values[j] - x).abs() < precision {
            Some(j)
        } else {
            None
        }
    }

    pub fn get(&self, i: usize) -> Result<f64> {
        self.values
            .get(i)
            .ok_or_else(|| anyhow!("Index out of bounds ({})", i))
            .map(|x| *x)
    }

    pub fn n(&self) -> usize {
        self.values.len()
    }

    pub fn get_first_and_last(&self) -> (f64, f64) {
        (self.values[0], *self.values.last().unwrap())
    }
}

#[derive(Clone)]
pub struct Grid {
    pub teff: Range,
    pub m: Range,
    pub logg: Range,
    pub logg_limits: Vec<(usize, usize)>, // For every Teff value, the min and max logg index
    // For every Teff value, the number of (teff, logg) pairs with lower Teff
    pub cumulative_grid_size: Vec<usize>,
    pub vsini: (f64, f64),
    pub rv: (f64, f64),
}

impl Grid {
    pub fn new(
        model_labels: Vec<(f64, f64, f64)>,
        vsini: (f64, f64),
        rv: (f64, f64),
    ) -> Result<Self> {
        let mut teffs = Vec::new();
        let mut ms = Vec::new();
        let mut loggs = Vec::new();
        for (teff, m, logg) in model_labels.iter() {
            if !teffs.contains(teff) {
                let pos = teffs
                    .binary_search_by(|v| v.partial_cmp(&teff).unwrap())
                    .unwrap_or_else(|x| x);
                teffs.insert(pos, *teff);
            }
            if !ms.contains(m) {
                let pos = ms
                    .binary_search_by(|v| v.partial_cmp(&m).unwrap())
                    .unwrap_or_else(|x| x);
                ms.insert(pos, *m);
            }
            if !loggs.contains(logg) {
                let pos = loggs
                    .binary_search_by(|v| v.partial_cmp(&logg).unwrap())
                    .unwrap_or_else(|x| x);
                loggs.insert(pos, *logg);
            }
        }

        let mut logg_limits = vec![(loggs.len() - 1, 0); teffs.len()];
        for (teff, _, logg) in model_labels {
            let i = teffs
                .binary_search_by(|v| v.partial_cmp(&teff).unwrap())
                .unwrap();
            let j = loggs
                .binary_search_by(|v| v.partial_cmp(&logg).unwrap())
                .unwrap();
            if j < logg_limits[i].0 {
                logg_limits[i].0 = j;
            }
            if j > logg_limits[i].1 {
                logg_limits[i].1 = j;
            }
        }
        // Sanity check
        for (left, right) in logg_limits.iter() {
            if !(left <= right) {
                bail!("Invalid logg limits");
            }
        }

        let cumulative_grid_size = logg_limits
            .iter()
            .scan(0, |acc, (left, right)| {
                let n = right - left + 1;
                let old = *acc;
                *acc += n;
                Some(old)
            })
            .collect();

        Ok(Grid {
            teff: Range { values: teffs },
            m: Range { values: ms },
            logg: Range { values: loggs },
            logg_limits,
            cumulative_grid_size,
            vsini,
            rv,
        })
    }

    pub fn minmax(&self) -> (na::SVector<f64, 5>, na::SVector<f64, 5>) {
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

    pub fn list_gridpoints(&self) -> Vec<[f64; 3]> {
        self.teff
            .values
            .iter()
            .enumerate()
            .flat_map(|(i, teff)| {
                let (left, right) = self.logg_limits[i];
                self.logg.values[left..=right]
                    .iter()
                    .map(move |logg| (teff, logg))
            })
            .cartesian_product(self.m.values.iter())
            .map(|((teff, logg), m)| [*teff, *m, *logg])
            .collect()
    }

    pub fn is_teff_logg_between_bounds(&self, teff: f64, logg: f64) -> bool {
        match self.teff.get_right_index(teff) {
            Ok(i) => {
                let bounds = self.logg_limits[i];
                logg >= self.logg.get(bounds.0).unwrap() && logg <= self.logg.get(bounds.1).unwrap()
            }
            Err(i) => {
                if i == 0 || i == self.teff.n() {
                    return false;
                }
                let bounds_left = self.logg_limits[i - 1];
                let bounds_right = self.logg_limits[i];
                logg >= self.logg.get(bounds_left.0).unwrap()
                    && logg <= self.logg.get(bounds_left.1).unwrap()
                    && logg >= self.logg.get(bounds_right.0).unwrap()
                    && logg <= self.logg.get(bounds_right.1).unwrap()
            }
        }
    }

    pub fn is_m_between_bounds(&self, m: f64) -> bool {
        m >= self.m.values[0] && m <= *self.m.values.last().unwrap()
    }

    pub fn is_vsini_between_bounds(&self, vsini: f64) -> bool {
        vsini >= self.vsini.0 && vsini <= self.vsini.1
    }

    pub fn is_rv_between_bounds(&self, rv: f64) -> bool {
        rv >= self.rv.0 && rv <= self.rv.1
    }

    pub fn is_within_bounds(&self, param: na::SVector<f64, 5>) -> bool {
        // When the parameters are not in the rectangular grid:
        self.is_rv_between_bounds(param[4])
            && self.is_vsini_between_bounds(param[3])
            && self.is_m_between_bounds(param[1])
            && self.is_teff_logg_between_bounds(param[0], param[2])
    }

    pub fn get_logg_index_limits_at(&self, teff: f64) -> Result<(usize, usize)> {
        match self.teff.get_right_index(teff) {
            Ok(i) => {
                if i == self.teff.n() - 1 {
                    Ok(self.logg_limits[i])
                } else {
                    let left = self.logg_limits[i];
                    let right = self.logg_limits[i + 1];
                    Ok((left.0.max(right.0), left.1.min(right.1)))
                }
            }
            Err(i) => {
                if i == 0 || i == self.teff.n() {
                    bail!("Teff out of bounds");
                }
                let left = self.logg_limits[i - 1];
                let right = self.logg_limits[i];
                Ok((left.0.max(right.0), left.1.min(right.1)))
            }
        }
    }

    pub fn get_teff_index_limits_at(&self, logg: f64) -> Result<(usize, usize)> {
        let min_index = self
            .logg_limits
            .iter()
            .position(|(left, right)| {
                logg >= self.logg.values[*left] && logg <= self.logg.values[*right]
            })
            .ok_or(anyhow!("No teff bounds at logg={}", logg))?;
        let max_index = self.logg_limits.len()
            - 1
            - self
                .logg_limits
                .iter()
                .rev()
                .position(|(left, right)| {
                    logg >= self.logg.values[*left] && logg <= self.logg.values[*right]
                })
                .ok_or(anyhow!("No teff bounds at logg={}", logg))?;
        Ok((min_index, max_index))
    }

    pub fn get_local_grid(&self, teff: f64, m: f64, logg: f64) -> Result<LocalGrid> {
        if !self.is_teff_logg_between_bounds(teff, logg) {
            bail!("Teff, logg out of bounds ({}, {})", teff, logg);
        }
        if !self.is_m_between_bounds(m) {
            bail!("M out of bounds ({})", m);
        }
        let teff_limits = self.get_teff_index_limits_at(logg)?;
        let m_limits = (0, self.m.n() - 1);
        let logg_limits = self.get_logg_index_limits_at(teff)?;

        let teff_neighbors = self
            .teff
            .find_neighbors(teff, teff_limits)
            .with_context(|| {
                format!(
                    "failed getting neighbors for teff: {}, {:?}",
                    teff, teff_limits
                )
            })?;
        let m_neighbors = self.m.find_neighbors(m, m_limits).with_context(|| {
            format!(
                "failed getting neighbors for m: {}, {:?}",
                logg, logg_limits
            )
        })?;
        let logg_neighbors = self
            .logg
            .find_neighbors(logg, logg_limits)
            .with_context(|| format!("failed getting neighbors for logg: {}, {:?}", m, m_limits))?;

        let teff_logg_indices =
            na::SMatrix::from_columns(&teff_neighbors.map(|teff_neighbor| match teff_neighbor {
                Some(i) => {
                    let (left, right) = self.logg_limits[i];
                    logg_neighbors
                        .map(|k| match k {
                            Some(k) => {
                                if k >= left && k <= right {
                                    Some((i, k))
                                } else {
                                    None
                                }
                            }
                            None => None,
                        })
                        .into()
                }
                None => [None; 4].into(),
            }));

        Ok(LocalGrid {
            teff: (
                teff,
                teff_neighbors.map(|x| {
                    x.map(|x| {
                        self.teff
                            .get(x)
                            .with_context(|| {
                                format!("teff={}, neighbors: {:?}", teff, teff_neighbors)
                            })
                            .unwrap()
                    })
                }),
            ),
            logg: (
                logg,
                logg_neighbors.map(|x| {
                    x.map(|x| {
                        self.logg
                            .get(x)
                            .with_context(|| {
                                format!("logg={}, neighbors: {:?}", logg, logg_neighbors)
                            })
                            .unwrap()
                    })
                }),
            ),
            teff_logg_indices,
            m: (
                m,
                m_neighbors.map(|x| {
                    x.map(|x| {
                        self.m
                            .get(x)
                            .with_context(|| format!("m={}, neighbors: {:?}", m, m_neighbors))
                            .unwrap()
                    })
                }),
            ),
            m_indices: m_neighbors.into(),
        })
    }
}

impl PSOBounds<5> for Grid {
    fn limits(&self) -> (na::SVector<f64, 5>, na::SVector<f64, 5>) {
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

    fn clamp_1d(&self, param: na::SVector<f64, 5>, index: usize) -> Result<f64> {
        let bound = match index {
            0 => {
                // Avoid doing more expensive checks if the parameter is already within bounds
                if self.is_within_bounds(param) {
                    return Ok(param[0]);
                } else {
                    let (left_index, right_index) = self
                        .get_teff_index_limits_at(param[2])
                        .context(anyhow!("Cannot clamp teff at logg={}", param[2]))?;
                    (self.teff.get(left_index)?, self.teff.get(right_index)?)
                }
            }
            1 => self.m.get_first_and_last(),
            2 => {
                let (left_index, right_index) = self
                    .get_logg_index_limits_at(param[0])
                    .context(anyhow!("Cannot clamp logg at Teff={}", param[0]))?;
                (self.logg.get(left_index)?, self.logg.get(right_index)?)
            }
            3 => self.vsini,
            4 => self.rv,
            _ => panic!("Index out of bounds"),
        };
        Ok(param[index].max(bound.0).min(bound.1))
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
                Box::new((0..*total).map(move |i| (first + step * i as f64)))
            }
            WlGrid::Logspace(first, step, total) => {
                Box::new((0..*total).map(move |i| 10_f64.powf(first + step * i as f64)))
            }
        }
    }
}

pub trait Interpolator: Send + Sync {
    type BS: PSOBounds<5>; // Bounds for single star
                           // type BB: PSOBounds<11>; // Bounds for binary star

    fn synth_wl(&self) -> WlGrid;
    fn bounds_single(&self) -> &Self::BS;
    // fn bounds_binary(&self) -> &Self::BB;
    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>>;
    fn produce_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>>;
    fn produce_model_on_grid(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>>;

    fn produce_binary_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        star1_parameters: (f64, f64, f64, f64, f64),
        star2_parameters: (f64, f64, f64, f64, f64),
        light_ratio: f64,
    ) -> Result<na::DVector<FluxFloat>> {
        let model1 = self.produce_model(
            target_dispersion,
            star1_parameters.0,
            star1_parameters.1,
            star1_parameters.2,
            star1_parameters.3,
            star1_parameters.4,
        )?;
        let model2 = self.produce_model(
            target_dispersion,
            star2_parameters.0,
            star2_parameters.1,
            star2_parameters.2,
            star2_parameters.3,
            star2_parameters.4,
        )?;
        Ok(model1 * (light_ratio as f32) + model2 * (1.0 - light_ratio as f32))
    }
}

pub trait ModelFetcher: Send + Sync {
    fn grid(&self) -> &Grid;
    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<CowVector>;
}

#[derive(Clone)]
pub struct GridInterpolator<F: ModelFetcher> {
    pub fetcher: F,
    pub synth_wl: WlGrid,
}

impl<F: ModelFetcher> GridInterpolator<F> {
    pub fn new(model_fetcher: F, synth_wl: WlGrid) -> Self {
        Self {
            fetcher: model_fetcher,
            synth_wl,
        }
    }

    pub fn grid(&self) -> &Grid {
        self.fetcher.grid()
    }
}

const BATCH_SIZE: usize = 128;

impl<F: ModelFetcher> Interpolator for GridInterpolator<F> {
    type BS = Grid;
    fn synth_wl(&self) -> WlGrid {
        self.synth_wl
    }
    fn bounds_single(&self) -> &Grid {
        self.grid()
    }
    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>> {
        let local_grid = self.grid().get_local_grid(teff, m, logg)?;

        let factors = calculate_interpolation_coefficients(&local_grid)?;
        let neighbors = local_grid
            .teff_logg_indices
            .iter()
            .flat_map(|teff_logg_index| {
                local_grid
                    .m_indices
                    .iter()
                    .map(move |m_index| match (teff_logg_index, m_index) {
                        (Some((i, j)), Some(k)) => self.fetcher.find_spectrum(*i, *k, *j),
                        _ => Ok(CowVector::Owned(na::DVector::zeros(self.synth_wl.n()))),
                    })
            })
            .collect::<Result<Vec<CowVector>>>()?;

        let model_length = neighbors[0].len();
        let factors_s =
            na::SVector::<FluxFloat, 64>::from_iterator(factors.iter().map(|x| *x as FluxFloat));
        let mut interpolated: na::DVector<FluxFloat> = na::DVector::zeros(model_length);
        let mut mat = na::SMatrix::<FluxFloat, BATCH_SIZE, 64>::zeros();
        for i in 0..(model_length / BATCH_SIZE) {
            let start = i * BATCH_SIZE;
            for j in 0..64 {
                mat.set_column(j, &neighbors[j].fixed_rows::<BATCH_SIZE>(start));
            }

            mat.mul_to(&factors_s, &mut interpolated.rows_mut(start, BATCH_SIZE));
        }
        // Add remaining part
        let start = (model_length / BATCH_SIZE) * BATCH_SIZE;
        let remaining = model_length - start;
        let mut mat = na::Matrix::<FluxFloat, na::Dyn, na::Const<64>, _>::zeros(remaining);
        for j in 0..64 {
            mat.set_column(j, &neighbors[j].rows(start, remaining));
        }
        mat.mul_to(&factors_s, &mut interpolated.rows_mut(start, remaining));
        Ok(interpolated)
    }

    fn produce_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>> {
        let interpolated = self
            .interpolate(teff, m, logg)
            .with_context(|| format!("Error while interpolating at ({}, {}, {})", teff, m, logg))?;
        let broadened = rot_broad_rv(interpolated, self.synth_wl(), target_dispersion, vsini, rv)
            .with_context(|| {
            format!(
                "Error during convolution/resampling step at ({}, {}, {})",
                teff, m, logg
            )
        })?;
        Ok(broadened)
    }

    fn produce_model_on_grid(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>> {
        let teff_limits = self.grid().get_teff_index_limits_at(logg)?;
        let m_limits = (0, self.grid().m.n() - 1);
        let logg_limits = self.grid().get_logg_index_limits_at(teff)?;

        let i = self
            .fetcher
            .grid()
            .teff
            .try_get_index(teff, teff_limits)
            .context("Teff not in grid")?;
        let j = self
            .fetcher
            .grid()
            .m
            .try_get_index(m, m_limits)
            .context("m not in grid")?;
        let k = self
            .fetcher
            .grid()
            .logg
            .try_get_index(logg, logg_limits)
            .context("logg not in grid")?;
        let spec = self.fetcher.find_spectrum(i, j, k)?;
        let broadened = rot_broad_rv(
            spec.into_owned(),
            self.synth_wl(),
            target_dispersion,
            vsini,
            rv,
        )?;
        Ok(broadened)
    }
}
