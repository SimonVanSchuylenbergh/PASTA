use crate::convolve_rv::{rot_broad_rv, WavelengthDispersion};
use crate::cubic::calculate_interpolation_coefficients;
use crate::particleswarm::PSOBounds;
use anyhow::{Context, Result};
use argmin_math::ArgminRandom;
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

pub trait GridBounds: Clone {
    fn is_within_bounds(&self, param: na::SVector<f64, 5>) -> bool;
    fn clamp_1d(&self, param: na::SVector<f64, 5>, index: usize) -> f64;
    fn minmax(&self) -> (na::SVector<f64, 5>, na::SVector<f64, 5>);
}

impl<GB: GridBounds> PSOBounds<5> for GB {
    fn minmax(&self) -> (nalgebra::SVector<f64, 5>, nalgebra::SVector<f64, 5>) {
        self.minmax()
    }

    fn clamp_1d(&self, param: nalgebra::SVector<f64, 5>, index: usize) -> f64 {
        self.clamp_1d(param, index)
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
    type B: GridBounds;

    fn synth_wl(&self) -> WlGrid;
    fn bounds(&self) -> &Self::B;
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

impl GridBounds for SquareBounds {
    fn is_within_bounds(&self, param: na::SVector<f64, 5>) -> bool {
        self.teff.between_bounds(param[0])
            && self.m.between_bounds(param[1])
            && self.logg.between_bounds(param[2])
            && param[3] >= self.vsini.0
            && param[3] <= self.vsini.1
            && param[4] >= self.rv.0
            && param[4] <= self.rv.1
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
}

pub trait ModelFetcher: Send + Sync {
    fn ranges(&self) -> &SquareBounds;
    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<CowVector>;
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
}

const BATCH_SIZE: usize = 128;

impl<F: ModelFetcher> Interpolator for SquareGridInterpolator<F> {
    type B = SquareBounds;
    fn synth_wl(&self) -> WlGrid {
        self.synth_wl
    }
    fn bounds(&self) -> &SquareBounds {
        self.ranges()
    }
    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>> {
        let factors = calculate_interpolation_coefficients(self.ranges(), teff, m, logg)?;

        let teff_range = self.ranges().teff.clone();
        let m_range = self.ranges().m.clone();
        let logg_range = self.ranges().logg.clone();

        let i = teff_range.get_left_index(teff);
        let j = m_range.get_left_index(m);
        let k = logg_range.get_left_index(logg);

        let neighbors = (-1..3)
            .flat_map(|di| (-1..3).flat_map(move |dj| (-1..3).map(move |dk| (di, dj, dk))))
            .map(|(di, dj, dk)| {
                let i = ((i as i64 + di).max(0).min(teff_range.n() as i64 - 1)) as usize;
                let j = ((j as i64 + dj).max(0).min(m_range.n() as i64 - 1)) as usize;
                let k = ((k as i64 + dk).max(0).min(logg_range.n() as i64 - 1)) as usize;
                self.fetcher.find_spectrum(i, j, k)
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
            .context("Error during interpolation step.")?;
        let broadened = rot_broad_rv(interpolated, self.synth_wl(), target_dispersion, vsini, rv)
            .context("Error during convolution/resampling step.")?;
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
            spec.into_owned(),
            self.synth_wl(),
            target_dispersion,
            vsini,
            rv,
        )?;
        Ok(broadened)
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

impl GridBounds for CompoundBounds {
    fn is_within_bounds(&self, param: na::SVector<f64, 5>) -> bool {
        self.bounds[0].is_within_bounds(param)
            || self.bounds[1].is_within_bounds(param)
            || self.bounds[2].is_within_bounds(param)
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
        let (min1, max1) = GridBounds::minmax(&self.bounds[0]);
        let (min2, max2) = GridBounds::minmax(&self.bounds[1]);
        let (min3, max3) = GridBounds::minmax(&self.bounds[2]);
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
}

pub struct CompoundInterpolator<F: ModelFetcher> {
    pub interpolators: [SquareGridInterpolator<F>; 3],
    pub bounds: CompoundBounds,
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

    fn synth_wl(&self) -> WlGrid {
        self.interpolators[0].synth_wl
    }
    fn bounds(&self) -> &CompoundBounds {
        &self.bounds
    }

    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>> {
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
    ) -> Result<na::DVector<FluxFloat>> {
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
    ) -> Result<na::DVector<FluxFloat>> {
        let i = self.choose_grid(teff, logg);
        self.interpolators[i].produce_model_on_grid(target_dispersion, teff, m, logg, vsini, rv)
    }
}
