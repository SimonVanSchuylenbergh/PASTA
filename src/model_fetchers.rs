use crate::interpolate::{CowVector, Grid, ModelFetcher};
use anyhow::{bail, Context, Result};
use indicatif::ProgressBar;
use nalgebra as na;
use npy::NpyData;
use rayon::prelude::*;
use std::cmp::Eq;
use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

pub fn read_npy_file(file_path: PathBuf) -> Result<Vec<u16>> {
    let mut file = std::fs::File::open(file_path.clone())
        .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let data: NpyData<u16> = NpyData::from_bytes(&buf)
        .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
    Ok(data.to_vec())
}

pub fn read_spectrum(
    dir: &Path,
    teff: f64,
    m: f64,
    logg: f64,
    includes_factor: bool,
) -> Result<(na::DVector<u16>, Option<f32>)> {
    // e.g. 00027_lm0050_07000_0350_0020_0000_Vsini_0000.npy
    let _teff = teff.round() as i32;
    let _m = (m * 100.0).round() as i32;
    let _logg = (logg * 100.0).round() as i32;
    let sign = if _m < 0 { "m" } else { "p" };
    let filename = format!("l{}{:04}_{:05}_{:04}", sign, _m.abs(), _teff, _logg);
    let file_path = dir.join(format!("{}.npy", filename));
    let result = read_npy_file(file_path.clone())?;
    if includes_factor {
        let bytes1 = result[0].to_le_bytes();
        let bytes2 = result[1].to_le_bytes();
        let factor = f32::from_le_bytes([bytes1[0], bytes1[1], bytes2[0], bytes2[1]]);
        if factor < 0.0 {
            bail!("Negative factor: {}", factor);
        }
        Ok((
            na::DVector::from_iterator(result.len() - 2, result.into_iter().skip(2)),
            Some(factor),
        ))
    } else {
        Ok((na::DVector::from_iterator(result.len(), result), None))
    }
}

fn labels_from_filename(filename: &str) -> Result<[f64; 3]> {
    // e.g. lp0020_08000_0430.npy
    let parts: Vec<&str> = filename.split(".").next().unwrap().split('_').collect();
    if parts.len() != 3 || parts[0].len() != 6 || parts[1].len() != 5 || parts[2].len() != 4 {
        bail!("Invalid filename: {}", filename);
    }
    let sign = if parts[0].starts_with("lm") {
        -1.0
    } else {
        1.0
    };
    let m = sign * parts[0][2..].parse::<f64>()? / 100.0;
    let teff = parts[1].parse::<f64>()?;
    let logg = parts[2].parse::<f64>()? / 100.0;
    Ok([teff, m, logg])
}

fn get_model_labels_in_dir(dir: &PathBuf) -> Result<Vec<[f64; 3]>> {
    std::fs::read_dir(dir)
        .with_context(|| format!("Not found {:?}", dir))?
        .map(|entry| {
            let path = entry?.path();
            let filename = path.file_name().unwrap().to_str().unwrap();
            labels_from_filename(filename)
        })
        .collect()
}

#[derive(Clone)]
pub struct OnDiskFetcher {
    pub dir: PathBuf,
    pub grid: Grid,
    pub includes_factor: bool,
}

impl OnDiskFetcher {
    pub fn new(dir: &str, includes_factor: bool) -> Result<Self> {
        let model_labels = get_model_labels_in_dir(&PathBuf::from(dir))?;
        let grid = Grid::new(model_labels)?;
        Ok(Self {
            dir: PathBuf::from(dir),
            grid,
            includes_factor,
        })
    }
}

impl ModelFetcher for OnDiskFetcher {
    fn grid(&self) -> &Grid {
        &self.grid
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<(CowVector, f32)> {
        let teff = self.grid.teff.get(i)?;
        let m = self.grid.m.get(j)?;
        let logg = self.grid.logg.get(k)?;
        // println!("Finding i={}, j={}, k={}", i, j, k);
        // println!("teff={}, m={}, logg={}", teff, m, logg);
        let (spec, factor) = read_spectrum(&self.dir, teff, m, logg, self.includes_factor)?;
        Ok((CowVector::Owned(spec), factor.unwrap_or(1.0)))
    }
}

#[derive(Clone)]
pub struct InMemFetcher {
    pub grid: Grid,
    pub loaded_spectra: na::DMatrix<u16>,
    pub factors: Option<na::DVector<f32>>,
}

fn load_spectra(
    dir: PathBuf,
    includes_factor: bool,
    grid: &Grid,
) -> Result<(na::DMatrix<u16>, Option<na::DVector<f32>>)> {
    let combinations = grid.list_gridpoints();

    let bar = ProgressBar::new(combinations.len() as u64);
    let vec_of_spectra_and_factors = combinations
        .into_par_iter()
        .map(|[teff, m, logg]| {
            bar.inc(1);
            read_spectrum(&dir, teff, m, logg, includes_factor)
        })
        .collect::<Result<Vec<_>>>()?;
    let (spectra, factors): (Vec<_>, Vec<_>) = vec_of_spectra_and_factors.into_iter().unzip();
    if includes_factor {
        Ok((
            na::DMatrix::from_columns(&spectra),
            factors
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .map(|v| na::DVector::from_iterator(v.len(), v)),
        ))
    } else {
        Ok((na::DMatrix::from_columns(&spectra), None))
    }
}

impl InMemFetcher {
    pub fn new(dir: &str, includes_factor: bool) -> Result<Self> {
        let model_labels = get_model_labels_in_dir(&PathBuf::from(dir))?;
        let grid = Grid::new(model_labels)?;
        let (loaded_spectra, factors) = load_spectra(PathBuf::from(dir), includes_factor, &grid)?;
        let n = loaded_spectra.shape().0;
        Ok(Self {
            grid,
            loaded_spectra,
            factors,
        })
    }
}

impl ModelFetcher for InMemFetcher {
    fn grid(&self) -> &Grid {
        &self.grid
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<(CowVector, f32)> {
        let idx = (self.grid.cumulative_grid_size[i] + k - self.grid.logg_limits[i].0)
            * self.grid.m.n()
            + j;
        Ok((
            CowVector::Borrowed(self.loaded_spectra.column(idx)),
            self.factors.as_ref().map(|fac| fac[idx]).unwrap_or(1.0),
        ))
    }
}

pub struct Cache<K: Eq, V: Clone> {
    queue: VecDeque<(K, V)>,
    cap: usize,
}

impl<K: Eq, V: Clone> Cache<K, V> {
    fn new(cap: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            cap,
        }
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn get(&self, k: &K) -> Option<V> {
        for (key, value) in &self.queue {
            if key == k {
                return Some(value.clone());
            }
        }
        None
    }

    fn put(&mut self, k: K, v: V) {
        if self.queue.len() == self.cap {
            self.queue.pop_front();
        }
        self.queue.push_back((k, v));
    }
}

#[derive(Clone)]
pub struct CachedFetcher {
    pub dir: PathBuf,
    pub grid: Grid,
    pub includes_factor: bool,
    shards: Vec<Arc<RwLock<Cache<(usize, usize, usize), (na::DVector<u16>, f32)>>>>,
}

impl CachedFetcher {
    pub fn new(dir: &str, includes_factor: bool, lrucap: usize, n_shards: usize) -> Result<Self> {
        let model_labels = get_model_labels_in_dir(&PathBuf::from(dir))?;
        let grid = Grid::new(model_labels)?;
        let shards = (0..n_shards)
            .map(|_| Arc::new(RwLock::new(Cache::new(lrucap / n_shards))))
            .collect();
        Ok(Self {
            dir: PathBuf::from(dir),
            includes_factor,
            grid,
            shards,
        })
    }

    pub fn cache_size(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.read().unwrap().len())
            .sum()
    }
}

impl ModelFetcher for CachedFetcher {
    fn grid(&self) -> &Grid {
        &self.grid
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<(CowVector, f32)> {
        let shard_idx = (i + j + k) % self.shards.len();
        let shard = self.shards[shard_idx].read().unwrap();
        if let Some((spec, factor)) = shard.get(&(i, j, k)) {
            Ok((CowVector::Owned(spec), factor))
        } else {
            std::mem::drop(shard);
            let teff = self.grid.teff.get(i)?;
            let m = self.grid.m.get(j)?;
            let logg = self.grid.logg.get(k)?;
            let (spec, factor) = read_spectrum(&self.dir, teff, m, logg, self.includes_factor)?;
            let mut shard = self.shards[shard_idx].write().unwrap();
            shard.put((i, j, k), (spec.clone(), factor.unwrap_or(1.0)));
            Ok((CowVector::Owned(spec), factor.unwrap_or(1.0)))
        }
    }
}
