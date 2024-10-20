use crate::interpolate::{CowVector, FluxFloat, Grid, ModelFetcher};
use anyhow::{bail, Context, Result};
use indicatif::ProgressBar;
use lru::LruCache;
use nalgebra as na;
use npy::NpyData;
use rayon::prelude::*;
use std::io::Read;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub fn read_npy_file(file_path: PathBuf) -> Result<Vec<u16>> {
    let mut file = std::fs::File::open(file_path.clone())
        .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let data: NpyData<u16> = NpyData::from_bytes(&buf)
        .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
    Ok(data.to_vec())
}

pub fn read_spectrum(dir: &Path, teff: f64, m: f64, logg: f64) -> Result<na::DVector<u16>> {
    // e.g. 00027_lm0050_07000_0350_0020_0000_Vsini_0000.npy
    let _teff = teff.round() as i32;
    let _m = (m * 100.0).round() as i32;
    let _logg = (logg * 100.0).round() as i32;
    let sign = if _m < 0 { "m" } else { "p" };
    let filename = format!("l{}{:04}_{:05}_{:04}", sign, _m.abs(), _teff, _logg);
    let file_path = dir.join(format!("{}.npy", filename));
    let result = read_npy_file(file_path.clone())?;
    Ok(na::DVector::from_iterator(result.len(), result.into_iter()))
}

fn labels_from_filename(filename: &str) -> Result<(f64, f64, f64)> {
    // e.g. lp0020_08000_0430.npy
    let parts: Vec<&str> = filename.split(".").nth(0).unwrap().split('_').collect();
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
    Ok((teff, m, logg))
}

fn get_model_labels_in_dir(dir: &PathBuf) -> Result<Vec<(f64, f64, f64)>> {
    std::fs::read_dir(dir)?
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
}

impl OnDiskFetcher {
    pub fn new(dir: &str, vsini_range: (f64, f64), rv_range: (f64, f64)) -> Result<Self> {
        let model_labels = get_model_labels_in_dir(&PathBuf::from(dir))?;
        let grid = Grid::new(model_labels, vsini_range, rv_range)?;
        Ok(Self {
            dir: PathBuf::from(dir),
            grid,
        })
    }
}

impl ModelFetcher for OnDiskFetcher {
    fn grid(&self) -> &Grid {
        &self.grid
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<CowVector> {
        let teff = self.grid.teff.get(i)?;
        let m = self.grid.m.get(j)?;
        let logg = self.grid.logg.get(k)?;
        // println!("Finding i={}, j={}, k={}", i, j, k);
        // println!("teff={}, m={}, logg={}", teff, m, logg);
        let spec = read_spectrum(&self.dir, teff, m, logg)?;
        Ok(CowVector::Owned(spec))
    }
}

#[derive(Clone)]
pub struct InMemFetcher {
    pub grid: Grid,
    pub loaded_spectra: na::DMatrix<u16>,
    pub n: usize,
}

fn load_spectra(dir: PathBuf, grid: &Grid) -> Result<na::DMatrix<u16>> {
    let combinations = grid.list_gridpoints();

    let bar = ProgressBar::new(combinations.len() as u64);
    let vec_of_spectra = combinations
        .into_par_iter()
        .map(|[teff, m, logg]| {
            bar.inc(1);
            read_spectrum(&dir, teff, m, logg)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(na::DMatrix::from_columns(&vec_of_spectra))
}

impl InMemFetcher {
    pub fn new(dir: &str, vsini_range: (f64, f64), rv_range: (f64, f64)) -> Result<Self> {
        let model_labels = get_model_labels_in_dir(&PathBuf::from(dir))?;
        let grid = Grid::new(model_labels, vsini_range, rv_range)?;
        let loaded_spectra = load_spectra(PathBuf::from(dir), &grid)?;
        let n = loaded_spectra.shape().0;
        Ok(Self {
            grid,
            loaded_spectra,
            n,
        })
    }
}

impl ModelFetcher for InMemFetcher {
    fn grid(&self) -> &Grid {
        &self.grid
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<CowVector> {
        let idx = (self.grid.cumulative_grid_size[i] + k - self.grid.logg_limits[i].0)
            * self.grid.m.n()
            + j;
        Ok(CowVector::Borrowed(
            self.loaded_spectra
                .column(idx)
        ))
    }
}

#[derive(Clone)]
pub struct CachedFetcher {
    pub dir: PathBuf,
    pub grid: Grid,
    cache: Arc<Mutex<LruCache<(usize, usize, usize), na::DVector<u16>>>>,
}

impl CachedFetcher {
    pub fn new(
        dir: &str,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
        lrucap: usize,
    ) -> Result<Self> {
        let model_labels = get_model_labels_in_dir(&PathBuf::from(dir))?;
        let grid = Grid::new(model_labels, vsini_range, rv_range)?;
        Ok(Self {
            dir: PathBuf::from(dir),
            grid,
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(lrucap).unwrap(),
            ))),
        })
    }

    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

impl ModelFetcher for CachedFetcher {
    fn grid(&self) -> &Grid {
        &self.grid
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<CowVector> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(spec) = cache.get(&(i, j, k)) {
            Ok(CowVector::Owned(spec.clone()))
        } else {
            std::mem::drop(cache);
            let teff = self.grid.teff.get(i)?;
            let m = self.grid.m.get(j)?;
            let logg = self.grid.logg.get(k)?;
            let spec = read_spectrum(&self.dir, teff, m, logg)?;
            let mut cache = self.cache.lock().unwrap();
            cache.put((i, j, k), spec.clone());
            Ok(CowVector::Owned(spec))
        }
    }
}
