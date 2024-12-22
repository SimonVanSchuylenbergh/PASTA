use crate::interpolate::{CowVector, Grid, ModelFetcher};
use anyhow::{anyhow, bail, Context, Result};
use indicatif::ProgressBar;
use nalgebra as na;
use npy::NpyData;
use rayon::prelude::*;
use std::cmp::Eq;
use std::collections::VecDeque;
use std::fs::{read_dir, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tar::Archive;
use zstd::stream::copy_decode;

pub fn read_npy_file(mut file: impl Read) -> Result<Vec<u16>> {
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let data: NpyData<u16> = NpyData::from_bytes(&buf)?;
    Ok(data.to_vec())
}

fn label_from_filename(filename: &str) -> Result<[f64; 3]> {
    // e.g. lp0020_08000_0430.npy
    let parts: Vec<&str> = filename.split('.').next().unwrap().split('_').collect();
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
fn filename_from_labels(teff: f64, m: f64, logg: f64) -> String {
    // e.g. 00027_lm0050_07000_0350_0020_0000_Vsini_0000.npy
    let _teff = teff.round() as i32;
    let _m = (m * 100.0).round() as i32;
    let _logg = (logg * 100.0).round() as i32;
    let sign = if _m < 0 { "m" } else { "p" };
    format!("l{}{:04}_{:05}_{:04}", sign, _m.abs(), _teff, _logg)
}

pub fn read_spectrum(
    file: impl Read,
    includes_factor: bool,
) -> Result<(na::DVector<u16>, Option<f32>)> {
    let result = read_npy_file(file)?;
    if includes_factor {
        let bytes1 = result[0].to_le_bytes();
        let bytes2 = result[1].to_le_bytes();
        let factor = f32::from_le_bytes([bytes1[0], bytes1[1], bytes2[0], bytes2[1]]);
        if factor < 0.0 {
            bail!("Negative factor: {}", factor);
        }
        Ok((
            na::DVector::from_iterator(result.len() - 2, result.into_iter().skip(2)),
            Some(factor / 65535.0),
        ))
    } else {
        Ok((na::DVector::from_iterator(result.len(), result), None))
    }
}

pub fn read_spectrum_from_dir(
    dir: &Path,
    teff: f64,
    m: f64,
    logg: f64,
    includes_factor: bool,
) -> Result<(na::DVector<u16>, Option<f32>)> {
    let filename = filename_from_labels(teff, m, logg);
    let file_path = dir.join(format!("{}.npy", filename));
    let file = File::open(file_path.clone())
        .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
    read_spectrum(file, includes_factor).with_context(|| format!("Error reading {}", filename))
}

fn get_model_labels_in_dir(dir: &PathBuf) -> Result<Vec<[f64; 3]>> {
    read_dir(dir)
        .with_context(|| format!("Not found {:?}", dir))?
        .map(|entry| {
            let path = entry?.path();
            let filename = path.file_name().unwrap().to_str().unwrap();
            label_from_filename(filename)
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
        let (spec, factor) =
            read_spectrum_from_dir(&self.dir, teff, m, logg, self.includes_factor)?;
        Ok((CowVector::Owned(spec), factor.unwrap_or(1.0 / 65535.0)))
    }
}

#[derive(Clone)]
pub struct InMemFetcher {
    pub grid: Grid,
    pub loaded_spectra: na::DMatrix<u16>,
    pub factors: Option<na::DVector<f32>>,
}

impl InMemFetcher {
    pub fn new(dir: &str, includes_factor: bool) -> Result<Self> {
        let model_labels = get_model_labels_in_dir(&PathBuf::from(dir))?;
        let grid = Grid::new(model_labels)?;
        let combinations = grid.list_gridpoints();
        let bar = ProgressBar::new(combinations.len() as u64);
        let vec_of_spectra_and_factors = combinations
            .into_par_iter()
            .map(|[teff, m, logg]| {
                bar.inc(1);
                read_spectrum_from_dir(&PathBuf::from(dir), teff, m, logg, includes_factor)
            })
            .collect::<Result<Vec<_>>>()?;
        let (spectra, factors): (Vec<_>, Vec<_>) = vec_of_spectra_and_factors.into_iter().unzip();
        let (loaded_spectra, factors) = if includes_factor {
            (
                na::DMatrix::from_columns(&spectra),
                factors
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .map(|v| na::DVector::from_iterator(v.len(), v)),
            )
        } else {
            (na::DMatrix::from_columns(&spectra), None)
        };
        let _n = loaded_spectra.shape().0;
        Ok(Self {
            grid,
            loaded_spectra,
            factors,
        })
    }

    pub fn from_tar_zstd(file_path: PathBuf, includes_factor: bool) -> Result<Self> {
        let file = File::open(file_path.clone())
            .context(format!("Error loading {}", file_path.to_str().unwrap()))?;
        let mut buffer = Vec::new();
        copy_decode(file, &mut buffer)?;
        let mut archive = Archive::new(&buffer[..]);
        let model_labels_files = archive
            .entries()?
            .filter_map(|entry| {
                let binding = entry.unwrap();
                let path = binding.path().unwrap();
                let filename = path
                    .file_name()
                    .ok_or(anyhow!("no file name"))
                    .unwrap()
                    .to_str()
                    .unwrap();
                if !filename.starts_with("l") {
                    return None;
                }
                let label = label_from_filename(filename).unwrap();
                let spec = read_spectrum(binding, includes_factor).unwrap();
                Some((label, spec))
            })
            .collect::<Vec<_>>();
        let (model_labels, spectra): (Vec<_>, Vec<_>) = model_labels_files.into_iter().unzip();
        let grid = Grid::new(model_labels.clone())?;
        let combinations = grid.list_gridpoints();
        let vec_of_spectra_and_factors = combinations
            .into_iter()
            .map(|label| {
                let pos = model_labels
                    .iter()
                    .position(|l| *l == label)
                    .ok_or(anyhow!("{:?} not found in archive", label))?;
                let spec: na::DVectorView<u16> = spectra[pos].0.as_view();
                let factor = spectra[pos].1;
                Ok((spec, factor))
            })
            .collect::<Result<Vec<_>>>()?;
        let (spectra, factors): (Vec<_>, Vec<_>) = vec_of_spectra_and_factors.into_iter().unzip();
        let loaded_spectra = na::DMatrix::from_columns(&spectra);
        let factors = if includes_factor {
            factors
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .map(|v| na::DVector::from_iterator(v.len(), v))
        } else {
            None
        };
        let _n = loaded_spectra.shape().0;
        Ok(Self {
            grid,
            loaded_spectra,
            factors,
        })
        // panic!("not implemented")
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
            self.factors
                .as_ref()
                .map(|fac| fac[idx])
                .unwrap_or(1.0 / 65535.0),
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
            let (spec, factor) =
                read_spectrum_from_dir(&self.dir, teff, m, logg, self.includes_factor)?;
            let mut shard = self.shards[shard_idx].write().unwrap();
            shard.put((i, j, k), (spec.clone(), factor.unwrap_or(1.0 / 65535.0)));
            Ok((CowVector::Owned(spec), factor.unwrap_or(1.0 / 65535.0)))
        }
    }
}
