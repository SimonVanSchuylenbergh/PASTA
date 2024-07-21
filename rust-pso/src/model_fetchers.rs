use crate::interpolate::{read_spectrum, CowVector, FluxFloat, ModelFetcher, Range, SquareBounds};
use anyhow::Result;
use indicatif::ProgressBar;
use itertools::Itertools;
use lru::LruCache;
use nalgebra as na;
use rayon::prelude::*;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct OnDiskFetcher {
    pub dir: PathBuf,
    pub ranges: SquareBounds,
}

impl OnDiskFetcher {
    pub fn new(
        dir: &str,
        teff_range: Range,
        m_range: Range,
        logg_range: Range,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
    ) -> Self {
        let ranges = SquareBounds {
            teff: teff_range,
            m: m_range,
            logg: logg_range,
            vsini: vsini_range,
            rv: rv_range,
        };
        Self {
            dir: PathBuf::from(dir),
            ranges,
        }
    }
}

impl ModelFetcher for OnDiskFetcher {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<CowVector> {
        let teff = self.ranges.teff.get_value(i);
        let m = self.ranges.m.get_value(j);
        let logg = self.ranges.logg.get_value(k);
        let spec = read_spectrum(&self.dir, teff, m, logg)?;
        Ok(CowVector::Owned(spec))
    }
}

#[derive(Clone)]
pub struct InMemFetcher {
    pub ranges: SquareBounds,
    pub loaded_spectra: na::DMatrix<FluxFloat>,
    pub n: usize,
}

fn load_spectra(dir: PathBuf, ranges: &SquareBounds) -> Result<na::DMatrix<FluxFloat>> {
    let combinations: Vec<[f64; 3]> = ranges
        .teff
        .values
        .iter()
        .cartesian_product(ranges.m.values.iter())
        .cartesian_product(ranges.logg.values.iter())
        .map(|((teff, m), logg)| [*teff, *m, *logg])
        .collect();

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
    pub fn new(
        dir: &str,
        teff_range: Range,
        m_range: Range,
        logg_range: Range,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
    ) -> Result<Self> {
        let ranges = SquareBounds {
            teff: teff_range,
            m: m_range,
            logg: logg_range,
            vsini: vsini_range,
            rv: rv_range,
        };
        let loaded_spectra = load_spectra(PathBuf::from(dir), &ranges)?;
        let n = loaded_spectra.shape().0;
        Ok(Self {
            ranges,
            loaded_spectra,
            n,
        })
    }
}

impl ModelFetcher for InMemFetcher {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<CowVector> {
        let idx = i * self.ranges.m.n() * self.ranges.logg.n() + j * self.ranges.logg.n() + k;
        Ok(CowVector::Borrowed(self.loaded_spectra.column(idx)))
    }
}

#[derive(Clone)]
pub struct CachedFetcher {
    pub dir: PathBuf,
    pub ranges: SquareBounds,
    cache: Arc<Mutex<LruCache<(usize, usize, usize), na::DVector<FluxFloat>>>>,
}

impl CachedFetcher {
    pub fn new(
        dir: &str,
        teff_range: Range,
        m_range: Range,
        logg_range: Range,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
        lrucap: usize,
    ) -> Self {
        let ranges = SquareBounds {
            teff: teff_range,
            m: m_range,
            logg: logg_range,
            vsini: vsini_range,
            rv: rv_range,
        };
        Self {
            dir: PathBuf::from(dir),
            ranges,
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(lrucap).unwrap(),
            ))),
        }
    }

    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

impl ModelFetcher for CachedFetcher {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<CowVector> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(spec) = cache.get(&(i, j, k)) {
            Ok(CowVector::Owned(spec.clone()))
        } else {
            std::mem::drop(cache);
            let teff = self.ranges.teff.get_value(i);
            let m = self.ranges.m.get_value(j);
            let logg = self.ranges.logg.get_value(k);
            let spec = read_spectrum(&self.dir, teff, m, logg)?;
            let mut cache = self.cache.lock().unwrap();
            cache.put((i, j, k), spec.clone());
            Ok(CowVector::Owned(spec))
        }
    }
}
