use crate::interpolate::{
    read_spectrum, ModelFetcher, Range, SquareBounds,
};
use crate::tensor::Tensor;
use anyhow::Result;
use indicatif::ProgressBar;
use itertools::Itertools;
use lru::LruCache;
use rayon::prelude::*;
use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct OnDiskFetcher {
    pub dir: PathBuf,
    pub ranges: SquareBounds,
    pub device: tch::Device,
}

impl OnDiskFetcher {
    pub fn new(
        dir: &str,
        teff_range: Range,
        m_range: Range,
        logg_range: Range,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
        device: tch::Device,
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
            device,
        }
    }
}

impl ModelFetcher for OnDiskFetcher {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn device(&self) -> &tch::Device {
        &self.device
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<Tensor>> {
        let teff = self.ranges.teff.get_value(i);
        let m = self.ranges.m.get_value(j);
        let logg = self.ranges.logg.get_value(k);
        let spec = read_spectrum(&self.dir, teff, m, logg)?;
        Ok(Cow::Owned(Tensor::from_slice(&spec[..], self.device())))
    }
}

#[derive(Clone)]
pub struct InMemFetcher {
    pub ranges: SquareBounds,
    pub loaded_spectra: Vec<Tensor>,
    pub n: i64,
    pub zeros: Tensor,
    pub device: tch::Device,
}

fn load_spectra(dir: PathBuf, ranges: &SquareBounds, device: &tch::Device) -> Result<Vec<Tensor>> {
    let combinations: Vec<[f64; 3]> = ranges
        .teff
        .values
        .iter()
        .cartesian_product(ranges.m.values.iter())
        .cartesian_product(ranges.logg.values.iter())
        .map(|((teff, m), logg)| [*teff, *m, *logg])
        .collect();

    let bar = ProgressBar::new(combinations.len() as u64);
    combinations
        .into_par_iter()
        .map(|[teff, m, logg]| {
            bar.inc(1);
            let spec = read_spectrum(&dir, teff, m, logg)?;
            Ok(Tensor::from_slice(&spec[..], device))
        })
        .collect()
}

impl InMemFetcher {
    pub fn new(
        dir: &str,
        teff_range: Range,
        m_range: Range,
        logg_range: Range,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
        device: tch::Device,
    ) -> Result<Self> {
        let ranges = SquareBounds {
            teff: teff_range,
            m: m_range,
            logg: logg_range,
            vsini: vsini_range,
            rv: rv_range,
        };
        let loaded_spectra = load_spectra(PathBuf::from(dir), &ranges, &device)?;
        let n = loaded_spectra[0].dims()[0];
        Ok(Self {
            ranges,
            loaded_spectra,
            n,
            zeros: Tensor::zeros([1, n], &device),
            device: device,
        })
    }
}

impl ModelFetcher for InMemFetcher {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn device(&self) -> &tch::Device {
        &self.device
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<Tensor>> {
        let idx = i * self.ranges.m.n() * self.ranges.logg.n() + j * self.ranges.logg.n() + k;
        Ok(Cow::Borrowed(&self.loaded_spectra[idx]))
    }
}

#[derive(Clone)]
pub struct CachedFetcher {
    pub dir: PathBuf,
    pub ranges: SquareBounds,
    cache: Arc<Mutex<LruCache<(usize, usize, usize), Vec<f32>>>>,
    pub device: tch::Device,
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
        device: tch::Device,
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
            device,
        }
    }
}

impl ModelFetcher for CachedFetcher {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn device(&self) -> &tch::Device {
        &self.device
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<Tensor>> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(spec) = cache.get(&(i, j, k)) {
            let tensor = Tensor::from_slice(&spec.clone()[..], &self.device);
            Ok(Cow::Owned(tensor))
        } else {
            std::mem::drop(cache);
            let teff = self.ranges.teff.get_value(i);
            let m = self.ranges.m.get_value(j);
            let logg = self.ranges.logg.get_value(k);
            let spec = read_spectrum(&self.dir, teff, m, logg)?;
            let tensor = Tensor::from_slice(&spec[..], &self.device);
            let mut cache = self.cache.lock().unwrap();
            cache.put((i, j, k), spec.clone());
            Ok(Cow::Owned(tensor))
        }
    }
}
