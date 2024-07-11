use crate::interpolate::{
    read_spectrum, ModelFetcher, Range, SquareBounds, SquareGridInterpolator, WlGrid,
};
use anyhow::Result;
use burn::prelude::Backend;
use burn::tensor::{Data, Tensor};
use indicatif::ProgressBar;
use itertools::Itertools;
use lru::LruCache;
use rayon::prelude::*;
use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct OnDiskFetcher<E: Backend> {
    pub dir: PathBuf,
    pub ranges: SquareBounds,
    pub device: E::Device,
}

impl<E: Backend> OnDiskFetcher<E> {
    pub fn new(
        dir: &str,
        teff_range: Range,
        m_range: Range,
        logg_range: Range,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
        device: E::Device,
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

impl<E: Backend> ModelFetcher for OnDiskFetcher<E> {
    type E = E;
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn device(&self) -> &<Self::E as Backend>::Device {
        &self.device
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<Tensor<Self::E, 1>>> {
        let teff = self.ranges.teff.get_value(i);
        let m = self.ranges.m.get_value(j);
        let logg = self.ranges.logg.get_value(k);
        let spec = read_spectrum(&self.dir, teff, m, logg)?;
        Ok(Cow::Owned(Tensor::<E, 1>::from_data(
            Data::from(&spec[..]).convert(),
            &self.device,
        )))
    }
}

pub struct InMemFetcher<E: Backend> {
    pub ranges: SquareBounds,
    pub loaded_spectra: Vec<Tensor<E, 1>>,
    pub device: E::Device,
}

fn load_spectra<E: Backend>(
    dir: PathBuf,
    ranges: &SquareBounds,
    device: &E::Device,
) -> Vec<Tensor<E, 1>> {
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
            let spec = read_spectrum(&dir, teff, m, logg).unwrap();
            Tensor::<E, 1>::from_data(Data::from(&spec[..]).convert(), device)
        })
        .collect()
}

impl<E: Backend> InMemFetcher<E> {
    pub fn new(
        dir: &str,
        teff_range: Range,
        m_range: Range,
        logg_range: Range,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
        device: E::Device,
    ) -> Self {
        let ranges = SquareBounds {
            teff: teff_range,
            m: m_range,
            logg: logg_range,
            vsini: vsini_range,
            rv: rv_range,
        };
        let loaded_spectra = load_spectra(PathBuf::from(dir), &ranges, &device);
        Self {
            loaded_spectra,
            ranges,
            device: device,
        }
    }
}

impl<E: Backend> ModelFetcher for InMemFetcher<E>
where
    <E as Backend>::FloatTensorPrimitive<1>: Sync,
{
    type E = E;
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn device(&self) -> &<Self::E as Backend>::Device {
        &self.device
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<Tensor<Self::E, 1>>> {
        let idx = i * self.ranges.m.n() * self.ranges.logg.n() + j * self.ranges.logg.n() + k;
        Ok(Cow::Borrowed(&self.loaded_spectra[idx]))
    }
}

#[derive(Clone)]
pub struct CachedFetcher<E: Backend> {
    pub dir: PathBuf,
    pub ranges: SquareBounds,
    cache: Arc<Mutex<LruCache<(usize, usize, usize), Vec<f64>>>>,
    pub device: E::Device,
}

impl<E: Backend> CachedFetcher<E> {
    pub fn new(
        dir: &str,
        teff_range: Range,
        m_range: Range,
        logg_range: Range,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
        lrucap: usize,
        device: E::Device,
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

impl<E: Backend> ModelFetcher for CachedFetcher<E> {
    type E = E;
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn device(&self) -> &<Self::E as Backend>::Device {
        &self.device
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<Tensor<Self::E, 1>>> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(spec) = cache.get(&(i, j, k)) {
            let tensor =
                Tensor::<E, 1>::from_data(Data::from(&spec.clone()[..]).convert(), &self.device);
            Ok(Cow::Owned(tensor))
        } else {
            std::mem::drop(cache);
            let teff = self.ranges.teff.get_value(i);
            let m = self.ranges.m.get_value(j);
            let logg = self.ranges.logg.get_value(k);
            let spec = read_spectrum(&self.dir, teff, m, logg)?;
            let tensor = Tensor::<E, 1>::from_data(Data::from(&spec[..]).convert(), &self.device);
            let mut cache = self.cache.lock().unwrap();
            cache.put((i, j, k), spec.clone());
            Ok(Cow::Owned(tensor))
        }
    }
}
