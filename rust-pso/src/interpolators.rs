use crate::interpolate::{read_spectrum, Range, SquareBounds, SquareGridInterpolator, WlGrid};
use anyhow::Result;
use indicatif::ProgressBar;
use itertools::Itertools;
use lru::LruCache;
use nalgebra as na;
use rayon::prelude::*;
use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct OnDiskInterpolator {
    pub dir: PathBuf,
    pub synth_wl: WlGrid,
    pub ranges: SquareBounds,
}

impl OnDiskInterpolator {
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
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
            synth_wl: wavelength,
            ranges,
        }
    }
}

impl SquareGridInterpolator for OnDiskInterpolator {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn synth_wl(&self) -> WlGrid {
        self.synth_wl
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<na::DVector<f64>>> {
        let teff = self.ranges.teff.get_value(i);
        let m = self.ranges.m.get_value(j);
        let logg = self.ranges.logg.get_value(k);
        Ok(Cow::Owned(read_spectrum(&self.dir, teff, m, logg)?))
    }
}

pub struct InMemInterpolator {
    pub synth_wl: WlGrid, // min, step
    pub ranges: SquareBounds,
    pub loaded_spectra: Vec<na::DVector<f64>>,
}

fn load_spectra(dir: PathBuf, ranges: &SquareBounds) -> Vec<na::DVector<f64>> {
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
            read_spectrum(&dir, teff, m, logg).unwrap()
        })
        .collect()
}

impl InMemInterpolator {
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
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
        let loaded_spectra = load_spectra(PathBuf::from(dir), &ranges);
        Self {
            loaded_spectra,
            synth_wl: wavelength,
            ranges,
        }
    }
}

impl SquareGridInterpolator for InMemInterpolator {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn synth_wl(&self) -> WlGrid {
        self.synth_wl
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<na::DVector<f64>>> {
        let idx = i * self.ranges.m.n() * self.ranges.logg.n() + j * self.ranges.logg.n() + k;
        Ok(Cow::Borrowed(&self.loaded_spectra[idx]))
    }
}

#[derive(Clone)]
pub struct CachedInterpolator {
    pub dir: PathBuf,
    pub synth_wl: WlGrid,
    pub ranges: SquareBounds,
    cache: Arc<Mutex<LruCache<(usize, usize, usize), na::DVector<f64>>>>,
}

impl CachedInterpolator {
    pub fn new(
        dir: &str,
        wavelength: WlGrid,
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
            synth_wl: wavelength,
            ranges,
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(lrucap).unwrap(),
            ))),
        }
    }
}

impl SquareGridInterpolator for CachedInterpolator {
    fn ranges(&self) -> &SquareBounds {
        &self.ranges
    }
    fn synth_wl(&self) -> WlGrid {
        self.synth_wl
    }

    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<Cow<na::DVector<f64>>> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(spec) = cache.get(&(i, j, k)) {
            Ok(Cow::Owned(spec.clone()))
        } else {
            std::mem::drop(cache);
            let teff = self.ranges.teff.get_value(i);
            let m = self.ranges.m.get_value(j);
            let logg = self.ranges.logg.get_value(k);
            let spec = read_spectrum(&self.dir, teff, m, logg)?;
            let mut cache = self.cache.lock().unwrap();
            cache.put((i, j, k), spec.clone());
            Ok(Cow::Owned(spec))
        }
    }
}
