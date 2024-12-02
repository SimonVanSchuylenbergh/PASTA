use crate::interpolate::GridBounds;
use anyhow::Result;
use approx::abs_diff_eq;
use argmin_math::{ArgminRandom, ArgminSub};
use nalgebra as na;
use rand::Rng;
use std::iter::once;

pub trait PSOBounds<const N: usize>: Clone {
    fn outer_limits(&self) -> (na::SVector<f64, N>, na::SVector<f64, N>);
    fn get_limits_at(&self, param: na::SVector<f64, N>, index: usize) -> Result<(f64, f64)>;
    fn is_within_bounds(&self, param: na::SVector<f64, N>) -> bool;
    fn clamp_1d(&self, param: na::SVector<f64, N>, index: usize) -> Result<f64> {
        let limits = self.get_limits_at(param, index)?;
        Ok(param[index].clamp(limits.0, limits.1))
    }
    fn widths(&self) -> na::SVector<f64, N> {
        let (min, max) = self.outer_limits();
        max.sub(&min)
    }
    fn generate_random_within_bounds(
        &self,
        rng: &mut impl Rng,
        num_particles: usize,
    ) -> Vec<na::SVector<f64, N>> {
        let mut particles = Vec::with_capacity(num_particles);
        let (min, max) = self.outer_limits();
        while particles.len() < num_particles {
            let param = na::SVector::rand_from_range(&min, &max, rng);
            if self.is_within_bounds(param) {
                particles.push(param);
            }
        }
        particles
    }

    fn with_constraints(
        &self,
        constraints: Vec<BoundsConstraint>,
    ) -> ConstrainedPSOBounds<N, Self> {
        ConstrainedPSOBounds::from_list(self.clone(), constraints)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Constraint {
    Fixed(f64),
    Range(f64, f64),
    None,
}

impl Constraint {
    pub fn intersect(&self, min: f64, max: f64) -> (f64, f64) {
        match self {
            Constraint::Fixed(f) => (*f, *f),
            Constraint::Range(a, b) => (min.max(*a), max.min(*b)),
            Constraint::None => (min, max),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundsConstraint {
    pub parameter: usize,
    pub constraint: Constraint,
}

#[derive(Clone)]
pub struct ConstrainedPSOBounds<const N: usize, B: PSOBounds<N>> {
    pso_bounds: B,
    constraints: [Constraint; N],
}

impl<const N: usize, B: PSOBounds<N>> ConstrainedPSOBounds<N, B> {
    fn from_list(pso_bounds: B, bounds_constraints: Vec<BoundsConstraint>) -> Self {
        let mut constraints = [Constraint::None; N];
        for c in bounds_constraints {
            constraints[c.parameter] = c.constraint;
        }
        Self {
            pso_bounds,
            constraints,
        }
    }
}

fn intersect_limits<const N: usize>(
    constraints: [Constraint; N],
    min: na::SVector<f64, N>,
    max: na::SVector<f64, N>,
) -> (na::SVector<f64, N>, na::SVector<f64, N>) {
    (
        na::SVector::from_iterator(min.into_iter().zip(constraints).map(|(m, c)| match c {
            Constraint::Fixed(f) => f,
            Constraint::Range(a, _) => m.max(a),
            Constraint::None => *m,
        })),
        na::SVector::from_iterator(max.into_iter().zip(constraints).map(|(m, c)| match c {
            Constraint::Fixed(f) => f,
            Constraint::Range(_, b) => m.min(b),
            Constraint::None => *m,
        })),
    )
}

impl<const N: usize, B: PSOBounds<N>> PSOBounds<N> for ConstrainedPSOBounds<N, B> {
    fn outer_limits(&self) -> (na::SVector<f64, N>, na::SVector<f64, N>) {
        let (min, max) = self.pso_bounds.outer_limits();
        intersect_limits(self.constraints, min, max)
    }

    fn get_limits_at(&self, param: nalgebra::SVector<f64, N>, index: usize) -> Result<(f64, f64)> {
        let (min, max) = self.pso_bounds.get_limits_at(param, index)?;
        Ok(self.constraints[index].intersect(min, max))
    }

    fn is_within_bounds(&self, param: nalgebra::SVector<f64, N>) -> bool {
        if !self.pso_bounds.is_within_bounds(param) {
            return false;
        }
        param
            .iter()
            .zip(self.constraints.iter())
            .all(|(p, c)| match c {
                Constraint::Fixed(f) => abs_diff_eq!(p, f),
                Constraint::Range(a, b) => *p >= *a && *p <= *b,
                Constraint::None => true,
            })
    }
}

#[derive(Clone)]
pub struct SingleBounds<B: GridBounds> {
    grid: B,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
}

impl<B: GridBounds> SingleBounds<B> {
    pub fn new(grid: B, vsini_range: (f64, f64), rv_range: (f64, f64)) -> Self {
        Self {
            grid,
            vsini_range,
            rv_range,
        }
    }
}

impl<B: GridBounds> PSOBounds<5> for SingleBounds<B> {
    fn outer_limits(&self) -> (nalgebra::SVector<f64, 5>, nalgebra::SVector<f64, 5>) {
        let (min, max) = self.grid.limits();
        (
            nalgebra::SVector::from_iterator(
                min.iter()
                    .copied()
                    .chain(once(self.vsini_range.0))
                    .chain(once(self.rv_range.0)),
            ),
            nalgebra::SVector::from_iterator(
                max.iter()
                    .copied()
                    .chain(once(self.vsini_range.1))
                    .chain(once(self.rv_range.1)),
            ),
        )
    }

    fn get_limits_at(&self, param: nalgebra::SVector<f64, 5>, index: usize) -> Result<(f64, f64)> {
        match index {
            3 => Ok(self.vsini_range),
            4 => Ok(self.rv_range),
            i => self
                .grid
                .get_limits_at(param.fixed_rows::<3>(0).into_owned(), i),
        }
    }

    fn is_within_bounds(&self, param: nalgebra::SVector<f64, 5>) -> bool {
        self.grid
            .is_within_bounds(param.fixed_rows::<3>(0).into_owned())
            && param[3] >= self.vsini_range.0
            && param[3] <= self.vsini_range.1
            && param[4] >= self.rv_range.0
            && param[4] <= self.rv_range.1
    }
}

#[derive(Clone)]
pub struct BinaryBounds<B: GridBounds> {
    grid: B,
    light_ratio_range: (f64, f64),
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
}

impl<B: GridBounds> BinaryBounds<B> {
    pub fn new(
        grid: B,
        light_ratio: (f64, f64),
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
    ) -> Self {
        Self {
            grid,
            light_ratio_range: light_ratio,
            vsini_range,
            rv_range,
        }
    }
}

impl<B: GridBounds> PSOBounds<11> for BinaryBounds<B> {
    fn outer_limits(&self) -> (na::SVector<f64, 11>, na::SVector<f64, 11>) {
        let (min, max) = self.grid.limits();
        (
            na::SVector::from_iterator(
                min.iter()
                    .copied()
                    .chain(once(self.vsini_range.0))
                    .chain(once(self.rv_range.0))
                    .chain(min.iter().copied())
                    .chain(once(self.vsini_range.0))
                    .chain(once(self.rv_range.0))
                    .chain(once(self.light_ratio_range.0)),
            ),
            na::SVector::from_iterator(
                max.iter()
                    .copied()
                    .chain(once(self.vsini_range.1))
                    .chain(once(self.rv_range.1))
                    .chain(max.iter().copied())
                    .chain(once(self.vsini_range.1))
                    .chain(once(self.rv_range.1))
                    .chain(once(self.light_ratio_range.1)),
            ),
        )
    }

    fn get_limits_at(&self, param: nalgebra::SVector<f64, 11>, index: usize) -> Result<(f64, f64)> {
        if index >= 11 {
            panic!("Index for binary clamp_1d out out bounds")
        }
        match index {
            3 => Ok(self.vsini_range),
            4 => Ok(self.rv_range),
            8 => Ok(self.vsini_range),
            9 => Ok(self.rv_range),
            10 => Ok(self.light_ratio_range),
            i if i < 5 => self
                .grid
                .get_limits_at(param.fixed_rows::<3>(0).into_owned(), i),
            i => self
                .grid
                .get_limits_at(param.fixed_rows::<3>(5).into_owned(), i - 5),
        }
    }

    fn is_within_bounds(&self, param: nalgebra::SVector<f64, 11>) -> bool {
        self.grid
            .is_within_bounds(param.fixed_rows::<3>(0).into_owned())
            && self
                .grid
                .is_within_bounds(param.fixed_rows::<3>(5).into_owned())
            && param[3] >= self.vsini_range.0
            && param[3] <= self.vsini_range.1
            && param[4] >= self.rv_range.0
            && param[4] <= self.rv_range.1
            && param[8] >= self.vsini_range.0
            && param[8] <= self.vsini_range.1
            && param[9] >= self.rv_range.0
            && param[9] <= self.rv_range.1
            && param[10] >= self.light_ratio_range.0
            && param[10] <= self.light_ratio_range.1
    }
}

#[derive(Clone)]
pub struct BinaryBoundsWithoutRV<B: GridBounds> {
    grid: B,
    light_ratio: (f64, f64),
    vsini_range: (f64, f64),
}

impl<B: GridBounds> BinaryBoundsWithoutRV<B> {
    pub fn new(grid: B, light_ratio: (f64, f64), vsini_range: (f64, f64)) -> Self {
        Self {
            grid,
            light_ratio,
            vsini_range,
        }
    }
}

impl<B: GridBounds> PSOBounds<9> for BinaryBoundsWithoutRV<B> {
    fn outer_limits(&self) -> (na::SVector<f64, 9>, na::SVector<f64, 9>) {
        let (min, max) = self.grid.limits();
        (
            na::SVector::from_iterator(
                min.iter()
                    .copied()
                    .chain(once(self.vsini_range.0))
                    .chain(min.iter().copied())
                    .chain(once(self.vsini_range.0))
                    .chain(once(self.light_ratio.0)),
            ),
            na::SVector::from_iterator(
                max.iter()
                    .copied()
                    .chain(once(self.vsini_range.1))
                    .chain(max.iter().copied())
                    .chain(once(self.vsini_range.1))
                    .chain(once(self.light_ratio.1)),
            ),
        )
    }

    fn get_limits_at(&self, param: na::SVector<f64, 9>, index: usize) -> Result<(f64, f64)> {
        match index {
            3 => Ok(self.vsini_range),
            7 => Ok(self.vsini_range),
            i => self
                .grid
                .get_limits_at(param.fixed_rows::<3>(0).into_owned(), i),
        }
    }

    fn is_within_bounds(&self, param: nalgebra::SVector<f64, 9>) -> bool {
        self.grid
            .is_within_bounds(param.fixed_rows::<3>(0).into_owned())
            && self
                .grid
                .is_within_bounds(param.fixed_rows::<3>(4).into_owned())
            && param[3] >= self.vsini_range.0
            && param[3] <= self.vsini_range.1
            && param[7] >= self.vsini_range.0
            && param[7] <= self.vsini_range.1
            && param[8] >= self.light_ratio.0
            && param[8] <= self.light_ratio.1
    }
}

#[derive(Clone)]
pub struct BinaryRVBounds {
    rv_range: (f64, f64),
}

impl BinaryRVBounds {
    pub fn new(rv_range: (f64, f64)) -> Self {
        Self { rv_range }
    }
}

impl PSOBounds<2> for BinaryRVBounds {
    fn outer_limits(&self) -> (na::SVector<f64, 2>, na::SVector<f64, 2>) {
        (
            na::Vector2::new(self.rv_range.0, self.rv_range.0),
            na::Vector2::new(self.rv_range.1, self.rv_range.1),
        )
    }

    fn get_limits_at(&self, _param: na::SVector<f64, 2>, _index: usize) -> Result<(f64, f64)> {
        Ok(self.rv_range)
    }

    fn is_within_bounds(&self, param: na::SVector<f64, 2>) -> bool {
        param[0] >= self.rv_range.0
            && param[0] <= self.rv_range.1
            && param[1] >= self.rv_range.0
            && param[1] <= self.rv_range.1
    }
}
