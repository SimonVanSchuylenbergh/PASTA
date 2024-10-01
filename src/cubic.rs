use crate::interpolate::{FluxFloat, Grid, Range};
use anyhow::{bail, Result};
use itertools::Itertools;
use nalgebra as na;

fn calculate_factors_quadratic(x: f64, x0: f64, x1: f64, x2: f64, start: bool) -> [f64; 4] {
    let xsq = x * x;

    let col0_denom = x0 * x0 - x0 * x1 - x0 * x2 + x1 * x2;
    let col1_denom = -x1 * x1 + x0 * x1 - x0 * x2 + x1 * x2;
    let col2_denom = x2 * x2 + x0 * x1 - x0 * x2 - x1 * x2;

    let f0 = (xsq - (x1 + x2) * x + x1 * x2) / col0_denom;
    let f1 = (-xsq + (x0 + x2) * x - x0 * x2) / col1_denom;
    let f2 = (xsq - (x0 + x1) * x + x0 * x1) / col2_denom;
    if start {
        [0.0, f0, f1, f2]
    } else {
        [f0, f1, f2, 0.0]
    }
}

fn calculate_factors_cubic(x: f64, x0: f64, x1: f64, x2: f64, x3: f64) -> [f64; 4] {
    let xsq = x * x;
    let xcu = xsq * x;

    let col0_denom = x0 * x0 * x0 - x0 * x0 * x1 - x0 * x0 * x2 - x0 * x0 * x3
        + x0 * x1 * x2
        + x0 * x1 * x3
        + x0 * x2 * x3
        - x1 * x2 * x3;
    let col1_denom =
        -x1 * x1 * x1 + x1 * x1 * x0 + x1 * x1 * x2 + x1 * x1 * x3 - x0 * x1 * x2 - x0 * x1 * x3
            + x0 * x2 * x3
            - x1 * x2 * x3;
    let col2_denom = x2 * x2 * x2 - x2 * x2 * x0 - x2 * x2 * x1 - x2 * x2 * x3 + x0 * x1 * x2
        - x0 * x1 * x3
        + x0 * x2 * x3
        + x1 * x2 * x3;
    let col3_denom = -x3 * x3 * x3 + x3 * x3 * x0 + x3 * x3 * x1 + x3 * x3 * x2 + x0 * x1 * x2
        - x0 * x1 * x3
        - x0 * x2 * x3
        - x1 * x2 * x3;

    let f0 = (xcu - (x1 + x2 + x3) * xsq + (x1 * x2 + x1 * x3 + x2 * x3) * x - (x1 * x2 * x3))
        / col0_denom;
    let f1 = (-xcu + (x0 + x2 + x3) * xsq - (x0 * x2 + x0 * x3 + x2 * x3) * x + (x0 * x2 * x3))
        / col1_denom;
    let f2 = (xcu - (x0 + x1 + x3) * xsq + (x0 * x1 + x0 * x3 + x1 * x3) * x - (x0 * x1 * x3))
        / col2_denom;
    let f3 = (-xcu + (x0 + x1 + x2) * xsq - (x0 * x1 + x0 * x2 + x1 * x2) * x + (x0 * x1 * x2))
        / col3_denom;
    [f0, f1, f2, f3]
}

pub fn calculate_factors(x: f64, index: usize, range: &Range, limits: (usize, usize)) -> [f64; 4] {
    if index == limits.0 {
        // Quadratic interpolation on left edge
        let neighbors = [range.values[0], range.values[1], range.values[2]];
        calculate_factors_quadratic(
            (x - neighbors[0]) / (neighbors[2] - neighbors[0]),
            0.0,
            (neighbors[1] - neighbors[0]) / (neighbors[2] - neighbors[0]),
            1.0,
            true,
        )
    } else if index == limits.1 {
        // Quadratic interpolation on right edge
        let neighbors = [
            range.values[index - 1],
            range.values[index],
            range.values[index + 1],
        ];
        calculate_factors_quadratic(
            (x - neighbors[0]) / (neighbors[2] - neighbors[0]),
            0.0,
            (neighbors[1] - neighbors[0]) / (neighbors[2] - neighbors[0]),
            1.0,
            false,
        )
    } else if index > 0 && index < range.n() - 2 {
        // Cubic interpolation
        let neighbors = [
            range.values[index - 1],
            range.values[index],
            range.values[index + 1],
            range.values[index + 2],
        ];
        calculate_factors_cubic(
            (x - neighbors[0]) / (neighbors[3] - neighbors[0]),
            0.0,
            (neighbors[1] - neighbors[0]) / (neighbors[3] - neighbors[0]),
            (neighbors[2] - neighbors[0]) / (neighbors[3] - neighbors[0]),
            1.0,
        )
    } else {
        panic!("Index out of bounds ({})", index);
    }
}

pub fn calculate_interpolation_coefficients_flat(
    grid: &Grid,
    teff: f64,
    m: f64,
    logg: f64,
) -> Result<[[f64; 4]; 3]> {
    let i = grid.teff.get_left_index(teff).unwrap();
    let j = grid.m.get_left_index(m).unwrap();
    let k = grid.logg.get_left_index(logg).unwrap();

    let factors_teff = calculate_factors(
        teff,
        i,
        &grid.teff,
        grid.get_teff_index_limits_at(logg).unwrap(),
    );
    let factors_m = calculate_factors(m, j, &grid.m, (0, grid.m.n() - 1));
    let factors_logg = calculate_factors(
        logg,
        k,
        &grid.logg,
        grid.get_logg_index_limits_at(teff).unwrap(),
    );

    Ok([factors_teff, factors_m, factors_logg])
}

pub fn calculate_interpolation_coefficients(
    grid: &Grid,
    teff: f64,
    m: f64,
    logg: f64,
) -> Result<na::DVector<FluxFloat>> {
    if !grid.is_teff_logg_between_bounds(teff, logg) {
        bail!("Teff, logg out of bounds ({}, {})", teff, logg);
    }
    if !grid.is_m_between_bounds(m) {
        bail!("M out of bounds ({})", m);
    }
    let [factors_teff, factors_m, factors_logg] =
        calculate_interpolation_coefficients_flat(grid, teff, m, logg)?;
    Ok(na::DVector::from_iterator(
        64,
        factors_teff
            .iter()
            .cartesian_product(factors_m)
            .cartesian_product(factors_logg)
            .map(|((f_teff, f_m), f_logg)| (f_teff * f_m * f_logg) as FluxFloat),
    ))
}
