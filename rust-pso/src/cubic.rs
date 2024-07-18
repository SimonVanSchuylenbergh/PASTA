use crate::interpolate::{FluxFloat, Range, SquareBounds};
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

pub fn calculate_factors(x: f64, index: usize, range: &Range) -> [f64; 4] {
    if index == 0 {
        // Quadratic interpolation on left edge
        let neighbors = [range.values[0], range.values[1], range.values[2]];
        calculate_factors_quadratic(
            (x - neighbors[0]) / (neighbors[2] - neighbors[0]),
            0.0,
            (neighbors[1] - neighbors[0]) / (neighbors[2] - neighbors[0]),
            1.0,
            true,
        )
    } else if index == range.n() - 2 {
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
    } else {
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
    }
}

pub fn calculate_interpolation_coefficients_flat(
    ranges: &SquareBounds,
    teff: f64,
    m: f64,
    logg: f64,
) -> Result<[[f64; 4]; 3]> {
    // 3D cubic interpolation followed by convolution
    if !ranges.teff.between_bounds(teff) {
        bail!("Teff out of bounds ({})", teff);
    }
    if !ranges.m.between_bounds(m) {
        bail!("M out of bounds ({})", m);
    }
    if !ranges.logg.between_bounds(logg) {
        bail!("Logg out of bounds ({})", logg);
    }

    let i = ranges.teff.get_left_index(teff);
    let j = ranges.m.get_left_index(m);
    let k = ranges.logg.get_left_index(logg);

    let factors_teff = calculate_factors(teff, i, &ranges.teff);
    let factors_m = calculate_factors(m, j, &ranges.m);
    let factors_logg = calculate_factors(logg, k, &ranges.logg);

    Ok([factors_teff, factors_m, factors_logg])
}

pub fn calculate_interpolation_coefficients(
    ranges: &SquareBounds,
    teff: f64,
    m: f64,
    logg: f64,
) -> Result<na::DVector<FluxFloat>> {
    let [factors_teff, factors_m, factors_logg] =
        calculate_interpolation_coefficients_flat(ranges, teff, m, logg)?;
    Ok(na::DVector::from_iterator(
        64,
        factors_teff
            .iter()
            .cartesian_product(factors_m)
            .cartesian_product(factors_logg)
            .map(|((f_teff, f_m), f_logg)| (f_teff * f_m * f_logg) as FluxFloat),
    ))
}
