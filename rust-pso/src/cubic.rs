use crate::interpolate::{Range, SquareBounds};
use crate::tensor::Tensor;
use anyhow::{bail, Result};
use itertools::Itertools;

fn get_indices(index: usize, range: &Range) -> Vec<isize> {
    if index == 0 {
        vec![0, 1, 2]
    } else if index == range.n() - 2 {
        vec![-1, 0, 1]
    } else {
        vec![-1, 0, 1, 2]
    }
}

fn get_range(index: usize, range: &Range) -> Vec<f64> {
    if index == 0 {
        range.values[0..3].into()
    } else if index == range.n() - 2 {
        range.values[index - 1..index + 2].into()
    } else {
        range.values[index - 1..index + 3].into()
    }
}

fn calculate_factors_quadratic(x: f64, x0: f64, x1: f64, x2: f64) -> [f64; 4] {
    let xsq = x * x;

    let col0_denom = x0 * x0 - x0 * x1 - x0 * x2 + x1 * x2;
    let col1_denom = -x1 * x1 + x0 * x1 - x0 * x2 + x1 * x2;
    let col2_denom = x2 * x2 + x0 * x1 - x0 * x2 - x1 * x2;

    let f0 = ((xsq - (x1 + x2) * x + x1 * x2) / col0_denom);
    let f1 = ((-xsq + (x0 + x2) * x - x0 * x2) / col1_denom);
    let f2 = ((xsq - (x0 + x1) * x + x0 * x1) / col2_denom);
    [f0, f1, f2, 0.0]
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

    let f0 = ((xcu - (x1 + x2 + x3) * xsq + (x1 * x2 + x1 * x3 + x2 * x3) * x - (x1 * x2 * x3))
        / col0_denom);
    let f1 = ((-xcu + (x0 + x2 + x3) * xsq - (x0 * x2 + x0 * x3 + x2 * x3) * x + (x0 * x2 * x3))
        / col1_denom);
    let f2 = ((xcu - (x0 + x1 + x3) * xsq + (x0 * x1 + x0 * x3 + x1 * x3) * x - (x0 * x1 * x3))
        / col2_denom);
    let f3 = ((-xcu + (x0 + x1 + x2) * xsq - (x0 * x1 + x0 * x2 + x1 * x2) * x + (x0 * x1 * x2))
        / col3_denom);
    [f0, f1, f2, f3]
}

pub fn calculate_interpolation_coefficients_flat(
    ranges: &SquareBounds,
    teff: f64,
    m: f64,
    logg: f64,
    device: &tch::Device,
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

    let teff_range = get_range(i, &ranges.teff);
    let m_range = get_range(j, &ranges.m);
    let logg_range = get_range(k, &ranges.logg);

    let teff_min = teff_range.first().unwrap();
    let teff_max = teff_range.last().unwrap();
    let delta_teff = teff_max - teff_min;
    let m_min = m_range.first().unwrap();
    let m_max = m_range.last().unwrap();
    let delta_m = m_max - m_min;
    let logg_min = logg_range.first().unwrap();
    let logg_max = logg_range.last().unwrap();
    let delta_logg = logg_max - logg_min;

    let factors_teff = if teff_range.len() == 3 {
        calculate_factors_quadratic(
            (teff - teff_min) / delta_teff,
            0.0,
            (teff_range[1] - teff_min) / delta_teff,
            1.0,
        )
    } else {
        calculate_factors_cubic(
            (teff - teff_min) / delta_teff,
            0.0,
            (teff_range[1] - teff_min) / delta_teff,
            (teff_range[2] - teff_min) / delta_teff,
            1.0,
        )
    };
    let factors_m = if m_range.len() == 3 {
        calculate_factors_quadratic(
            (m - m_min) / delta_m,
            0.0,
            (m_range[1] - m_min) / delta_m,
            1.0,
        )
    } else {
        calculate_factors_cubic(
            (m - m_min) / delta_m,
            0.0,
            (m_range[1] - m_min) / delta_m,
            (m_range[2] - m_min) / delta_m,
            1.0,
        )
    };
    let factors_logg = if logg_range.len() == 3 {
        calculate_factors_quadratic(
            (logg - logg_min) / delta_logg,
            0.0,
            (logg_range[1] - logg_min) / delta_logg,
            1.0,
        )
    } else {
        calculate_factors_cubic(
            (logg - logg_min) / delta_logg,
            0.0,
            (logg_range[1] - logg_min) / delta_logg,
            (logg_range[2] - logg_min) / delta_logg,
            1.0,
        )
    };
    Ok([factors_teff, factors_m, factors_logg])
}

pub fn calculate_interpolation_coefficients(
    ranges: &SquareBounds,
    teff: f64,
    m: f64,
    logg: f64,
    device: &tch::Device,
) -> Result<Tensor> {
    let [factors_teff, factors_m, factors_logg] =
        calculate_interpolation_coefficients_flat(ranges, teff, m, logg, device)?;
    let all_factors: Vec<f32> = factors_teff
        .iter()
        .cartesian_product(factors_m)
        .cartesian_product(factors_logg)
        .map(|((f_teff, f_m), f_logg)| (f_teff * f_m * f_logg) as f32)
        .collect();

    let factors = Tensor::from_slice(&all_factors[..], device);
    Ok(factors)
}
