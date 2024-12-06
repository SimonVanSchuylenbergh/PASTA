use crate::interpolate::FluxFloat;
use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use nalgebra as na;

fn calculate_factors_linear(x: f64, x0: f64, x1: f64) -> [f64; 2] {
    let f0 = (x - x1) / (x0 - x1);
    let f1 = (x - x0) / (x1 - x0);
    [f0, f1]
}

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

pub enum Neighbors {
    LeftOnly([f64; 3]),
    Both([f64; 4]),
    RightOnly([f64; 3]),
}

pub fn calculate_factors(x: f64, neighbors: Neighbors) -> na::SVector<f64, 4> {
    match neighbors {
        Neighbors::LeftOnly([x0, x1, x2]) => calculate_factors_quadratic(
            (x - x0) / (x2 - x0),
            0.0,
            (x1 - x0) / (x2 - x0),
            1.0,
            false,
        )
        .into(),
        Neighbors::Both([x0, x1, x2, x3]) => calculate_factors_cubic(
            (x - x0) / (x3 - x0),
            0.0,
            (x1 - x0) / (x3 - x0),
            (x2 - x0) / (x3 - x0),
            1.0,
        )
        .into(),
        Neighbors::RightOnly([x0, x1, x2]) => {
            calculate_factors_quadratic((x - x0) / (x2 - x0), 0.0, (x1 - x0) / (x2 - x0), 1.0, true)
                .into()
        }
    }
}

fn map_columns<const R: usize, const C: usize>(
    mat: &na::SMatrix<Option<(usize, usize)>, R, C>,
    f: impl Fn(usize, na::SVectorView<Option<(usize, usize)>, R>) -> Result<na::SVector<f64, R>>,
) -> Result<na::SMatrix<f64, R, C>> {
    let mut result: na::SMatrix<f64, R, C> = na::SMatrix::zeros();
    for i in 0..C {
        result.set_column(i, &f(i, mat.column(i))?);
    }
    Ok(result)
}

fn map_rows<const R: usize, const C: usize>(
    mat: &na::SMatrix<Option<(usize, usize)>, R, C>,
    f: impl Fn(usize, na::SVector<Option<(usize, usize)>, C>) -> Result<na::SVector<f64, C>>,
) -> Result<na::SMatrix<f64, R, C>> {
    let mut result: na::SMatrix<f64, R, C> = na::SMatrix::zeros();
    for i in 0..R {
        result.set_row(i, &f(i, mat.row(i).transpose())?.transpose());
    }
    Ok(result)
}

// Represent the 4x4x4 grid of neighbors between which will be interpolated
pub struct LocalGrid {
    // Teff value to be interpolated to, and Teff value of the 4 neighbors
    // Values can be None if the point is outside of the grid.
    // The interpolator will fall back to quadratic interpolation in that case.
    pub teff: (f64, [Option<f64>; 4]),
    // logg value to be interpolated to, and logg value of the 4 neighbors
    pub logg: (f64, [Option<f64>; 4]),
    // 4x4 matrix of the indices of the teff and logg neighbors.
    pub teff_logg_indices: na::SMatrix<Option<(usize, usize)>, 4, 4>,
    // M value to be interpolated to, and M value of the 4 neighbors
    pub m: (f64, [Option<f64>; 4]),
    // Indices of the M neighbors
    pub m_indices: na::SVector<Option<usize>, 4>,
}

// Represent the 2x2x2 grid of neighbors used for linear interpolation
// Similar to LocalGrid, but with only 2 neighbors in each dimension
// In this case all neighbors need to exist, so there are no Option fields
pub struct LocalGridLinear {
    pub teff: (f64, [f64; 2]),
    pub teff_indices: na::SVector<usize, 2>,
    pub logg: (f64, [f64; 2]),
    pub logg_indices: na::SVector<usize, 2>,
    pub m: (f64, [f64; 2]),
    pub m_indices: na::SVector<usize, 2>,
}

fn coefficients_from_neighbors(x: f64, neighbors: [Option<f64>; 4]) -> Result<na::SVector<f64, 4>> {
    match neighbors {
        [Some(n0), Some(n1), Some(n2), Some(n3)] => {
            Ok(calculate_factors(x, Neighbors::Both([n0, n1, n2, n3])))
        }
        [Some(n0), Some(n1), Some(n2), None] => {
            Ok(calculate_factors(x, Neighbors::LeftOnly([n0, n1, n2])))
        }
        [None, Some(n1), Some(n2), Some(n3)] => {
            Ok(calculate_factors(x, Neighbors::RightOnly([n1, n2, n3])))
        }
        [None, None, None, None] => Ok(na::SVector::zeros()),
        _ => Err(anyhow!("Not enough neighbors: {}, {:?}", x, neighbors)),
    }
}

pub fn calculate_interpolation_coefficients_flat(
    local_grid: &LocalGrid,
) -> Result<(
    na::SMatrix<f64, 4, 4>,
    na::SMatrix<f64, 4, 4>,
    na::SVector<f64, 4>,
)> {
    let (teff, teff_neighbors) = local_grid.teff;
    let (logg, logg_neighbors) = local_grid.logg;
    let (m, m_neighbors) = local_grid.m;
    let factors_teff = map_rows(&local_grid.teff_logg_indices, |_, row| {
        coefficients_from_neighbors(
            teff,
            teff_neighbors
                .iter()
                .zip(row.iter())
                .map(|(value, index)| match index {
                    Some(_) => {
                        Ok(Some(value.ok_or_else(|| {
                            anyhow!("No Teff value for this neighbor")
                        })?))
                    }
                    None => Ok(None),
                })
                .collect::<Result<Vec<_>>>()
                .with_context(|| {
                    anyhow!(
                        "teff={}, neighbors: {:?}, row: {:?}",
                        teff,
                        teff_neighbors,
                        row
                    )
                })?
                .as_slice()
                .try_into()?,
        )
        .with_context(|| format!("Error get teff coefficients: row={:?}", row))
    })?;
    let factors_logg = map_columns(&local_grid.teff_logg_indices, |_, col| {
        coefficients_from_neighbors(
            logg,
            logg_neighbors
                .iter()
                .zip(col.iter())
                .map(|(value, index)| match index {
                    Some(_) => {
                        Ok(Some(value.ok_or_else(|| {
                            anyhow!("No logg value for this neighbor")
                        })?))
                    }
                    None => Ok(None),
                })
                .collect::<Result<Vec<_>>>()
                .with_context(|| {
                    anyhow!(
                        "logg={}, neighbors: {:?}, col: {:?}",
                        logg,
                        logg_neighbors,
                        col
                    )
                })?
                .as_slice()
                .try_into()?,
        )
        .context("Error get logg coefficients")
    })?;
    let factors_m =
        coefficients_from_neighbors(m, m_neighbors).context("Error get m coefficients")?;

    Ok((factors_teff, factors_logg, factors_m))
}

// Calculate the weights of the 64 neighbors in the cubic interpolation. 
pub fn calculate_interpolation_coefficients(
    local_grid: &LocalGrid,
) -> Result<na::SVector<FluxFloat, 64>> {
    let (factors_teff, factors_logg, factors_m) =
    calculate_interpolation_coefficients_flat(local_grid)?;
    Ok(na::SVector::from_iterator(
        (factors_teff.component_mul(&factors_logg))
        .iter()
        .cartesian_product(factors_m.iter())
        .map(|(a, b)| (a * b) as f32),
    ))
}


// Calculate the weights of the 8 neighbors in the linear interpolation. 
pub fn calculate_interpolation_coefficients_linear(
    local_grid: &LocalGridLinear,
) -> Result<na::SVector<FluxFloat, 8>> {
    let LocalGridLinear {
        teff: (teff, teff_neighbors),
        logg: (logg, logg_neighbors),
        m: (m, m_neighbors),
        ..
    } = local_grid;

    let factors_teff = calculate_factors_linear(*teff, teff_neighbors[0], teff_neighbors[1]);
    let factors_logg = calculate_factors_linear(*logg, logg_neighbors[0], logg_neighbors[1]);
    let factors_m = calculate_factors_linear(*m, m_neighbors[0], m_neighbors[1]);
    Ok(na::SVector::from_iterator(
        factors_teff
            .iter()
            .cartesian_product(factors_logg.iter())
            .cartesian_product(factors_m.iter())
            .map(|((a, b), c)| (a * b * c) as FluxFloat),
    ))
}
