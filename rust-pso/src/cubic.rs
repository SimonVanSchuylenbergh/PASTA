use crate::interpolate::{Range, SquareBounds};
use anyhow::{bail, Result};
use burn::tensor::{backend::Backend, Tensor};
use nalgebra as na;

type SVectorView<'a, const N: usize, const M: usize> = na::Matrix<
    f64,
    na::Const<N>,
    na::Const<1>,
    na::ViewStorage<'a, f64, na::Const<N>, na::Const<1>, na::Const<1>, na::Const<M>>,
>;

pub struct InterpolInput<E: Backend> {
    pub factors: Tensor<E, 2>, // (3, 4) tensor, factors to multiply with the 4 neighbors
    pub shape: [usize; 3],
    pub local_4x4x4_indices: [[isize; 3]; 64],
}

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

fn calculate_factors_quadratic(x: f64, x0: f64, x1: f64, x2: f64) -> [f32; 4] {
    let xsq = x * x;

    let col0_denom = x0 * x0 - x0 * x1 - x0 * x2 + x1 * x2;
    let col1_denom = -x1 * x1 + x0 * x1 - x0 * x2 + x1 * x2;
    let col2_denom = x2 * x2 + x0 * x1 - x0 * x2 - x1 * x2;

    let f0 = ((xsq - (x1 + x2) * x + x1 * x2) / col0_denom) as f32;
    let f1 = ((-xsq + (x0 + x2) * x - x0 * x2) / col1_denom) as f32;
    let f2 = ((xsq - (x0 + x1) * x + x0 * x1) / col2_denom) as f32;
    [f0, f1, f2, 0.0]
}

fn calculate_factors_cubic(x: f64, x0: f64, x1: f64, x2: f64, x3: f64) -> [f32; 4] {
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
        / col0_denom) as f32;
    let f1 = ((-xcu + (x0 + x2 + x3) * xsq - (x0 * x2 + x0 * x3 + x2 * x3) * x + (x0 * x2 * x3))
        / col1_denom) as f32;
    let f2 = ((xcu - (x0 + x1 + x3) * xsq + (x0 * x1 + x0 * x3 + x1 * x3) * x - (x0 * x1 * x3))
        / col2_denom) as f32;
    let f3 = ((-xcu + (x0 + x1 + x2) * xsq - (x0 * x1 + x0 * x2 + x1 * x2) * x + (x0 * x1 * x2))
        / col3_denom) as f32;
    [f0, f1, f2, f3]
}

pub fn prepare_interpolate<E: Backend>(
    ranges: &SquareBounds,
    teff: f64,
    m: f64,
    logg: f64,
    device: &E::Device,
) -> Result<InterpolInput<E>> {
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

    let factors_teff = if (teff_range.len() == 3) {
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
    let factors_m = if (m_range.len() == 3) {
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
    let factors_logg = if (logg_range.len() == 3) {
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

    let factors = Tensor::from_floats(
        [
            factors_teff[0],
            factors_teff[1],
            factors_teff[2],
            factors_teff[3],
            factors_m[0],
            factors_m[1],
            factors_m[2],
            factors_m[3],
            factors_logg[0],
            factors_logg[1],
            factors_logg[2],
            factors_logg[3],
        ],
        device,
    )
    .reshape([3, 4]);

    let dis = get_indices(i, &ranges.teff);
    let djs = get_indices(j, &ranges.m);
    let dks = get_indices(k, &ranges.logg);
    let shape = [dis.len(), djs.len(), dks.len()];
    let mut local_4x4x4_indices = [[0; 3]; 64];
    for index_i in 0..4 {
        for index_j in 0..4 {
            for index_k in 0..4 {
                if index_i < dis.len() && index_j < djs.len() && index_k < dks.len() {
                    local_4x4x4_indices[index_i * 16 + index_j * 4 + index_k] = [
                        i as isize + dis[index_i],
                        j as isize + djs[index_j],
                        k as isize + dks[index_k],
                    ]
                }
            }
        }
    }

    Ok(InterpolInput {
        factors,
        local_4x4x4_indices,
        shape,
    })
}

/// Interpolates a 1D quadratic function.
/// factors: Tensor with factors to multiply with the neighbors (4)
/// yp: Tensor with three model spectra (4, N) (last row is thrown away)
fn quadratic_1d<E: Backend>(factors: Tensor<E, 1>, yp: Tensor<E, 2>) -> Tensor<E, 1> {
    let y = yp.narrow(0, 0, 3); // Throw away last row
    let factors = factors.narrow(0, 0, 3); // Throw away last factor

    (factors.unsqueeze_dim(1) * y).sum_dim(0).squeeze(0)
}

/// Interpolates a 1D cubic function.
/// factors: Tensor with factors to multiply with the neighbors (4)
/// yp: Tensor with four model spectra (4, N)
fn cubic_1d<E: Backend>(factors: Tensor<E, 1>, yp: Tensor<E, 2>) -> Tensor<E, 1> {
    (factors.unsqueeze_dim(1) * yp).sum_dim(0).squeeze(0)
}

/// Interpolates a 2D cubic function.
/// factors: Tensor with factors to multiply with the neighbors (2, 4)
/// yp: Tensor with 16 model spectra (4, 4, N)
/// shape: Equals 3 or 4 for each dimension (quadratic or cubic)
fn cubic_2d<E: Backend>(
    factors: Tensor<E, 2>,
    shape: [usize; 2],
    yp: Tensor<E, 3>,
) -> Tensor<E, 1> {
    // 1D subcoordinates
    let logg_factors = factors.clone().narrow(0, 1, 1).squeeze(0); // logg factors (4)
                                                                   // Interpolate in logg
    let local1d = Tensor::stack(
        yp.iter_dim(0)
            .map(|t| {
                let subtensor = t.squeeze(0);
                if shape[1] == 3 {
                    quadratic_1d(logg_factors.clone(), subtensor)
                } else if shape[1] == 4 {
                    cubic_1d(logg_factors.clone(), subtensor)
                } else {
                    panic!("Second dimension of shape is neither 3 nor 4")
                }
            })
            .collect(),
        0,
    );

    // Interpolate in m
    let m_factors = factors.clone().narrow(0, 0, 1).squeeze(0); // m factors (4)
    if shape[0] == 4 {
        cubic_1d(m_factors, local1d)
    } else if shape[0] == 3 {
        quadratic_1d(m_factors, local1d)
    } else {
        panic!("First dimension of shape is neither 3 nor 4")
    }
}

/// Interpolates a 3D cubic function.
/// factors: Tensor with factors to multiply with the neighbors (3, 4) (teff, m, logg)
/// yp: Tensor with 64 model spectra (4, 4, 4, N)
pub fn cubic_3d<E: Backend>(
    factors: Tensor<E, 2>,
    shape: [usize; 3],
    yp: Tensor<E, 4>,
) -> Tensor<E, 1> {
    // Subgrid in m, logg
    let m_logg_shape: [usize; 2] = [shape[1], shape[2]];
    let m_logg_factors = factors.clone().narrow(0, 1, 2); // m, logg factors (2, 4)
                                                          // Interpolate in logg and m
    let local1d = Tensor::stack(
        yp.iter_dim(0)
            .map(|t| {
                let subtensor = t.squeeze(0);
                cubic_2d(m_logg_factors.clone(), m_logg_shape, subtensor)
            })
            .collect(),
        0,
    ); // (4, N) tensor

    // Interpolate in teff
    let teff_factors = factors.narrow(0, 0, 1).squeeze(0); // teff factors (4)
    if shape[0] == 4 {
        cubic_1d(teff_factors, local1d)
    } else if shape[0] == 3 {
        quadratic_1d(teff_factors, local1d)
    } else {
        panic!("First dimension of shape is neither 3 nor 4")
    }
}
