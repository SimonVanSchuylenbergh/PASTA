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

pub struct InterpolInput {
    pub coord: [f64; 3],
    pub xp: na::SVector<f64, 12>,
    pub local_4x4x4_indices: [[isize; 3]; 64],
    pub shape: [usize; 3],
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

pub fn prepare_interpolate(
    ranges: &SquareBounds,
    teff: f64,
    m: f64,
    logg: f64,
) -> Result<InterpolInput> {
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

    let dis = get_indices(i, &ranges.teff);
    let djs = get_indices(j, &ranges.m);
    let dks = get_indices(k, &ranges.logg);

    let teff_range = get_range(i, &ranges.teff);
    let m_range = get_range(j, &ranges.m);
    let logg_range = get_range(k, &ranges.logg);

    let mut xp = na::SVector::zeros();
    let teff_min = teff_range.first().unwrap();
    let teff_max = teff_range.last().unwrap();
    let delta_teff = teff_max - teff_min;
    let m_min = m_range.first().unwrap();
    let m_max = m_range.last().unwrap();
    let delta_m = m_max - m_min;
    let logg_min = logg_range.first().unwrap();
    let logg_max = logg_range.last().unwrap();
    let delta_logg = logg_max - logg_min;

    for i in 0..teff_range.len() {
        xp[i] = (teff_range[i] - teff_min) / delta_teff;
    }
    for i in 0..m_range.len() {
        xp[i + 4] = (m_range[i] - m_min) / delta_m;
    }
    for i in 0..logg_range.len() {
        xp[i + 8] = (logg_range[i] - logg_min) / delta_logg;
    }
    let coord = [
        (teff - teff_min) / delta_teff,
        (m - m_min) / delta_m,
        (logg - logg_min) / delta_logg,
    ];

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
        coord,
        xp,
        local_4x4x4_indices,
        shape,
    })
}

/// Interpolates a 1D quadratic function.
/// x: coordinate compared to neighbors between 0 and 1
/// xp: x coordinates of the 3 neighbors
/// yp: Tensor with three model spectra (4, N) (last row is thrown away)
fn quadratic_1d<E: Backend>(x: f64, xp: SVectorView<4, 12>, yp: Tensor<E, 2>) -> Tensor<E, 1> {
    let x0 = xp[0]; // Should be 0
    let x1 = xp[1];
    let x2 = xp[2]; // Should be 1

    let xsq = x * x;
    let device = yp.device();
    let y = yp.narrow(0, 0, 3); // Throw away last row

    let col0_denom = x0 * x0 - x0 * x1 - x0 * x2 + x1 * x2;
    let col1_denom = -x1 * x1 + x0 * x1 - x0 * x2 + x1 * x2;
    let col2_denom = x2 * x2 + x0 * x1 - x0 * x2 - x1 * x2;

    let f0 = ((xsq - (x1 + x2) * x + x1 * x2) / col0_denom) as f32;
    let f1 = ((-xsq + (x0 + x2) * x - x0 * x2) / col1_denom) as f32;
    let f2 = ((xsq - (x0 + x1) * x + x0 * x1) / col2_denom) as f32;

    let f = Tensor::from_floats([f0, f1, f2], &device);

    (f.unsqueeze_dim(1) * y).sum_dim(0).squeeze(0)
}

/// Interpolates a 1D cubic function.
/// x: coordinate compared to neighbors between 0 and 1
/// xp: x coordinates of the 4 neighbors
/// yp: Tensor with four model spectra (4, N)
fn cubic_1d<E: Backend>(x: f64, xp: SVectorView<4, 12>, yp: Tensor<E, 2>) -> Tensor<E, 1> {
    let x0 = xp[0]; // Should be 0
    let x1 = xp[1];
    let x2 = xp[2];
    let x3 = xp[3]; // Should be 1

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

    let f = Tensor::from_floats([f0, f1, f2, f3], &yp.device());

    (f.unsqueeze_dim(1) * yp).sum_dim(0).squeeze(0)
}

/// Interpolates a 2D cubic function.
/// x: coordinate compared to neighbors, between 0 and 1, [m, logg]
/// xp: x coordinates of the 8 neighbors
/// yp: Tensor with 16 model spectra (4, 4, N)
/// shape: Equals 3 or 4 for each dimension (quadratic or cubic)
fn cubic_2d<E: Backend>(
    x: [f64; 2],
    xp: SVectorView<8, 12>,
    yp: Tensor<E, 3>,
    shape: [usize; 2],
) -> Tensor<E, 1> {
    // 1D subcoordinates
    let xp1d = xp.fixed_rows::<4>(4);

    // Interpolate in logg
    let local1d = Tensor::stack(
        yp.iter_dim(0)
            .map(|t| {
                let subtensor = t.squeeze(0);
                if shape[1] == 3 {
                    quadratic_1d(x[1], xp1d, subtensor)
                } else if shape[1] == 4 {
                    cubic_1d(x[1], xp1d, subtensor)
                } else {
                    panic!("Second dimension of shape is neither 3 nor 4")
                }
            })
            .collect(),
        0,
    );

    // Interpolate in m
    let localxp = xp.fixed_rows::<4>(0);
    if shape[0] == 4 {
        cubic_1d(x[0], localxp, local1d)
    } else if shape[0] == 3 {
        quadratic_1d(x[0], localxp, local1d)
    } else {
        panic!("First dimension of shape is neither 3 nor 4")
    }
}

/// Interpolates a 3D cubic function.
/// coord: coordinate compared to neighbors, between 0 and 1, [teff, m, logg]
/// xp: x coordinates of the 12 neighbors
/// yp: Tensor with 64 model spectra (4, 4, 4, N)
pub fn cubic_3d<E: Backend>(
    coord: [f64; 3],
    xp: &na::SVector<f64, 12>,
    yp: Tensor<E, 4>,
    shape: [usize; 3],
) -> Tensor<E, 1> {
    // Subgrid in m, logg
    let subcoord: [f64; 2] = [coord[1], coord[2]];
    let subshape: [usize; 2] = [shape[1], shape[2]];
    let xp2d = xp.fixed_rows::<8>(4);

    // Interpolate in logg and m
    let local1d = Tensor::stack(
        yp.iter_dim(0)
            .map(|t| {
                let subtensor = t.squeeze(0);
                cubic_2d(subcoord, xp2d, subtensor, subshape)
            })
            .collect(),
        0,
    ); // (4, N) tensor

    // Interpolate in teff
    let localxp = xp.fixed_rows::<4>(0);
    if shape[0] == 4 {
        cubic_1d(coord[0], localxp, local1d)
    } else if shape[0] == 3 {
        quadratic_1d(coord[0], localxp, local1d)
    } else {
        panic!("First dimension of shape is neither 3 nor 4")
    }
}
