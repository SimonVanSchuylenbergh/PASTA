use burn::tensor::{backend::Backend, Tensor};
use nalgebra as na;

type SVectorView<'a, const N: usize, const M: usize> = na::Matrix<
    f64,
    na::Const<N>,
    na::Const<1>,
    na::ViewStorage<'a, f64, na::Const<N>, na::Const<1>, na::Const<1>, na::Const<M>>,
>;

fn quadratic_1d<E: Backend>(x: f64, xp: SVectorView<4, 12>, yp: Tensor<E, 2>) -> Tensor<E, 1> {
    let x0 = xp[0];
    let x1 = xp[1];
    let x2 = xp[2];

    let xsq = x * x;

    let device = yp.device();
    let y = yp.narrow(0, 0, 3);

    let col0_denom = x0 * x0 - x0 * x1 - x0 * x2 + x1 * x2;
    let col1_denom = -x1 * x1 + x0 * x1 - x0 * x2 + x1 * x2;
    let col2_denom = x2 * x2 + x0 * x1 - x0 * x2 - x1 * x2;

    let f0 = ((xsq - (x1 + x2) * x + x1 * x2) / col0_denom) as f32;
    let f1 = ((-xsq + (x0 + x2) * x - x0 * x2) / col1_denom) as f32;
    let f2 = ((xsq - (x0 + x1) * x + x0 * x1) / col2_denom) as f32;

    let f = Tensor::from_floats([f0, f1, f2], &device);

    (f.unsqueeze_dim(1) * y).sum_dim(0).squeeze(0)
}

fn cubic_1d<E: Backend>(x: f64, xp: SVectorView<4, 12>, yp: Tensor<E, 2>) -> Tensor<E, 1> {
    let x0 = xp[0];
    let x1 = xp[1];
    let x2 = xp[2];
    let x3 = xp[3];

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

    let device = yp.device();
    let f0 = ((xcu - (x1 + x2 + x3) * xsq + (x1 * x2 + x1 * x3 + x2 * x3) * x - (x1 * x2 * x3))
        / col0_denom) as f32;
    let f1 = ((-xcu + (x0 + x2 + x3) * xsq - (x0 * x2 + x0 * x3 + x2 * x3) * x + (x0 * x2 * x3))
        / col1_denom) as f32;
    let f2 = ((xcu - (x0 + x1 + x3) * xsq + (x0 * x1 + x0 * x3 + x1 * x3) * x - (x0 * x1 * x3))
        / col2_denom) as f32;
    let f3 = ((-xcu + (x0 + x1 + x2) * xsq - (x0 * x1 + x0 * x2 + x1 * x2) * x + (x0 * x1 * x2))
        / col3_denom) as f32;

    let f = Tensor::from_floats([f0, f1, f2, f3], &device);

    (f.unsqueeze_dim(1) * yp).sum_dim(0).squeeze(0)
}

fn cubic_2d<E: Backend>(
    x: [f64; 2],
    xp: SVectorView<8, 12>,
    yp: Tensor<E, 3>, // (4, 4, N)
    shape: [usize; 2],
) -> Tensor<E, 1> {
    let xp1d = xp.fixed_rows::<4>(4);

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

    let localxp = xp.fixed_rows::<4>(0);
    if shape[0] == 4 {
        cubic_1d(x[0], localxp, local1d)
    } else if shape[0] == 3 {
        quadratic_1d(x[0], localxp, local1d)
    } else {
        panic!("First dimension of shape is neither 3 nor 4")
    }
}

pub fn cubic_3d<E: Backend>(
    coord: [f64; 3], // Coordinate compared to neighbors, between 0 and 1, [teff, m, logg]
    xp: &na::SVector<f64, 12>, // x coordinates of the 3D grid
    yp: Tensor<E, 4>, // (4, 4, 4, N) tensor of neighboring grid spectra
    shape: [usize; 3], // 3 or 4 for each dimension (quadratic or cubic)
) -> Tensor<E, 1> {
    // Subgrid in m, logg
    let subcoord: [f64; 2] = [coord[1], coord[2]];
    let subshape: [usize; 2] = [shape[1], shape[2]];
    let xp2d = xp.fixed_rows::<8>(4);
    // Interpolate in m, logg
    // (4, N)
    let local1d = Tensor::stack(
        yp.iter_dim(0)
            .map(|t| {
                let subtensor = t.squeeze(0);
                cubic_2d(subcoord, xp2d, subtensor, subshape)
            })
            .collect(),
        0,
    );

    let localxp = xp.fixed_rows::<4>(0);
    if shape[0] == 4 {
        cubic_1d(coord[0], localxp, local1d)
    } else if shape[0] == 3 {
        quadratic_1d(coord[0], localxp, local1d)
    } else {
        panic!("OH NO PANIC!!! (First dimension of shape is neither 3 nor 4)")
    }
    // yp.narrow(0, 0, 1).squeeze::<3>(0).narrow(0, 0, 1).squeeze::<2>(0).narrow(0, 0, 1).squeeze::<1>(0)
}
