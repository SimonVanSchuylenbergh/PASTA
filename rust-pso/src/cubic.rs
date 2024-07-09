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
    // let y0 = yp[0];
    // let y1 = yp[1];
    // let y2 = yp[2];
    let device = yp.device();
    let y = yp.narrow(0, 0, 3);

    let col0_denom = 1.0 / (x0 * x0 - x0 * x1 - x0 * x2 + x1 * x2);
    let col1_denom = 1.0 / (-x1 * x1 + x0 * x1 - x0 * x2 + x1 * x2);
    let col2_denom = 1.0 / (x2 * x2 + x0 * x1 - x0 * x2 - x1 * x2);

    // let a = y0 * col0_denom - y1 * col1_denom + y2 * col2_denom;
    // let b =
    //     -y0 * (x1 + x2) * col0_denom + y1 * (x0 + x2) * col1_denom - y2 * (x0 + x1) * col2_denom;
    // let c = y0 * x1 * x2 * col0_denom - y1 * x0 * x2 * col1_denom + y2 * x0 * x1 * col2_denom;
    let af = Tensor::from_floats(
        [col0_denom as f32, -col1_denom as f32, col2_denom as f32],
        &device,
    );
    let bf = Tensor::from_floats(
        [
            (-(x1 + x2) * col0_denom) as f32,
            ((x0 + x2) * col1_denom) as f32,
            (-(x0 + x1) * col2_denom) as f32,
        ],
        &device,
    );
    let cf = Tensor::from_floats(
        [
            (x1 * x2 * col0_denom) as f32,
            (-x0 * x2 * col1_denom) as f32,
            (x0 * x1 * col2_denom) as f32,
        ],
        &device,
    );

    let a = (af.unsqueeze_dim(1) * y.clone()).sum_dim(0).squeeze(0);
    let b = (bf.unsqueeze_dim(1) * y.clone()).sum_dim(0).squeeze(0);
    let c = (cf.unsqueeze_dim(1) * y).sum_dim(0).squeeze(0);

    a * x * x + b * x + c
}

fn cubic_1d<E: Backend>(x: f64, xp: SVectorView<4, 12>, yp: Tensor<E, 2>) -> Tensor<E, 1> {
    let x0 = xp[0];
    let x1 = xp[1];
    let x2 = xp[2];
    let x3 = xp[3];
    // let y0 = yp[0];
    // let y1 = yp[1];
    // let y2 = yp[2];
    // let y3 = yp[3];

    let col0_denom = 1.0
        / (x0 * x0 * x0 - x0 * x0 * x1 - x0 * x0 * x2 - x0 * x0 * x3
            + x0 * x1 * x2
            + x0 * x1 * x3
            + x0 * x2 * x3
            - x1 * x2 * x3);
    let col1_denom = 1.0
        / (-x1 * x1 * x1 + x1 * x1 * x0 + x1 * x1 * x2 + x1 * x1 * x3
            - x0 * x1 * x2
            - x0 * x1 * x3
            + x0 * x2 * x3
            - x1 * x2 * x3);
    let col2_denom = 1.0
        / (x2 * x2 * x2 - x2 * x2 * x0 - x2 * x2 * x1 - x2 * x2 * x3 + x0 * x1 * x2 - x0 * x1 * x3
            + x0 * x2 * x3
            + x1 * x2 * x3);
    let col3_denom = 1.0
        / (-x3 * x3 * x3 + x3 * x3 * x0 + x3 * x3 * x1 + x3 * x3 * x2 + x0 * x1 * x2
            - x0 * x1 * x3
            - x0 * x2 * x3
            - x1 * x2 * x3);

    // let a = y0 * col0_denom - y1 * col1_denom + y2 * col2_denom - y3 * col3_denom;
    // let b = y0 * (-x1 - x2 - x3) * col0_denom
    //     + y1 * (x0 + x2 + x3) * col1_denom
    //     + y2 * (-x0 - x1 - x3) * col2_denom
    //     + y3 * (x0 + x1 + x2) * col3_denom;
    // let c = y0 * (x1 * x2 + x1 * x3 + x2 * x3) * col0_denom
    //     + y1 * (-x0 * x2 - x0 * x3 - x2 * x3) * col1_denom
    //     + y2 * (x0 * x1 + x0 * x3 + x1 * x3) * col2_denom
    //     + y3 * (-x0 * x1 - x0 * x2 - x1 * x2) * col3_denom;
    // let d = -y0 * x1 * x2 * x3 * col0_denom + y1 * x0 * x2 * x3 * col1_denom
    //     - y2 * x0 * x1 * x3 * col2_denom
    //     + y3 * x0 * x1 * x2 * col3_denom;

    let device = yp.device();
    let af = Tensor::from_floats(
        [
            col0_denom as f32,
            -col1_denom as f32,
            col2_denom as f32,
            -col3_denom as f32,
        ],
        &device,
    );
    let bf = Tensor::from_floats(
        [
            (-(x1 + x2 + x3) * col0_denom) as f32,
            ((x0 + x2 + x3) * col1_denom) as f32,
            (-(x0 + x1 + x3) * col2_denom) as f32,
            ((x0 + x1 + x2) * col3_denom) as f32,
        ],
        &device,
    );
    let cf = Tensor::from_floats(
        [
            ((x1 * x2 + x1 * x3 + x2 * x3) * col0_denom) as f32,
            ((-x0 * x2 - x0 * x3 - x2 * x3) * col1_denom) as f32,
            ((x0 * x1 + x0 * x3 + x1 * x3) * col2_denom) as f32,
            ((-x0 * x1 - x0 * x2 - x1 * x2) * col3_denom) as f32,
        ],
        &device,
    );
    let df = Tensor::from_floats(
        [
            ((x1 * x2 * x3) * col0_denom) as f32,
            ((-x0 * x2 * x3) * col1_denom) as f32,
            ((x0 * x1 * x3) * col2_denom) as f32,
            ((-x0 * x1 * x2) * col3_denom) as f32,
        ],
        &device,
    );

    let a = (af.unsqueeze_dim(1) * yp.clone()).sum_dim(0).squeeze(0);
    let b = (bf.unsqueeze_dim(1) * yp.clone()).sum_dim(0).squeeze(0);
    let c = (cf.unsqueeze_dim(1) * yp.clone()).sum_dim(0).squeeze(0);
    let d = (df.unsqueeze_dim(1) * yp).sum_dim(0).squeeze(0);

    a * x * x * x + b * x * x + c * x + d
}

fn cubic_2d<E: Backend>(
    x: [f64; 2],
    xp: SVectorView<8, 12>,
    yp: Tensor<E, 3>, // was (16, 64)
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
}
