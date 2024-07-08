use nalgebra as na;

type SVectorView<'a, const N: usize, const M: usize> = na::Matrix<
    f64,
    na::Const<N>,
    na::Const<1>,
    na::ViewStorage<'a, f64, na::Const<N>, na::Const<1>, na::Const<1>, na::Const<M>>,
>;

fn quadratic_1d<const N: usize>(x: f64, xp: SVectorView<4, 12>, yp: SVectorView<4, N>) -> f64 {
    let x0 = xp[0];
    let x1 = xp[1];
    let x2 = xp[2];
    let y0 = yp[0];
    let y1 = yp[1];
    let y2 = yp[2];

    let col0_denom = 1.0 / (x0 * x0 - x0 * x1 - x0 * x2 + x1 * x2);
    let col1_denom = 1.0 / (-x1 * x1 + x0 * x1 - x0 * x2 + x1 * x2);
    let col2_denom = 1.0 / (x2 * x2 + x0 * x1 - x0 * x2 - x1 * x2);

    let a = y0 * col0_denom - y1 * col1_denom + y2 * col2_denom;
    let b =
        -y0 * (x1 + x2) * col0_denom + y1 * (x0 + x2) * col1_denom - y2 * (x0 + x1) * col2_denom;
    let c = y0 * x1 * x2 * col0_denom - y1 * x0 * x2 * col1_denom + y2 * x0 * x1 * col2_denom;

    a * x * x + b * x + c
}

fn cubic_1d<const N: usize>(x: f64, xp: SVectorView<4, 12>, yp: SVectorView<4, N>) -> f64 {
    let x0 = xp[0];
    let x1 = xp[1];
    let x2 = xp[2];
    let x3 = xp[3];
    let y0 = yp[0];
    let y1 = yp[1];
    let y2 = yp[2];
    let y3 = yp[3];

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

    let a = y0 * col0_denom - y1 * col1_denom + y2 * col2_denom - y3 * col3_denom;
    let b = y0 * (-x1 - x2 - x3) * col0_denom
        + y1 * (x0 + x2 + x3) * col1_denom
        + y2 * (-x0 - x1 - x3) * col2_denom
        + y3 * (x0 + x1 + x2) * col3_denom;
    let c = y0 * (x1 * x2 + x1 * x3 + x2 * x3) * col0_denom
        + y1 * (-x0 * x2 - x0 * x3 - x2 * x3) * col1_denom
        + y2 * (x0 * x1 + x0 * x3 + x1 * x3) * col2_denom
        + y3 * (-x0 * x1 - x0 * x2 - x1 * x2) * col3_denom;
    let d = -y0 * x1 * x2 * x3 * col0_denom + y1 * x0 * x2 * x3 * col1_denom
        - y2 * x0 * x1 * x3 * col2_denom
        + y3 * x0 * x1 * x2 * col3_denom;

    a * x * x * x + b * x * x + c * x + d
}

fn cubic_2d(
    x: [f64; 2],
    xp: SVectorView<8, 12>,
    yp: SVectorView<16, 64>,
    shape: [usize; 2],
) -> f64 {
    let mut local1d = na::SVector::<f64, 4>::zeros();
    let xp1d = xp.fixed_rows::<4>(4);
    for i in 0..4 {
        let yp1d = yp.fixed_rows::<4>(i * 4);
        let sub_y = if shape[1] == 3 {
            quadratic_1d(x[1], xp1d, yp1d)
        } else if shape[1] == 4 {
            cubic_1d(x[1], xp1d, yp1d)
        } else {
            panic!("OH NO PANIC!!! (Second dimension of shape is neither 3 nor 4)")
        };
        local1d[i] = sub_y;
    }

    let localxp = xp.fixed_rows::<4>(0);
    if shape[0] == 4 {
        cubic_1d(x[0], localxp, local1d.as_view())
    } else if shape[0] == 3 {
        quadratic_1d(x[0], localxp, local1d.as_view())
    } else {
        panic!("OH NO PANIC!!! (First dimension of shape is neither 3 nor 4)")
    }
}

pub fn cubic_3d(
    coord: [f64; 3],
    xp: &na::SVector<f64, 12>,
    yp: &na::SVector<f64, 64>,
    shape: [usize; 3],
) -> f64 {
    let mut local1d = na::SVector::<f64, 4>::zeros();
    let subcoord: [f64; 2] = [coord[1], coord[2]];
    let subshape: [usize; 2] = [shape[1], shape[2]];

    let xp2d = xp.fixed_rows::<8>(4);
    for i in 0..4 {
        let sub_y = cubic_2d(subcoord, xp2d, yp.fixed_rows::<16>(i * 16), subshape);
        local1d[i] = sub_y
    }

    let localxp = xp.fixed_rows::<4>(0);
    if shape[0] == 4 {
        cubic_1d(coord[0], localxp, local1d.as_view())
    } else if shape[0] == 3 {
        quadratic_1d(coord[0], localxp, local1d.as_view())
    } else {
        panic!("OH NO PANIC!!! (First dimension of shape is neither 3 nor 4)")
    }
}
