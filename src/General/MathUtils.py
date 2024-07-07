import numpy as np
import torch


def map_range(
    x: float,
    from_low: float,
    from_high: float,
    to_low: float,
    to_high: float
) -> float:
    return (x - from_low) / (from_high - from_low) * (to_high - to_low) + to_low


def gaussian(x, x0, a, sigma):
    return a * np.exp(-0.5 * (x - x0) ** 2 / sigma**2)


def vsini_kernel(vsini: float, spacing=5.428688517383762e-06):
    p_halfwidth = vsini / (spacing * np.log(10) * 3e5)
    kernel = np.arange(float(-p_halfwidth), float(p_halfwidth + 1))

    if len(kernel) <= 3:
        return np.array([1])

    kernel[0] = 0
    kernel[-1] = 0
    kernel[1:-1] = np.sqrt(1 - (kernel[1:-1] / p_halfwidth) ** 2)
    kernel /= np.sum(kernel)
    return kernel



def interp_tensor(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)

    return m[indices] * x + b[indices]

