import numpy as np


def mapRange(
    x: float,
    from_low: float,
    from_high: float,
    to_low: float,
    to_high: float,
    clamp=False,
) -> float:
    """
    If `x` is a quantity between `from_low` and `from_high`, this function returns the value `y`
    that corresponds to the same relative position in the interval between `to_low` and `to_high`.
    If `clamp` is `True`, the output is clamped between `to_low` and `to_high`.
    """
    if clamp:
        return np.clip(
            (x - from_low) / (from_high - from_low) * (to_high - to_low) + to_low,
            to_low,
            to_high,
        )
    return (x - from_low) / (from_high - from_low) * (to_high - to_low) + to_low


def gkern(l=5, sig=1.0) -> np.ndarray:
    """
    Creates gaussian kernel of width `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return kernel / np.sum(kernel)


def spike_detect_kern(l: int = 11) -> np.ndarray:
    kernel = np.ones(l)
    kernel[l // 2 - 1] = 0.5
    kernel[l // 2 + 1] = 0.5
    kernel[: l // 2 - 1] = -2.0 / (l - 3)
    kernel[l // 2 + 2 :] = -2.0 / (l - 3)
    return kernel


def grad1d(array: np.ndarray, deltax: float):
    """
    Calculates the 1D gradient of the given array. `deltax` is the distance between two points in
    the array. It is assumed that the points in the array are evenly spread.

    ## Parameters
    * `array`: `np.ndarray[float]`
    * `deltax`: `float`
    """

    result = np.zeros(len(array))
    result[0] = (array[1] - array[0]) / deltax
    result[-1] - (array[-1] - array[-2]) / deltax

    result[1:-1] = ((np.roll(array, -1) - np.roll(array, 1)) / deltax)[1:-1]
    return result


def vsini_kernel(vsini: float, spacing=2.25855074e-06):
    p_halfwidth = vsini / (spacing * np.log(10) * 3e5)
    kernel = np.arange(float(-p_halfwidth), float(p_halfwidth + 1))

    if len(kernel) <= 3:
        return np.array([1])

    kernel[0] = 0
    kernel[-1] = 0
    kernel[1:-1] = np.sqrt(1 - (kernel[1:-1] / p_halfwidth) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def convolve_ivs(
    flux_spec: np.ndarray, vrot: float, dvelo=5.200538593541637e-06, epsilon=0.6
) -> np.ndarray:
    # -- convert wavelength array into velocity space, this is easier
    #   we also need to make it equidistant!
    flux_ = flux_spec
    vrot = vrot / 299792
    # -- compute the convolution kernel and normalise it
    n = int(2 * vrot / dvelo)
    velo_k = np.arange(n) * dvelo
    velo_k -= velo_k[-1] / 2.0
    y = 1 - (velo_k / vrot) ** 2  # transformation of velocity
    G = (2 * (1 - epsilon) * np.sqrt(y) + np.pi * epsilon / 2.0 * y) / (
        np.pi * vrot * (1 - epsilon / 3.0)
    )  # the kernel
    G /= G.sum()
    # -- convolve the flux with the kernel
    flux_conv = np.convolve(1 - flux_, G, mode="same")
    return 1 - flux_conv


def gaussian(x, x0, a, sigma):
    return a * np.exp(-0.5 * (x - x0) ** 2 / sigma**2)


def lorentzian(x, x0, a, gamma):
    return a * gamma**2 / ((x - x0) ** 2 + gamma**2)


def ccf_fit_pseudoVoight(x, x0, a, b, sigma, gamma):
    f_g = 2 * sigma * np.sqrt(2 * np.log(2))
    f_l = 2 * gamma
    f = (
        f_g**5
        + 2.69269 * f_g**4 * f_l
        + 2.42843 * f_g**3 * f_l**2
        + 4.47163 * f_g**2 * f_l**3
        + 0.07842 * f_g**f_l**4
        + f_l**5
    ) ** 0.2
    eta = 1.36603 * f_l / f - 0.47719 * (f_l / f) ** 2 + 0.11116 * (f_l / f) ** 3

    return b - eta * lorentzian(x, x0, a, gamma) - (1 - eta) * gaussian(x, x0, a, sigma)


def ccf_fit_gaussian(x, x0, a, b, sigma):
    return b - gaussian(x, x0, a, sigma)


def ccf_fit_lorentzian(x, x0, a, b, gamma):
    return b - gaussian(x, x0, a, gamma)
