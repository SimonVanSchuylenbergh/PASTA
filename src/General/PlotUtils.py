import astropy.constants as cc
import numpy as np
from astropy import units as u # type: ignore
from matplotlib.cm import get_cmap
from numpy import exp, sqrt
from scipy.integrate import simps


def rgb_from_T(T, std=False):
    """
    Calculate RGB color of a Planck spectrum of temperature T.
    See rgb.__doc__ for a description of keywords.
    """

    def planck(lam, T):
        """
        Spectral radiance of a blackbody of temperature T.

        Keywords
        -------
        lam     Wavelength in nm
        T       Temperature in K

        Returns
        -------
        Spectral radiance in cgs units (erg/s/sr/cm2)
        """
        lam = lam * u.nm
        x = cc.h * cc.c / lam / cc.k_B / (T * u.K)
        B = 2 * cc.h * cc.c**2 / lam**5 / (exp(x) - 1)
        return B.cgs.value

    lam = np.linspace(350, 800, 100)
    B = planck(lam, T)

    if T < 670:
        RGB = np.array([1, 0, 0])
    elif 670 <= T < 675:
        RGB = rgb(lam, B, std=std)
        RGB[2] = 0
    elif 675 <= T < 1e7:
        RGB = rgb(lam, B, std=std)
    else:
        RGB = np.array([0.63130101, 0.71233531, 1.0])

    return RGB


def rgb(lam, spec, std=False):
    """
    Return RGB color of a spectrum.

    Keywords
    --------
    lam:        Wavelength in nm
    spec:       Radiance, or intensity, or brightness, or luminosity, or
                whatever quantity with units that are sorta kinda
                energy/s/sterad/cm2/wavelength. Normalization doesn't matter.
    ncol:       Normalization constant, e.g. set ncol=255 for color range [0-255]
    """

    def xy(lam, L):
        """
        Return x,y position in CIE 1931 color space chromaticity diagram for an
        arbitrary spectrum.

        Keywords
        -------
        lam:    Wavelength in nm
        L:      Spectral radiance
        """

        def cie():
            """
            Color matching functions. Columns are wavelength in nm, and xbar, ybar,
            and zbar, are the functions for R, G, and B, respectively.
            """
            lxyz = np.array(
                [
                    [380.0, 0.0014, 0.0000, 0.0065],
                    [385.0, 0.0022, 0.0001, 0.0105],
                    [390.0, 0.0042, 0.0001, 0.0201],
                    [395.0, 0.0076, 0.0002, 0.0362],
                    [400.0, 0.0143, 0.0004, 0.0679],
                    [405.0, 0.0232, 0.0006, 0.1102],
                    [410.0, 0.0435, 0.0012, 0.2074],
                    [415.0, 0.0776, 0.0022, 0.3713],
                    [420.0, 0.1344, 0.0040, 0.6456],
                    [425.0, 0.2148, 0.0073, 1.0391],
                    [430.0, 0.2839, 0.0116, 1.3856],
                    [435.0, 0.3285, 0.0168, 1.6230],
                    [440.0, 0.3483, 0.0230, 1.7471],
                    [445.0, 0.3481, 0.0298, 1.7826],
                    [450.0, 0.3362, 0.0380, 1.7721],
                    [455.0, 0.3187, 0.0480, 1.7441],
                    [460.0, 0.2908, 0.0600, 1.6692],
                    [465.0, 0.2511, 0.0739, 1.5281],
                    [470.0, 0.1954, 0.0910, 1.2876],
                    [475.0, 0.1421, 0.1126, 1.0419],
                    [480.0, 0.0956, 0.1390, 0.8130],
                    [485.0, 0.0580, 0.1693, 0.6162],
                    [490.0, 0.0320, 0.2080, 0.4652],
                    [495.0, 0.0147, 0.2586, 0.3533],
                    [500.0, 0.0049, 0.3230, 0.2720],
                    [505.0, 0.0024, 0.4073, 0.2123],
                    [510.0, 0.0093, 0.5030, 0.1582],
                    [515.0, 0.0291, 0.6082, 0.1117],
                    [520.0, 0.0633, 0.7100, 0.0782],
                    [525.0, 0.1096, 0.7932, 0.0573],
                    [530.0, 0.1655, 0.8620, 0.0422],
                    [535.0, 0.2257, 0.9149, 0.0298],
                    [540.0, 0.2904, 0.9540, 0.0203],
                    [545.0, 0.3597, 0.9803, 0.0134],
                    [550.0, 0.4334, 0.9950, 0.0087],
                    [555.0, 0.5121, 1.0000, 0.0057],
                    [560.0, 0.5945, 0.9950, 0.0039],
                    [565.0, 0.6784, 0.9786, 0.0027],
                    [570.0, 0.7621, 0.9520, 0.0021],
                    [575.0, 0.8425, 0.9154, 0.0018],
                    [580.0, 0.9163, 0.8700, 0.0017],
                    [585.0, 0.9786, 0.8163, 0.0014],
                    [590.0, 1.0263, 0.7570, 0.0011],
                    [595.0, 1.0567, 0.6949, 0.0010],
                    [600.0, 1.0622, 0.6310, 0.0008],
                    [605.0, 1.0456, 0.5668, 0.0006],
                    [610.0, 1.0026, 0.5030, 0.0003],
                    [615.0, 0.9384, 0.4412, 0.0002],
                    [620.0, 0.8544, 0.3810, 0.0002],
                    [625.0, 0.7514, 0.3210, 0.0001],
                    [630.0, 0.6424, 0.2650, 0.0000],
                    [635.0, 0.5419, 0.2170, 0.0000],
                    [640.0, 0.4479, 0.1750, 0.0000],
                    [645.0, 0.3608, 0.1382, 0.0000],
                    [650.0, 0.2835, 0.1070, 0.0000],
                    [655.0, 0.2187, 0.0816, 0.0000],
                    [660.0, 0.1649, 0.0610, 0.0000],
                    [665.0, 0.1212, 0.0446, 0.0000],
                    [670.0, 0.0874, 0.0320, 0.0000],
                    [675.0, 0.0636, 0.0232, 0.0000],
                    [680.0, 0.0468, 0.0170, 0.0000],
                    [685.0, 0.0329, 0.0119, 0.0000],
                    [690.0, 0.0227, 0.0082, 0.0000],
                    [695.0, 0.0158, 0.0057, 0.0000],
                    [700.0, 0.0114, 0.0041, 0.0000],
                    [705.0, 0.0081, 0.0029, 0.0000],
                    [710.0, 0.0058, 0.0021, 0.0000],
                    [715.0, 0.0041, 0.0015, 0.0000],
                    [720.0, 0.0029, 0.0010, 0.0000],
                    [725.0, 0.0020, 0.0007, 0.0000],
                    [730.0, 0.0014, 0.0005, 0.0000],
                    [735.0, 0.0010, 0.0004, 0.0000],
                    [740.0, 0.0007, 0.0002, 0.0000],
                    [745.0, 0.0005, 0.0002, 0.0000],
                    [750.0, 0.0003, 0.0001, 0.0000],
                    [755.0, 0.0002, 0.0001, 0.0000],
                    [760.0, 0.0002, 0.0001, 0.0000],
                    [765.0, 0.0001, 0.0000, 0.0000],
                    [770.0, 0.0001, 0.0000, 0.0000],
                    [775.0, 0.0001, 0.0000, 0.0000],
                    [780.0, 0.0000, 0.0000, 0.0000],
                ]
            )
            return lxyz.T

        lamcie, xbar, ybar, zbar = cie()  # Color matching functions
        L = np.interp(lamcie, lam, L)  # Interpolate to same axis

        # Tristimulus values
        X = simps(L * xbar, lamcie)
        Y = simps(L * ybar, lamcie)
        Z = simps(L * zbar, lamcie)
        XYZ = np.array([X, Y, Z])
        x = X / sum(XYZ)
        y = Y / sum(XYZ)
        z = Z / sum(XYZ)
        return x, y

    def adjust_gamma(RGB):
        """
        Adjust gamma value of RGB color
        """
        a = 0.055
        for i, color in enumerate(RGB):
            if color <= 0.0031308:
                RGB[i] = 12.92 * color
            else:
                RGB[i] = (1 + a) * color ** (1 / 2.4) - a
        return RGB

    x, y = xy(lam, spec)
    z = 1 - x - y
    Y = 1.0
    X = (Y / y) * x
    Z = (Y / y) * z
    XYZ = np.array([X, Y, Z])

    # Matrix for Wide RGB D65 conversion
    if std:
        XYZ2RGB = np.array(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ]
        )
    else:
        XYZ2RGB = np.array(
            [
                [1.656492, -0.354851, -0.255038],
                [-0.707196, 1.655397, 0.036152],
                [0.051713, -0.121364, 1.011530],
            ]
        )

    RGB = np.dot(XYZ2RGB, XYZ)  # Map XYZ to RGB
    RGB = adjust_gamma(RGB)  # Adjust gamma
    # RGB = RGB / np.array([0.9505, 1., 1.0890])  #Scale so that Y of "white" (D65) is (0.9505, 1.0000, 1.0890)
    maxRGB = max(RGB.flatten())
    if maxRGB > 1:
        RGB = RGB / maxRGB  # Normalize to 1 if there are values above
    RGB = RGB.clip(min=0)  # Clip negative values

    return RGB


def blackbody_colors(
    n_colors: int,
    T_lower=1000.0,
    T_upper=9000.0,
    log_space=True,
    norm=0.4,
    brightness=1.0,
):
    """
    log_space: evenly space temperature in log space instead of linear space, tends to give
    visually more evenly spaced colors for high temperatures

    norm: between 0 and 1, 0 is original colors, 1 is normalized colors, higher values improve
    visibility for temperatures close to 6500 K on white backgrounds

    brightness: multiply each color by this value to increase or decrease the brightness
    """
    if log_space:
        T = np.logspace(np.log10(T_lower), np.log10(T_upper), n_colors)
    else:
        T = np.linspace(T_lower, T_upper, n_colors)
    return [
        brightness
        * (
            (1 - norm) * rgb_from_T(t)
            + norm * rgb_from_T(t) / sqrt(np.sum(np.square(rgb_from_T(t))))
        )
        for t in T
    ]


def colormap_colors(cmap: str, n_colors: int, start=0.0, end=1.0):
    values = np.linspace(start, end, n_colors)
    return [get_cmap(cmap)(v) for v in values]


def color_value_sat(color: tuple[float], value=1.0, sat=1.0):
    if len(color) == 3:
        mean_value = (color[0] + color[1] + color[2]) * 0.33333
        return (
            np.clip((color[0] * sat + (1 - sat) * mean_value) * value, 0.0, 1.0),
            np.clip((color[1] * sat + (1 - sat) * mean_value) * value, 0.0, 1.0),
            np.clip((color[2] * sat + (1 - sat) * mean_value) * value, 0.0, 1.0),
        )
    elif len(color) == 4:
        mean_value = (color[0] + color[1] + color[2]) * 0.33333
        return (
            np.clip((color[0] * sat + (1 - sat) * mean_value) * value, 0.0, 1.0),
            np.clip((color[1] * sat + (1 - sat) * mean_value) * value, 0.0, 1.0),
            np.clip((color[2] * sat + (1 - sat) * mean_value) * value, 0.0, 1.0),
            color[3],
        )
    else:
        raise ValueError("color must be a tuple of length 3 or 4!")
