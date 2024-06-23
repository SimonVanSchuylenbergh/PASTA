from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore


def read_spectrum(
    path: Path,
    wl_start: float | None = None,
    wl_end: float | None = None,
    rescale=False,
):
    """
    Returns (wl, flux, var)
    """

    with fits.open(path) as image:
        flux = image[0].data  # type: ignore

        wl = np.exp(
            np.linspace(
                image[0].header["CRVAL1"],  # type: ignore
                image[0].header["CDELT1"] * (len(flux) - 1) + image[0].header["CRVAL1"],  # type: ignore
                len(flux),
            )
        )
    with fits.open(str(path).replace("merged", "mergedVar")) as image:
        var = image[0].data  # type: ignore

    if wl_start:
        flux = flux[wl > wl_start]
        var = var[wl > wl_start]
        wl = wl[wl > wl_start]
    if wl_end:
        flux = flux[wl < wl_end]
        var = var[wl < wl_end]
        wl = wl[wl < wl_end]
    if rescale:
        var /= np.median(np.nan_to_num(flux))
        flux /= np.median(np.nan_to_num(flux))
    return (wl, flux, var)
