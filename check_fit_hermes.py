from json import load
from os import environ
from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore
from matplotlib import pyplot as plt
from pasta import (
    ChunkContinuumFitter,
    NoConvolutionDispersion,
    OnDiskCompound,
    OnDiskInterpolator,
    WlGrid,
)

# Get more info on errors on Rust side
environ["RUST_BACKTRACE"] = "1"


# Set up the interpolator object for the model grid
# We combine three rectangular grids

path = "/STER/hermesnet/fine_grid_log"
# The wavelength grid of the models (first wavelength, step, number of pixels) in log10 scale
wl_grid = WlGrid(np.log10(4_000), 1.6833119454e-06, 90470, log=True)
vsini_range = (1, 500)
rv_range = (-150, 150)
InterpolatorType = OnDiskInterpolator

interpolator1 = InterpolatorType(
    path,
    wavelength=wl_grid,
    teff_range=list(np.arange(25_250, 30_250, 250)),
    m_range=list(np.arange(-0.8, 0.9, 0.1)),
    logg_range=list(np.arange(3.3, 5.1, 0.1)),
    vsini_range=vsini_range,
    rv_range=rv_range,
)
interpolator2 = InterpolatorType(
    path,
    wavelength=wl_grid,
    teff_range=list(
        np.concatenate([np.arange(9800, 10000, 100), np.arange(10000, 26_000, 250)])
    ),
    m_range=list(np.arange(-0.8, 0.9, 0.1)),
    logg_range=list(np.arange(3.0, 5.1, 0.1)),
    vsini_range=vsini_range,
    rv_range=rv_range,
)
interpolator3 = InterpolatorType(
    path,
    wavelength=wl_grid,
    teff_range=list(np.arange(6000, 10_100, 100)),
    m_range=list(np.arange(-0.8, 0.9, 0.1)),
    logg_range=list(np.arange(2.5, 5.1, 0.1)),
    vsini_range=vsini_range,
    rv_range=rv_range,
)

# The 'compound' interpolator combines the three grids
# Replace by OnDiskCompound or CachedCompound if necessary
interpolator = OnDiskCompound(interpolator1, interpolator2, interpolator3)

spectra_folder = Path("/path/to/observed_spectra")


def read_spectrum(index: str, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    path = spectra_folder / f"{index}_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits"
    with fits.open(path) as image:
        flux = np.array(image[0].data, dtype=dtype)  # type: ignore
        # Hermes uses a logarithmic wavelength scale
        wl = np.exp(
            np.linspace(
                image[0].header["CRVAL1"],  # type: ignore
                image[0].header["CDELT1"] * (len(flux) - 1) + image[0].header["CRVAL1"],  # type: ignore
                len(flux),
            )
        )
    mask = (wl >= 4007) & (wl <= 5673)
    wl = wl[mask]
    flux = flux[mask]
    return wl, flux


solutions_file = Path("output.json")
with open(solutions_file, "r") as file:
    solutions = load(file)

# Index of the spectrum to plot
index = "00275318"

wl, flux = read_spectrum(index)
solution = [s for s in solutions if s["index"] == index][0]

labels = solution["labels"]
number_of_chunks = 10
polynomial_degree = 8
blending_length = 0.2
fitter = ChunkContinuumFitter(wl, number_of_chunks, polynomial_degree, blending_length)
dispersion = NoConvolutionDispersion(wl)

normalized_model = interpolator.produce_model(dispersion, *labels)
continuum = fitter.build_continuum(solution["continuum"])


fig, ax = plt.subplots()
ax.plot(wl, flux / continuum, c="k", linewidth=0.8)
ax.plot(wl, normalized_model, c="r", linewidth=0.8)
ax.axhline(1, c="blue", linestyle="--", linewidth=0.8)
ax.set_xlabel("Wavelength [A]")
ax.set_ylabel("Normalized flux")
ax.set_title(
    f"{index}, Teff={round(labels[0])} K, [M/H]={labels[1]:.2f} dex,\nlogg={labels[2]:.2f} dex, vsini={labels[3]:.1f} km/s, RV={labels[4]:.1f} km/s"
)
plt.show()
