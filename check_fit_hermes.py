from json import load
from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore
from matplotlib import pyplot as plt
from pasta import (  # type: ignore
    CachedInterpolator,
    ChunkContinuumFitter,
    FixedResolutionDispersion,
    InMemInterpolator,
    NoConvolutionDispersion,
    PSOSettings,
    WlGrid,
)

# Get more info on errors on Rust side

wl_rng = (4007, 5673)


# Read the observed spectrum from a fits file
# Return the wavelength, and flux arrays
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
    print(flux.shape, wl.shape)
    print(wl[0], wl[-1])
    mask = (wl >= wl_rng[0]) & (wl <= wl_rng[1])
    wl = wl[mask]
    flux = flux[mask]
    return wl, flux


# Set up the interpolator object for the model grid
# We combine three rectangular grids

# Set up the interpolator object for the model grid
# InMemInterpolator can be replaced by OnDiskInterpolator or CachedInterpolator
# if the models are too large to fit in memory

# We use the pre-convolved model grid
# model_path = "/scratch/ragnarv/hermes_norm_convolved_u16"
model_path = "/STER/hermesnet/hermes_norm_convolved_u16"
# The wavelength grid of the models (first wavelength, step, number of pixels) in log10 scale
wl_grid = WlGrid(np.log10(4000), 2e-6, 76145, log=True)

# Alternatively use the unconvoled models:
# model_path = "/scratch/ragnarv/unconvolved_norm_u16"
# wl_grid = WlGrid(np.log10(3500), 1.5e-6, 316699, log=True)

interpolator = CachedInterpolator(
    str(model_path),
    False,  # We are using normalized models that don't include the max value.
    wavelength=wl_grid,
    n_shards=24,  # Applies to CachedInterpolator only
    lrucap=50_000,  # Applies to CachedInterpolator only
)

spectra_folder = Path("/STER/hermesnet/observed")
solutions_file = Path("output.json")

with open(solutions_file, "r") as file:
    solutions = load(file)

# Index of the spectrum to plot
index = "00272174"

wl, flux = read_spectrum(index)
solution = [s for s in solutions if s["index"] == index][0]

label = solution["label"]
number_of_chunks = 5
polynomial_degree = 8
blending_length = 0.2
fitter = ChunkContinuumFitter(wl, number_of_chunks, polynomial_degree, blending_length)
dispersion = NoConvolutionDispersion(wl)

normalized_model = interpolator.produce_model(
    dispersion,
    label["teff"],
    label["m"],
    label["logg"],
    label["vsini"],
    label["rv"],
)
continuum = fitter.build_continuum(solution["continuum_params"])


fig, ax = plt.subplots()
ax.plot(wl, flux / continuum, c="k", linewidth=0.8)
ax.plot(wl, normalized_model, c="r", linewidth=0.8)
ax.axhline(1, c="blue", linestyle="--", linewidth=0.8)
ax.set_xlabel("Wavelength [A]")
ax.set_ylabel("Normalized flux")
ax.set_title(
    f"{index}, Teff={round(label['teff'])} K, [M/H]={label['m']:.2f} dex,\nlogg={label['logg']:.2f} dex, vsini={label['vsini']:.1f} km/s, RV={label['rv']:.1f} km/s"
)
plt.show()
fig.savefig("fit_example.png")
