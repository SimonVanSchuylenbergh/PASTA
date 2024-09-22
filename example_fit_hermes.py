# This is an example of fitting a high resolution spectrum with particle swarm optimization.
# The models are assumed to be already convolved to the resolution of the instrument.
# Example for the HERMES spectrograph.

from json import dump, load
from os import environ
from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore
from pasta import (
    ChunkContinuumFitter,
    InMemCompound,
    InMemInterpolator,
    NoConvolutionDispersion,
    OnDiskCompound,
    OnDiskInterpolator,
    PSOSettings,
    WlGrid,
)
from tqdm.auto import tqdm  # Optional progress bar

# Get more info on errors on Rust side
environ["RUST_BACKTRACE"] = "1"


# Read the observed spectrum from a fits file
# Return the wavelength, flux, and variance arrays
def read_and_prepare_spectrum(
    path: Path, dtype=np.float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # Read the variance from the corresponding file
    with fits.open(str(path).replace("merged", "mergedVar")) as image:
        var = np.array(image[0].data, dtype=dtype)  # type: ignore

    # Only use 4007-5673 Angstroms
    mask = (wl >= 4007) & (wl <= 5673)
    wl = wl[mask]
    flux = flux[mask]
    var = var[mask]
    # Ignore pixels with zero or nan variance
    mask = (np.isnan(var)) | (var <= 0) | (np.isinf(var))
    var[mask] = 1e30  # Set a large value to ignore these pixels
    return (wl, flux, var)


# Set up the interpolator object for the model grid
# We combine three rectangular grids
# InMemInterpolator can be replaced by OnDiskInterpolator or CachedInterpolator
# if the models are too large to fit in memory
path = "path/to/models"
# The wavelength grid of the models (first wavelength, step, number of pixels) in log10 scale
wl_grid = WlGrid(np.log10(4_000), 1.6833119454e-06, 90470, log=True)
vsini_range = (1, 500)
rv_range = (-150, 150)
InterpolatorType = InMemInterpolator

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
interpolator = InMemCompound(interpolator1, interpolator2, interpolator3)


# Output is written in json format
output_file = Path("output.json")
# Settings to the PSO algorithm
settings = PSOSettings(
    num_particles=44,  # Recommended to use a multiple of the number of cores on the system
    max_iters=100,
    social_factor=0.5,
)
# Parameters to continuum fitting
number_of_chunks = 10
polynomial_degree = 8
blending_length = 0.2

# In case the output file already exists, overwrite it,
# or append to it and skip the already computed solutions
overwrite = False
if not overwrite and output_file.exists():
    with open(output_file) as f:
        solutions = load(f)
else:
    solutions = []

spectra_folder = Path("path/to/observed_spectra_folder")
# For HERMES, don't include the Var files here
flux_files = sorted(
    [f for f in list(spectra_folder.glob("*.fits")) if "Var" not in f.name]
)

with open(
    "prog.txt", "w"
) as prog_file:  # Optional progress bar written to file prog.txt (useful for slurm)
    for file in tqdm(flux_files, file=prog_file):
        index = file.stem.split("_")[0]
        if index in [sol["index"] for sol in solutions]:
            continue  # Skip if the solution is already computed

        wl, flux, var = read_and_prepare_spectrum(file)
        # The object for fitting the continuum in chunks
        fitter = ChunkContinuumFitter(
            wl, number_of_chunks, polynomial_degree, blending_length
        )
        # The dispersion object, in this case no convolution for resolution is needed
        # because the models are already convolved
        dispersion = NoConvolutionDispersion(wl)
        # Fit the spectrum and write the solution to the output file
        solution = interpolator.fit_pso(fitter, dispersion, flux, var, settings)
        sol = {
            "index": index,
            "labels": solution.labels,  # (teff, logg, m, vsini, rv)
            "continuum": solution.continuum_params,  # Polynomial coefficients for every chunk
            "chi2": solution.chi2,  # Chi-squared of the best solution
        }
        solutions.append(sol)
        with open(output_file, "w") as f:
            dump(solutions, f)
