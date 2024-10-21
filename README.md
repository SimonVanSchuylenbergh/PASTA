# PASTA (Pipeline for Automated Spectral Analysis)
 
1. [Introduction](#introduction)
2. [Installation](#installation)

## Introduction
PASTA is a fully automatic pipeline for estimating stellar labels from unnormalized spectra of single stars by comparison to model spectra.
In its current form, five labels are derived:
- Effective temperature $T_\text{eff}$
- Metallicity [M/H]
- Surface gravity $\log g$
- Rotational velocity $v \sin i$
- Radial velocity RV

The code does not have the capability of generating model spectra, but relies on a pre-computed grid of models from which spectra are produced through interpolation.
Hence the underlying physics of the pipeline is determined by the model grid that the user chooses to use.

The basic principle behind the pipeline is to compute the $\chi^2$ function between the observed spectrum and a set of model spectra, and find the set of labels that minimizes this function.
An important feature is that the spectra do not need to be normalized beforehand.
The pipeline handles normalization by fitting a function to represent the pseudo continuum for every model spectrum that is evaluated.
If $F_\text{obs}$ is the observed flux and $F_\text{model}$ that of the model for which the $\chi^2$ function is evaluated, the function representing the pseudo continuum is fitted to $\frac{F_\text{obs}}{F_\text{model}}$.
It is important that this function is flexible enough to fit the true pseudo continuum well, but not so flexible that it starts fitting the defects of the model spectrum, hence obscuring the $\chi^2$ landscape. 
The required ingredients for this method are a fast way to produce a model spectrum at any set of labels, a parametrized function to fit the pseudo continuum, and a minimization algorithm.

To produce a model spectrum, the code performs the following steps:
1. 3D cubic interpolation in a grid of models with different $T_\text{eff}$, [M/H] and $\log g$ to produce a model at arbitrary values of those three labels.
2. Convolve with a rotational broadening kernel depending on the desired value of $v \sin i$
3. Convolve with a gaussian kernel to match the spectral resolution of the instrument. This resolution may be constant or wavelength dependent. Alternatively the model grid can ve pre-convolved to improve speed.
4. Shift the spectrum along wavelength to the desired RV value.
5. Resample to the wavelength sampling of the observed spectrum


For the pseudo continuum function, we developed a method that splits the continuum into overlapping chunks, fits each with a polynomial, and merges them together to one smooth function.
The code allows the user to tweak the number of chunks, degree of the polynomials and width of the overlap regions.
Alternatively, it is possible to use a linear model by providing a design matrix.
The normalization can also easily be turned off for use on pre-normalized spectra.
Additionally, due to the code's modularity it is relatively easy to add a custom normalization function.

As the minimization algorithm the code uses particle swarm optimization (PSO).
Specifically an adapted version of the implementation in [argmin-rs](https://github.com/argmin-rs/argmin).
The metaparameters of the algorithm (e.g. number of particles, social attraction factor, etc) can be chosen by the user.

Example usage in `example_fit_hermes.py` and `check_fit_hermes.py`.

## Installation
The code is written in Rust, with an interface to Python. It can be installed as a Python package, either from the pre-built wheel or by compiling the source code.


### Pre-built wheel
A pre-built wheel for x86_64 Linux is provided under the Github [releases](https://github.com/SimonVanSchuylenbergh/PASTA/releases). It can be installed in the active Python environment with `pip`:
```sh
pip install pasta-0.1.0-cp37-abi3-manylinux_2_35_x86_64.whl
```

### Compiling from source
The Python package is built from source using [maturin](https://github.com/PyO3/maturin). To install in the active Python environment use
```sh
maturin develop --release
```

## The grid
The code expects a 3D grid of model spectra for varying $T_\text{eff}$, [M/H] and $\log g$.
The model spectra may be normalized or include the continuum flux.
In the latter case, the the normalization function can no longer be interpreted as the pseudo continuum, but as the response function of the instrument, i.e. the ratio of the observed flux and the model flux.
The files must be in Numpy's binary `.npy` format in uint16 and contain only one column.
The flux values need to be converted to 16-bit integers for performance reasons.
For normalized models ($F\le1$) this is done simply by multiplying by $2^{16}-1$ and using `np.uint16`.
For models including continuum flux, they will first need to be divided by their maximum value.


The files must be named as
```
l{m/p}{[M/H]*100:04d}_{Teff:05d}_{logg*100:04d}.npy
```
e.g. for $T_\text{eff}=7100$ K, [M/H] $=-0.3$ dex, $\log g = 3.9$ dex this is
```
lm0030_07100_0390.npy
```

The grid does not need to be rectangular, but may have a 'staircase' shape in $T_\text{eff}$ vs $\log g$.
That is, not all $\log g$ values need to be available for every available $T_\text{eff}$.
The code will automatically detect the available range in $\log g$ for every different $T_\text{eff}$ in the grid.
The full range in [M/H] must exist for every $T_\text{eff}$, $\log g$ combination however.

## Python API docs
All the functionality of the code can be accessed from Python through a few exposed classes and functions.

### Interpolator
This is the central object that exposes methods for generating synthetic spectra, fitting observed spectra and calculating error margins.

