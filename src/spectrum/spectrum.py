from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Generator, Optional, overload

import numpy as np
from numba import float64, njit
from numba.types import Array  # type: ignore
from rust_nmf import rv_shift_bulk  # type: ignore
from spectres import spectres
from tqdm.auto import tqdm


@njit(
    Array(float64, 1, "C")(Array(float64, 1, "C"), Array(float64, 1, "C")),
    cache=True,
)
def convolve(input_array, kernel):
    # Get the length of the input array and the kernel
    input_len = len(input_array)
    kernel_len = len(kernel)

    # The output length will be input_len + kernel_len - 1
    output_len = input_len + kernel_len - 1

    # Initialize the output array with zeros
    output_array = np.zeros(output_len)

    for i in range(input_len):
        for j in range(kernel_len):
            output_array[i + j] += input_array[i] * kernel[j]

    start = kernel_len // 2
    end = start + input_len
    return output_array[start:end]


def build_vsini_kernel(vsini: float, dvelo=5.200538593541637e-06) -> np.ndarray:
    epsilon = 0.6
    vrot = vsini / 299792

    n = int(2 * vrot / dvelo)
    velo_k = np.arange(n) * dvelo
    velo_k -= velo_k[-1] / 2.0

    y = 1 - (velo_k / vrot) ** 2
    G = (2 * (1 - epsilon) * np.sqrt(y) + np.pi * epsilon / 2.0 * y) / (
        np.pi * vrot * (1 - epsilon / 3.0)
    )
    G /= G.sum()
    return G


@dataclass
class Spectrum:
    """Spectrum without variance"""

    wl: np.ndarray
    flux: np.ndarray

    def plot(self, ax, **kwargs):
        ax.plot(self.wl, self.flux, **kwargs)

    def chi2(self, other: Spectrum) -> float:
        """Chi squared between two spectra."""
        n = len(self.flux) - len(other.flux)
        if n < 0:
            raise ValueError("Spectra must have the same length")
        elif n == 0:
            s = self.flux
        else:
            s = self.flux[n // 2 : -n // 2]
        return float(np.mean((s - other.flux) ** 2))

    def add_noise(self, snr: float) -> Spectrum:
        """Add noise of constant SNR"""
        return Spectrum(
            self.wl,
            np.random.normal(self.flux, self.flux / snr),
        )

    def convolve(self, vsini: float) -> Spectrum:
        """
        Convolve the spectrum with a rotational broadening kernel
        """
        n = int(vsini / 1.5590798660350345)
        kernel = build_vsini_kernel(vsini)
        flux = convolve(self.flux, kernel)
        return Spectrum(self.wl[n:-n], flux[n:-n])

    def resample_to(self, wl: np.ndarray) -> Spectrum:
        """
        Resample spectrum to a given wavelength array
        """
        flux = spectres(wl, self.wl, self.flux)
        if not isinstance(flux, np.ndarray):
            raise TypeError
        return Spectrum(wl, flux)

    def cut(self, wl_min: float, wl_max: float) -> Spectrum:
        """
        Cut the spectrum to a given wavelength range
        """
        mask = (self.wl > wl_min) & (self.wl < wl_max)
        return Spectrum(
            self.wl[mask],
            self.flux[mask],
        )

    def shift(self, RV: float) -> Spectrum:
        """
        Shift the spectrum by a given radial velocity.
        Positive value will shift the spectrum to the red
        """
        new_wl = self.wl * (1 + RV / 299792.458)
        return Spectrum(new_wl, self.flux)

    def rescale(self, factor: float) -> Spectrum:
        """
        Rescale the flux by a given factor
        """
        return Spectrum(self.wl, self.flux * factor)

    def clip(self, _min: float, _max: float) -> Spectrum:
        """Clip the flux values."""
        return Spectrum(self.wl, np.clip(self.flux, _min, _max))

    def __add__(self, other: Spectrum) -> Spectrum:
        return Spectrum(self.wl, self.flux + other.flux)

    def __sub__(self, other: Spectrum) -> Spectrum:
        return Spectrum(self.wl, self.flux - other.flux)

    def __mul__(self, other: Spectrum) -> Spectrum:
        return Spectrum(self.wl, self.flux * other.flux)

    def __truediv__(self, other: Spectrum) -> Spectrum:
        return Spectrum(self.wl, self.flux / other.flux)

    def __pow__(self, other: Spectrum) -> Spectrum:
        return Spectrum(self.wl, self.flux**other.flux)

    def __getitem__(self, i: slice | Sequence | np.ndarray) -> Spectrum:
        return Spectrum(self.wl[i], self.flux[i])


@dataclass
class ObservedSpectrum:
    """Spectrum with variance information"""

    wl: np.ndarray
    flux: np.ndarray
    var: np.ndarray

    def plot(self, ax, **kwargs):
        ax.plot(self.wl, self.flux, **kwargs)

    def chi2(self, other: Spectrum) -> float:
        """Chi squared between two spectra.
        The other spectrum must not have variance information."""
        n = len(self.flux) - len(other.flux)
        if n < 0:
            raise ValueError("Spectra must have the same length")
        elif n == 0:
            s = self.flux
        else:
            s = self.flux[n // 2 : -n // 2]
        return float(np.mean(((s - other.flux) / s.var) ** 2))

    def resample_to(self, wl: np.ndarray) -> ObservedSpectrum:
        """
        Resample spectrum to a given grid
        """
        flux = spectres(wl, self.wl, self.flux)
        var = spectres(wl, self.wl, self.var)
        if not isinstance(flux, np.ndarray):
            raise TypeError
        if not isinstance(var, np.ndarray):
            raise TypeError
        return ObservedSpectrum(wl, flux, var)

    def as_spectrum(self) -> Spectrum:
        return Spectrum(self.wl, self.flux)

    def cut(self, wl_min: float, wl_max: float) -> ObservedSpectrum:
        """
        Cut the spectrum to a given wavelength range
        """
        mask = (self.wl > wl_min) & (self.wl < wl_max)
        return ObservedSpectrum(self.wl[mask], self.flux[mask], self.var[mask])

    def shift(self, RV: float) -> ObservedSpectrum:
        """
        Shift the spectrum by a given radial velocity. Positive value will shift the spectrum to the red
        """
        new_wl = self.wl * (1 + RV / 299792.458)
        return ObservedSpectrum(new_wl, self.flux, self.var)

    def rescale(self, factor: float) -> ObservedSpectrum:
        """
        Rescale the flux by a given factor
        """
        return ObservedSpectrum(self.wl, self.flux * factor, self.var * factor**2)

    def rescale_median(self) -> ObservedSpectrum:
        """
        Rescale the flux by the median value
        """
        median = np.median(self.flux)
        return ObservedSpectrum(self.wl, self.flux / median, self.var / median**2)

    def invert_from_max(self) -> ObservedSpectrum:
        """invert spectrum by subtracting from the maximum value"""
        _max = np.max(self.flux)
        flux = _max - self.flux
        return ObservedSpectrum(self.wl, flux, self.var)

    def invert_from_unity(self) -> ObservedSpectrum:
        """invert spectrum by subtracting from unity. Negative values are clipped to zero"""
        flux = np.clip(1 - self.flux, 0, 1)
        return ObservedSpectrum(self.wl, flux, self.var)

    def clip(self, _min: Optional[float], _max: Optional[float]) -> ObservedSpectrum:
        """Clip the flux values."""
        return ObservedSpectrum(self.wl, np.clip(self.flux, _min, _max), self.var)

    def bootstrap(self) -> ObservedSpectrum:
        """Generate a new spectrum with noise scaled by the variance array."""
        new_flux = np.random.normal(self.flux, np.sqrt(self.var), self.flux.shape)
        return ObservedSpectrum(self.wl, new_flux, self.var)

    def __getitem__(self, i: slice | Sequence | np.ndarray) -> ObservedSpectrum:
        return ObservedSpectrum(self.wl[i], self.flux[i], self.var[i])


class Resampler:
    """Helper class for resampling with multithreading"""

    def __init__(self, wl: np.ndarray):
        self.wl = wl

    def __call__(self, spectrum: Spectrum) -> np.ndarray:
        return spectrum.resample_to(self.wl).flux


class ResamplerObserved:
    """Helper class for resampling with multithreading"""

    def __init__(self, wl: np.ndarray):
        self.wl = wl

    def __call__(self, spectrum: ObservedSpectrum) -> tuple[np.ndarray, np.ndarray]:
        resampled = spectrum.resample_to(self.wl)
        return resampled.flux, resampled.var


@dataclass
class Spectra:
    """List of spectra with the same wavelength grid."""

    wl: np.ndarray
    array: np.ndarray

    @classmethod
    def from_list(cls, spectra: list[Spectrum]) -> Spectra:
        wl = spectra[0].wl
        arr = np.array([s.flux for s in spectra])
        return cls(wl, arr)

    def add_noise(self, snr: float) -> ResampledObservedSpectra:
        """Add noise to all spectra with constant SNR."""
        return ResampledObservedSpectra(
            self.wl, np.random.normal(self.array, self.array / snr), self.array / snr**2
        )

    def clip(self, _min: float | None, _max: float | None) -> Spectra:
        """Clip the flux values."""
        return Spectra(self.wl, np.clip(self.array, _min, _max))

    def resample_to(self, wl: np.ndarray, n_cores=1) -> Spectra:
        """Resample all spectra to a given wavelength grid."""
        resampler = Resampler(wl)
        if n_cores == 1:
            out = [resampler(spec) for spec in self]
        else:
            with Pool(n_cores) as p:
                out = list(tqdm(p.imap(resampler, self), total=len(self)))
        return Spectra(
            wl,
            np.array(out),
        )

    def cut(self, wl_min: float, wl_max: float) -> Spectra:
        """Cut all spectra to a given wavelength range."""
        mask = (self.wl > wl_min) & (self.wl < wl_max)
        return Spectra(self.wl[mask], self.array[:, mask])

    def mean(self) -> np.ndarray:
        """Mean of all spectra."""
        return np.mean(self.array, axis=0)

    def rescale(self, spec_array: np.ndarray) -> Spectra:
        """Rescale all spectra by a given array."""
        return Spectra(self.wl, self.array * spec_array)

    def rescale_and_shift(self, spec_array: np.ndarray) -> Spectra:
        """Rescale all spectra by a given array and shift by unity."""
        return Spectra(self.wl, self.array / spec_array - 1)

    def rescale_median(self) -> Spectra:
        """Rescale all spectra by their median value."""
        medians = np.median(self.array, axis=1)[:, None]
        return Spectra(self.wl, self.array / medians)

    def invert_from_max(self) -> Spectra:
        """Invert all spectra by subtracting from the maximum value."""
        wl = self.wl
        arr = np.clip(1 - self.array, 0, 1)
        return Spectra(wl, arr)

    def invert_from_unity(self, clip=True) -> Spectra:
        """Invert all spectra by subtracting from unity."""
        wl = self.wl
        if clip:
            arr = np.clip(1 - self.array, 0, 1)
        else:
            arr = 1 - self.array
        return Spectra(wl, arr)

    def chi2(self, other: Spectra) -> float:
        """Chi squared between two sets of spectra."""
        return float(np.mean((self.array - other.array) ** 2))

    def chi2_with_stddev(self, other: Spectra, rescaling: np.ndarray) -> float:
        """Chi squared between two sets of spectra with rescaling."""
        return float(np.mean(((self.array - other.array) / rescaling) ** 2))

    @overload
    def __getitem__(self, i: slice | Sequence | np.ndarray) -> Spectra: ...

    @overload
    def __getitem__(self, i: int) -> Spectrum: ...

    def __getitem__(self, i):
        if isinstance(i, (slice, np.ndarray)):
            return Spectra(self.wl, self.array[i])
        elif isinstance(i, int):
            return Spectrum(self.wl, self.array[i])
        else:
            raise TypeError(f"Invalid index type {type(i)}")

    def __iter__(self) -> Generator[Spectrum, None, None]:
        return (Spectrum(self.wl, spectrum) for spectrum in self.array)

    def __len__(self) -> int:
        return len(self.array)


@dataclass
class ObservedSpectra:
    """
    List of observed spectra with variance information.
    Not necessarily on same wavelength grid.
    """

    spectra: list[ObservedSpectrum]

    def apply_RV_and_resample(
        self,
        wl: np.ndarray,
        rvs: list[float] | np.ndarray,
        rescale_factor: float = 1,
        include_list: Optional[list[bool]] = None,
    ) -> ResampledObservedSpectra:
        """Shift every spectrum by a given RV and resample to a common wavelength grid."""
        if not include_list:
            include_list = [True] * len(self.spectra)

        sp_rvs = [
            (spectrum, rv)
            for spectrum, rv, include in zip(self.spectra, rvs, include_list)
            if include
        ]
        original_wls = [s.wl for s, _ in sp_rvs]
        fluxes = [s.flux for s, _ in sp_rvs]
        variances = [s.var for s, _ in sp_rvs]
        rvs = [rv for _, rv in sp_rvs]
        resampled_fluxes, resampled_variances = rv_shift_bulk(
            original_wls, fluxes, variances, wl, rvs
        )
        return ResampledObservedSpectra(
            wl,
            resampled_fluxes * rescale_factor,
            resampled_variances * rescale_factor**2,
        )

    def cut(self, wl_min: float, wl_max: float) -> ObservedSpectra:
        """Cut all spectra to a given wavelength range."""
        return ObservedSpectra(
            [spectrum.cut(wl_min, wl_max) for spectrum in self.spectra]
        )

    def rescale_median(self) -> ObservedSpectra:
        """Rescale all spectra by their median value."""
        return ObservedSpectra([spectrum.rescale_median() for spectrum in self.spectra])

    def invert_from_max(self) -> ObservedSpectra:
        """Invert all spectra by subtracting from the maximum value."""
        return ObservedSpectra(
            [spectrum.invert_from_max() for spectrum in self.spectra]
        )

    def invert_from_unity(self) -> ObservedSpectra:
        """Invert all spectra by subtracting from unity."""
        return ObservedSpectra(
            [spectrum.invert_from_unity() for spectrum in self.spectra]
        )

    def clip(self, _min: Optional[float], _max: Optional[float]) -> ObservedSpectra:
        """Clip the flux values."""
        return ObservedSpectra([spectrum.clip(_min, _max) for spectrum in self.spectra])

    def as_resampled(self) -> ResampledObservedSpectra:
        """
        If all spectra are already resampled to the same grid,
        convert to ResampledObservedSpectra object.
        """
        wl = self.spectra[0].wl
        N = len(wl)
        for spectrum in self.spectra:
            if len(spectrum.wl) != N:
                raise ValueError("All spectra must have the same length")
        fluxes = np.array([spectrum.flux for spectrum in self.spectra])
        variances = np.array([spectrum.var for spectrum in self.spectra])
        return ResampledObservedSpectra(wl, fluxes, variances)

    @property
    def wls(self) -> list[np.ndarray]:
        """List of wavelength grids."""
        return [spectrum.wl for spectrum in self.spectra]

    @property
    def fluxs(self):
        """List of flux arrays."""
        return [spectrum.flux for spectrum in self.spectra]

    @overload
    def __getitem__(self, i: slice) -> ObservedSpectra: ...

    @overload
    def __getitem__(self, i: int) -> ObservedSpectrum: ...

    def __getitem__(self, i):
        if isinstance(i, int):
            return ObservedSpectrum(
                self.spectra[i].wl, self.spectra[i].flux, self.spectra[i].var
            )
        return ObservedSpectra(self.spectra[i])

    def __iter__(self) -> Generator[ObservedSpectrum, None, None]:
        return (spectrum for spectrum in self.spectra)

    def __len__(self) -> int:
        return len(self.spectra)


@dataclass
class ResampledObservedSpectra:
    """List of resampled observed spectra with variance information on the same wavelength grid."""

    wl: np.ndarray
    flux: np.ndarray
    var: np.ndarray

    def clip(self, _min: float | None, _max: float | None) -> ResampledObservedSpectra:
        """Clip the flux values."""
        return ResampledObservedSpectra(
            self.wl, np.clip(self.flux, _min, _max), self.var
        )

    def cut(self, wl_min: float, wl_max: float) -> ResampledObservedSpectra:
        """Cut all spectra to a given wavelength range."""
        mask = (self.wl > wl_min) & (self.wl < wl_max)
        return ResampledObservedSpectra(
            self.wl[mask], self.flux[:, mask], self.var[:, mask]
        )

    def rescale_median(self) -> ResampledObservedSpectra:
        """Rescale all spectra by their median value."""
        medians = np.median(self.flux, axis=1)[:, None]
        return ResampledObservedSpectra(
            self.wl, self.flux / medians, self.var / medians**2
        )

    def mean(self) -> np.ndarray:
        """Mean of all spectra."""
        return np.mean(self.flux, axis=0)

    def rescale(self, spec_array: np.ndarray) -> ResampledObservedSpectra:
        """Rescale all spectra by a given array."""
        return ResampledObservedSpectra(
            self.wl, self.flux * spec_array, self.var * spec_array**2
        )

    def rescale_and_shift(self, spec_array: np.ndarray) -> ResampledObservedSpectra:
        """Rescale all spectra by a given array and shift by unity."""
        return ResampledObservedSpectra(
            self.wl, self.flux / spec_array - 1, self.var / spec_array**2
        )

    def invert_from_unity(self) -> ResampledObservedSpectra:
        """Invert all spectra by subtracting from unity."""
        flux = np.clip(1 - self.flux, 0, 1)
        return ResampledObservedSpectra(self.wl, flux, self.var)

    def invert_from_max(self) -> ResampledObservedSpectra:
        _max = np.max(self.flux, axis=1)
        flux = _max[:, None] - self.flux
        return ResampledObservedSpectra(self.wl, flux, self.var)            

    def bootstrap(self, N: int) -> list[ResampledObservedSpectra]:
        """Generate N new spectra for every spectrum with noise scaled by the variance array."""
        return [
            ResampledObservedSpectra(
                self.wl,
                np.random.normal(self.flux, np.sqrt(self.var), self.flux.shape),
                self.var,
            )
            for _ in range(N)
        ]

    def bootstrap_single(self, rng) -> ResampledObservedSpectra:
        """Generate one new spectrum for every spectrum with noise scaled by the variance array."""
        return ResampledObservedSpectra(
            self.wl,
            rng.normal(self.flux, np.nan_to_num(np.sqrt(self.var)), self.flux.shape),
            self.var,
        )

    def to_spectra(self) -> Spectra:
        """Remove variance information."""
        return Spectra(self.wl, self.flux)

    def chi2(self, spectra: Spectra):
        return np.nanmean((self.flux - spectra.array) ** 2 / self.var)

    def relative_residuals(self, spectra: Spectra):
        """Relative residuals between two sets of spectra."""
        return (self.flux - spectra.array) / np.sqrt(self.var)

    @overload
    def __getitem__(
        self, i: slice | Sequence | np.ndarray
    ) -> ResampledObservedSpectra: ...

    @overload
    def __getitem__(self, i: int) -> ObservedSpectrum: ...

    def __getitem__(self, i):
        if isinstance(i, (slice, np.ndarray)):
            return ResampledObservedSpectra(self.wl, self.flux[i], self.var[i])
        elif isinstance(i, int):
            return ObservedSpectrum(self.wl, self.flux[i], self.var[i])
        else:
            raise TypeError(f"Invalid index type {type(i)}")

    def __iter__(self) -> Generator[ObservedSpectrum, None, None]:
        return (ObservedSpectrum(self.wl, f, v) for f, v in zip(self.flux, self.var))

    def __len__(self) -> int:
        return len(self.flux)
