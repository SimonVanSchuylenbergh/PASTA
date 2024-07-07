from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Sequence, overload

import numpy as np
from spectrum.spectrum import (
    ObservedSpectra,
    ObservedSpectrum,
    ResampledObservedSpectra,
    Spectra,
    Spectrum,
)

from pca.labels import (
    Label,
    Labels,
    LabelScaling,
    ObservedLabel,
    ObservedLabels,
    linear_scaler,
)


@dataclass
class TrainingData:
    """Training data of labeled spectra."""

    x: Spectra
    labels: Labels

    def with_rescaler(self, rescaler: LabelScaling) -> TrainingData:
        """Use a different scaling for the labels"""
        return TrainingData(self.x, self.labels.with_rescaler(rescaler))

    def pick(self, N, rng=None):
        """Pick N random samples from the training data"""
        if rng is None:
            return self[np.random.choice(len(self), N, replace=False)]
        else:
            return self[
                np.array(rng.choice(len(self), N, replace=False), dtype=np.uint32)
            ]

    def split(self, N: int) -> tuple[TrainingData, TrainingData]:
        """Split the training data into two parts"""
        return self[:N], self[N:]

    def sort(self, key: str = "teff") -> TrainingData:
        """Sort the training data by a label"""
        values = [getattr(label, key) for label in self.labels]
        indices = np.argsort(values)
        return TrainingData(self.x[indices], self.labels[indices])

    def filter_by_labels(self, labels: Labels) -> TrainingData:
        """Take only the samples with the given labels"""
        mask = np.isin(self.labels.hashes(), labels.hashes())
        return TrainingData(self.x[mask], self.labels[mask])  # type: ignore

    def cut(self, wl_min: float, wl_max: float) -> TrainingData:
        """Cut the spectra to a given wavelength range"""
        return TrainingData(self.x.cut(wl_min, wl_max), self.labels)

    def clip(self, _min: float | None, _max: float | None) -> TrainingData:
        """Clip the flux values"""
        return TrainingData(self.x.clip(_min, _max), self.labels)

    def resample_to(self, wl: np.ndarray, n_cores=1) -> TrainingData:
        """Resample the spectra to a new wavelength grid"""
        return TrainingData(self.x.resample_to(wl, n_cores), self.labels)

    def rescale_median(self) -> TrainingData:
        """Rescale the spectra to"""
        return TrainingData(self.x.rescale_median(), self.labels)

    def rescale(self, spec_array: np.ndarray) -> TrainingData:
        """Rescale the flux of every spectrum by a given array of factors"""
        return TrainingData(self.x.rescale(spec_array), self.labels)

    def mean(self) -> np.ndarray:
        """Return the mean of all spectra"""
        return self.x.mean()

    def rescale_and_shift(self, spec_array: np.ndarray) -> TrainingData:
        """Rescale all spectra by a given array and shift by unity."""
        return TrainingData(self.x.rescale_and_shift(spec_array), self.labels)

    def invert_from_max(self) -> TrainingData:
        """Invert all spectra by subtracting from the maximum value."""
        return TrainingData(self.x.invert_from_max(), self.labels)

    def invert_from_unity(self, clip=True) -> TrainingData:
        """Invert all spectra by subtracting from unity."""
        return TrainingData(self.x.invert_from_unity(clip), self.labels)

    def __len__(self) -> int:
        return len(self.x)

    @overload
    def __getitem__(self, i: slice | Sequence | np.ndarray) -> TrainingData: ...

    @overload
    def __getitem__(self: TrainingData, i: int) -> tuple[Spectrum, Label]: ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            return TrainingData(self.x[i], self.labels[i])
        elif isinstance(i, (list, np.ndarray)):
            return TrainingData(self.x[i], self.labels[i])
        elif isinstance(i, int):
            return self.x[i], self.labels[i]
        else:
            raise TypeError(f"Invalid index type {type(i)}")

    def __iter__(self) -> Generator[tuple[Spectrum, Label], None, None]:
        if not isinstance(self.x, Spectra):
            raise TypeError("Only spectra can be iterated")
        return ((self.x[i], self.labels[i]) for i in range(len(self)))


@dataclass
class RegressorTraining:
    """Training data of PCA coefficients and labels"""

    x: np.ndarray  # PCA coefficients
    labels: Labels

    def with_rescaler(self, rescaler: LabelScaling) -> RegressorTraining:
        """Use a different scaling for the labels"""
        return RegressorTraining(self.x, self.labels.with_rescaler(rescaler))

    def pick(self, N, rng=None):
        """Pick N random samples from the training data"""
        if rng is None:
            return self[np.random.choice(len(self), N, replace=False)]
        else:
            return self[
                np.array(rng.choice(len(self), N, replace=False), dtype=np.uint32)
            ]

    def split(self, N: int) -> tuple[RegressorTraining, RegressorTraining]:
        """Split the training data into two parts"""
        return self[:N], self[N:]

    def sort(self, key: str = "teff") -> RegressorTraining:
        """Sort the training data by a label"""
        values = [getattr(label, key) for label in self.labels]
        indices = np.argsort(values)
        return RegressorTraining(self.x[indices], self.labels[indices])

    def filter_by_labels(self, labels: Labels) -> RegressorTraining:
        """Take only the samples with the given labels"""
        mask = np.isin(self.labels.hashes(), labels.hashes())
        return RegressorTraining(self.x[mask], self.labels[mask])  # type: ignore

    def __len__(self) -> int:
        return len(self.x)

    @overload
    def __getitem__(self, i: slice | Sequence | np.ndarray) -> RegressorTraining: ...

    @overload
    def __getitem__(self: TrainingData, i: int) -> tuple[np.ndarray, Label]: ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            return RegressorTraining(self.x[i], self.labels[i])
        elif isinstance(i, (list, np.ndarray)):
            return RegressorTraining(self.x[i], self.labels[i])
        elif isinstance(i, int):
            return self.x[i], self.labels[i]
        else:
            raise TypeError(f"Invalid index type {type(i)}")

    def __iter__(self) -> Generator[tuple[Spectrum, Label], None, None]:
        if not isinstance(self.x, Spectra):
            raise TypeError("Only spectra can be iterated")
        return ((self.x[i], self.labels[i]) for i in range(len(self)))


@dataclass
class ObservedTrainingData:
    x: ObservedSpectra
    labels: ObservedLabels

    def apply_RV_and_resample(
        self,
        wl: np.ndarray,
        rescale_factor=1 / 1000,
        remove_flagged=True,
        n_cores=1,
    ) -> ObservedResampledTrainingData:
        """Apply RV shifts and resample the spectra to a new wavelength grid."""
        rvs = [l.rv for l in self.labels]
        if remove_flagged:
            include_list = [l.flag == 0 for l in self.labels]
            labels = self.labels.filter(include_list)
        else:
            labels = self.labels
        return ObservedResampledTrainingData(
            self.x.apply_RV_and_resample(wl, rvs, rescale_factor, include_list),
            labels,
        )

    def as_resampled(self) -> ObservedResampledTrainingData:
        """
        If all spectra are already resampled to the same grid,
        convert to ResampledObservedSpectra object.
        """
        return ObservedResampledTrainingData(self.x.as_resampled(), self.labels)

    def cut(self, wl_min: float, wl_max: float) -> ObservedTrainingData:

        return ObservedTrainingData(self.x.cut(wl_min, wl_max), self.labels)

    def rescale_median(self) -> ObservedTrainingData:
        return ObservedTrainingData(self.x.rescale_median(), self.labels)

    def invert_from_max(self) -> ObservedTrainingData:
        return ObservedTrainingData(self.x.invert_from_max(), self.labels)

    def invert_from_unity(self) -> ObservedTrainingData:
        return ObservedTrainingData(self.x.invert_from_unity(), self.labels)

    def clip(
        self, min_val: Optional[float], max_val: Optional[float]
    ) -> ObservedTrainingData:
        return ObservedTrainingData(self.x.clip(min_val, max_val), self.labels)

    @overload
    def __getitem__(self, i: slice) -> ObservedTrainingData: ...

    @overload
    def __getitem__(self, i: int) -> tuple[ObservedSpectrum, ObservedLabel]: ...

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.x[i], self.labels[i]
        elif isinstance(i, slice):
            return ObservedTrainingData(self.x[i], self.labels[i])
        else:
            raise TypeError(f"Invalid index type {type(i)}")

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class ObservedResampledTrainingData:
    x: ResampledObservedSpectra
    labels: ObservedLabels

    @classmethod
    def from_training_with_noise(
        cls, training_data: TrainingData, snr: float
    ) -> ObservedResampledTrainingData:
        return ObservedResampledTrainingData(
            training_data.x.add_noise(snr),
            ObservedLabels.from_labels(training_data.labels),
        )

    def cut(self, wl_min: float, wl_max: float) -> ObservedResampledTrainingData:
        """Cut the spectra to a given wavelength range"""
        return ObservedResampledTrainingData(self.x.cut(wl_min, wl_max), self.labels)

    def rescale_median(self) -> ObservedResampledTrainingData:
        """Rescale every spectrum by its median value"""
        return ObservedResampledTrainingData(self.x.rescale_median(), self.labels)

    def mean(self) -> np.ndarray:
        """Return the mean of each spectrum"""
        return self.x.mean()

    def rescale(self, spec_array: np.ndarray) -> ObservedResampledTrainingData:
        """Rescale every spectrum by a given array of factors"""
        return ObservedResampledTrainingData(self.x.rescale(spec_array), self.labels)

    def rescale_and_shift(
        self, spec_array: np.ndarray
    ) -> ObservedResampledTrainingData:
        """Rescale every spectrum by a given array and shift by unity."""
        return ObservedResampledTrainingData(
            self.x.rescale_and_shift(spec_array), self.labels
        )

    def invert_from_unity(self) -> ObservedResampledTrainingData:
        """Invert every spectrum by subtracting from unity."""
        return ObservedResampledTrainingData(self.x.invert_from_unity(), self.labels)

    def invert_from_max(self) -> ObservedResampledTrainingData:
        """Invert every spectrum by subtracting from the maximum value."""
        return ObservedResampledTrainingData(self.x.invert_from_max(), self.labels)

    def clip(
        self, min_val: Optional[float], max_val: Optional[float]
    ) -> ObservedResampledTrainingData:
        """Clip the flux values"""
        return ObservedResampledTrainingData(self.x.clip(min_val, max_val), self.labels)

    def to_training_data(self, rescaler: LabelScaling = linear_scaler) -> TrainingData:
        """Convert to TrainingData object"""
        return TrainingData(self.x.to_spectra(), self.labels.to_labels(rescaler))

    @overload
    def __getitem__(self, i: slice) -> ObservedResampledTrainingData: ...

    @overload
    def __getitem__(self, i: int) -> tuple[ObservedSpectrum, ObservedLabel]: ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ObservedResampledTrainingData(self.x[i], self.labels[i])
        elif isinstance(i, int):
            return self.x[i], self.labels[i]
        else:
            raise TypeError(f"Invalid index type {type(i)}")
