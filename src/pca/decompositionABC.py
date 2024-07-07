from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pca.labels import Labels
from pca.training_data import TrainingData, RegressorTraining
from spectrum.spectrum import Spectra, Spectrum


class DimensionalityReducer(ABC):

    @abstractmethod
    def train(
        self, training: TrainingData
    ) -> TrainedDimensionalityReducer: ...


class TrainedDimensionalityReducer(ABC):

    def __init__(
        self,
        wl: np.ndarray,
        training_coefficients: np.ndarray,
        training_labels: Labels,
    ) -> None:
        self.wl = wl
        self.training_coefficients = training_coefficients
        self.training_labels = training_labels

    def transform_single(self, spectrum: Spectrum) -> np.ndarray:
        spectra = Spectra(self.wl, np.array([spectrum.flux]))
        return self.transform(spectra)[0]

    @abstractmethod
    def transform(self, spectra: Spectra) -> np.ndarray: ...

    def inverse_transform_single(self, coefficients: np.ndarray) -> Spectrum:
        coefficient = np.array([coefficients])
        return self.inverse_transform(coefficient)[0]

    @abstractmethod
    def inverse_transform(self, coefficients: np.ndarray) -> Spectra: ...

    def get_regressor_training(
        self, training_data: TrainingData
    ) -> RegressorTraining:
        return RegressorTraining(self.transform(training_data.x), training_data.labels)

    @property
    def regressor_training(self) -> RegressorTraining:
        return RegressorTraining(self.training_coefficients, self.training_labels)


class TrainedNMFABC(TrainedDimensionalityReducer):
    @property
    @abstractmethod
    def components(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def wl(self) -> np.ndarray: ...

    def resample_spec_and_get_components(
        self, spec: Spectrum
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wl_min = np.max([self.wl[0], spec.wl[0]])
        wl_max = np.min([self.wl[-1], spec.wl[-1]])

        i_min = np.searchsorted(self.wl, wl_min)
        i_max = np.searchsorted(self.wl, wl_max)
        H = self.components[:, i_min:i_max]

        dispersion = self.wl[i_min:i_max]
        new_flux = spec.resample_to(dispersion).flux

        return (
            np.ascontiguousarray(dispersion),
            np.ascontiguousarray(new_flux),
            np.ascontiguousarray(H),
        )
