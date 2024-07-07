from __future__ import annotations

import numpy as np
from sklearn import decomposition

from pca.decompositionABC import DimensionalityReducer, TrainedDimensionalityReducer
from pca.labels import Labels
from spectrum.spectrum import Spectra, Spectrum
from pca.training_data import TrainingData


class PCA(DimensionalityReducer):

    def __init__(self, n_components: int, **kwargs) -> None:
        self.pca = decomposition.PCA(n_components=n_components, **kwargs)

    def train(self, training_data: TrainingData) -> TrainedPCA:
        coefficients = self.pca.fit_transform(training_data.x.array)
        return TrainedPCA(
            self.pca, training_data.x.wl, coefficients, training_data.labels
        )


class TrainedPCA(TrainedDimensionalityReducer):

    def __init__(
        self,
        pca: decomposition.PCA,
        wl: np.ndarray,
        training_coefficients: np.ndarray,
        training_labels: Labels,
    ) -> None:
        self.pca = pca
        self.wl = wl
        self.training_coefficients = training_coefficients
        self.training_labels = training_labels

    def transform_single(self, spectrum: np.ndarray) -> np.ndarray:
        return self.pca.transform(spectrum.reshape(1, -1))[0]

    def transform(self, spectra: Spectra) -> np.ndarray:
        return self.pca.transform(spectra.array)

    def inverse_transform(self, coefficients: np.ndarray) -> Spectra:
        return Spectra(self.wl, self.pca.inverse_transform(coefficients))

    def inverse_transform_single(self, coefficients: np.ndarray) -> Spectrum:
        return Spectrum(self.wl, self.pca.inverse_transform(coefficients))

    @property
    def components(self) -> np.ndarray:
        return self.pca.components_
