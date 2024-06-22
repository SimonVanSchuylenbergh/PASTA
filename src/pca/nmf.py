from __future__ import annotations

import numpy as np
from rust_nmf import RegularNMF as RegularRustNMF  # type: ignore
from rust_nmf import RegularNMFFitter  # type: ignore

from pca.decomposition import TrainedPCA
from pca.decompositionABC import DimensionalityReducer, TrainedNMFABC
from pca.labels import Labels
from pca.spectrum import ObservedSpectra, Spectra
from pca.training_data import TrainingData


class RegularNMF(DimensionalityReducer):

    def __init__(self, n_components: int, force_positive: bool = True) -> None:
        self.nmf = RegularNMFFitter(n_components, force_positive)

    def train(
        self,
        training_data: TrainingData[Spectra],
        iterations=1000,
        shift_mean=False,
        parallelize_loop=True,
        parallelize_mul=False,
    ) -> TrainedRegularNMF:
        trained_nmf = self.nmf.train(
            training_data.x.array,
            training_data.x.wl,
            iterations,
            shift_mean,
            parallelize_loop,
            parallelize_mul,
        )
        coefficients = trained_nmf.transform(training_data.x.array)
        return TrainedRegularNMF(trained_nmf, coefficients, training_data.labels)

    def train_fixed_weight(
        self,
        training_data: TrainingData[Spectra],
        W: np.ndarray,
        iterations=1000,
        shift_mean=False,
        parallelize_loop=True,
        parallelize_mul=False,
    ) -> TrainedRegularNMF:
        trained_nmf = self.nmf.train_fixed_weight(
            training_data.x.array,
            training_data.x.wl,
            W,
            iterations,
            shift_mean,
            parallelize_loop,
            parallelize_mul,
        )
        coefficients = trained_nmf.transform(training_data.x.array)
        return TrainedRegularNMF(trained_nmf, coefficients, training_data.labels)

    def train_with_guess(
        self,
        training_data: TrainingData[Spectra],
        H: np.ndarray,
        W: np.ndarray,
        iterations=1000,
        shift_mean=False,
        parallelize_loop=True,
        parallelize_mul=False,
    ) -> TrainedRegularNMF:
        trained_nmf = self.nmf.train_with_guess(
            training_data.x.array,
            training_data.x.wl,
            H,
            W,
            iterations,
            shift_mean,
            parallelize_loop,
            parallelize_mul,
        )
        coefficients = trained_nmf.transform(training_data.x.array)
        return TrainedRegularNMF(trained_nmf, coefficients, training_data.labels)


class TrainedRegularNMF(TrainedNMFABC):
    def __init__(
        self,
        nmf: RegularRustNMF,
        training_coefficients: np.ndarray,
        training_labels: Labels,
    ) -> None:
        self.nmf = nmf
        self.training_coefficients = training_coefficients
        self.training_labels = training_labels

    @classmethod
    def from_pca(cls, pca: TrainedPCA):
        guess = pca.training_coefficients.mean(axis=0)
        nmf = RegularRustNMF(pca.components, pca.pca.mean_, pca.wl, guess, False)
        return TrainedRegularNMF(nmf, pca.training_coefficients, pca.training_labels)

    def add_constant_component(self) -> TrainedRegularNMF:
        constant = np.ones(self.components.shape[1]) / 100
        new_nmf = self.nmf.add_component(constant)
        new_training_coef = np.hstack(
            [
                self.training_coefficients,
                np.zeros(self.training_coefficients.shape[0])[:, None],
            ]
        )
        return TrainedRegularNMF(new_nmf, new_training_coef, self.training_labels)

    def __getstate__(self) -> dict:
        return {
            "H": self.nmf.H,
            "mean": self.mean,
            "wl": self.nmf.wl,
            "guess": self.nmf.guess,
            "force_positive": self.nmf.force_positive,
            "training_coefficients": self.training_coefficients,
            "training_labels": self.training_labels,
        }

    def __setstate__(self, state: dict) -> None:
        self.nmf = RegularRustNMF(
            state["H"],
            state["mean"],
            state["wl"],
            state["guess"],
            state["force_positive"],
        )
        self.training_coefficients = state["training_coefficients"]
        self.training_labels = state["training_labels"]

    @property
    def wl(self) -> np.ndarray:
        return self.nmf.wl

    @property
    def components(self) -> np.ndarray:
        return np.array(self.nmf.H)

    @property
    def mean(self) -> np.ndarray:
        return self.nmf.mean

    def transform(self, spectra: Spectra) -> np.ndarray:
        return self.nmf.transform(spectra.array)

    def transform_with_rv(
        self, spectra: ObservedSpectra
    ) -> tuple[np.ndarray, np.ndarray]:
        rvs = self.nmf.fit_rv(spectra.wls, spectra.fluxs)
        shifted_spectra = spectra.apply_RV_and_resample(self.wl, rvs).to_spectra()
        transformed = self.transform(shifted_spectra)
        return rvs, transformed

    def inverse_transform(self, coefficients: np.ndarray) -> Spectra:
        return Spectra(self.wl, np.asarray(self.nmf.predict(coefficients)))

    def fit_rv(self, spectra: ObservedSpectra, parallelize=True) -> np.ndarray:
        return self.nmf.fit_rv(spectra.wls, spectra.fluxs, parallelize)
