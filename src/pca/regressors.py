from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

from pca.labels import Label, Labels, LabelScaling
from pca.training_data import RegressorTraining


def calculate_score(Y: Labels, Y_pred: Labels) -> np.ndarray:
    y = Y.array
    y_pred = Y_pred.array
    return np.mean((y - y_pred) ** 2, axis=0)


class CoefficientRegressor(ABC):
    @abstractmethod
    def train(
        self, trainingdata: RegressorTraining, **kwargs
    ) -> TrainedCoefficientRegressor: ...


class TrainedCoefficientRegressor(ABC):
    @abstractmethod
    def predict_single(self, coefficients: np.ndarray) -> Label: ...

    @abstractmethod
    def predict(self, coefficients: np.ndarray) -> Labels: ...

    def score(self, data: RegressorTraining) -> np.ndarray:
        return calculate_score(data.labels, self.predict(data.x))


class LinearRegressor(CoefficientRegressor):

    def train(
        self, training: RegressorTraining, multiplier=100
    ) -> TrainedLinearRegressor:
        Y = training.labels.array
        model = linear_model.LinearRegression().fit(training.x * multiplier, Y)
        return TrainedLinearRegressor(model, training.labels.rescaler, multiplier)


class TrainedLinearRegressor(TrainedCoefficientRegressor):

    def __init__(
        self,
        model: linear_model.LinearRegression,
        rescaler: LabelScaling,
        multiplier: float,
    ) -> None:
        self.model = model
        self.rescaler = rescaler
        self.multiplier = multiplier

    def predict_single(self, coefficients: np.ndarray) -> Label:
        return Label(*self.model.predict(coefficients.reshape(1, -1))[0] * self.multiplier, self.rescaler)  # type: ignore

    def predict(self, coefficients: np.ndarray) -> Labels:
        return Labels(self.model.predict(coefficients * self.multiplier), self.rescaler)


class LinearRegressorVsini(CoefficientRegressor):

    def train(self, training: RegressorTraining) -> TrainedLinearRegressorVsini:
        Y = training.labels.array[:, :-1]  # Don't include vsini
        model = linear_model.LinearRegression().fit(training.x, Y)
        return TrainedLinearRegressorVsini(model, training.labels.rescaler)


class TrainedLinearRegressorVsini(TrainedCoefficientRegressor):

    def __init__(
        self, model: linear_model.LinearRegression, rescaler: LabelScaling
    ) -> None:
        self.model = model
        self.rescaler = rescaler

    def predict_single(self, coefficients: np.ndarray) -> Label:
        vsini = coefficients[0]
        coeffs = coefficients[1:]
        teff, mh, logg = self.model.predict(coeffs.reshape(1, -1))[0]
        return Label(teff, mh, logg, vsini * 1e-2, self.rescaler)

    def predict(self, coefficients: np.ndarray) -> Labels:
        vsini = coefficients[:, 0]
        coeffs = coefficients[:, 1:]
        predictions = np.zeros((coefficients.shape[0], 4))
        predictions[:, :3] = self.model.predict(coeffs)
        predictions[:, 3] = vsini * 1e-2
        return Labels(predictions, self.rescaler)


class PolynomialRegressor(CoefficientRegressor):

    def __init__(self, degree: int = 2, multiplier=100) -> None:
        self.degree = degree
        self.multiplier = multiplier

    def train(
        self, training: RegressorTraining, **kwargs
    ) -> TrainedPolynomialRegressor:
        Y = training.labels.array
        poly = PolynomialFeatures(degree=self.degree, **kwargs)
        X_ = poly.fit_transform(training.x * self.multiplier)
        model = linear_model.LinearRegression().fit(X_, Y)
        return TrainedPolynomialRegressor(
            model, poly, training.labels.rescaler, self.multiplier
        )


class TrainedPolynomialRegressor(TrainedCoefficientRegressor):

    def __init__(
        self,
        model: linear_model.LinearRegression,
        poly: PolynomialFeatures,
        rescaler: LabelScaling,
        multiplier: float,
    ) -> None:
        self.model = model
        self.poly = poly
        self.rescaler = rescaler
        self.multiplier = multiplier

    def predict_single(self, coefficients: np.ndarray) -> Label:
        X_ = self.poly.fit_transform(coefficients.reshape(1, -1) * self.multiplier)
        return Label(*self.model.predict(X_)[0], self.rescaler)  # type: ignore

    def predict(self, coefficients: np.ndarray) -> Labels:
        X_ = self.poly.fit_transform(coefficients * self.multiplier)
        return Labels(self.model.predict(X_), self.rescaler)


class RidgeRegressor(CoefficientRegressor):

    def __init__(
        self, degree: int = 2, alpha: float = 1, multiplier: float | int = 100, **kwargs
    ) -> None:
        self.degree = degree
        self.alpha = alpha
        self.kwargs = kwargs
        self.multiplier = multiplier

    def train(self, training: RegressorTraining, **kwargs) -> TrainedRidgeRegressor:
        Y = training.labels.array
        poly = PolynomialFeatures(degree=self.degree, **kwargs)
        X_ = poly.fit_transform(training.x * self.multiplier)
        model = linear_model.Ridge(alpha=self.alpha, **self.kwargs).fit(X_, Y)
        return TrainedRidgeRegressor(
            model, poly, training.labels.rescaler, self.multiplier
        )


class TrainedRidgeRegressor(TrainedCoefficientRegressor):

    def __init__(
        self,
        model: linear_model.Ridge,
        poly: PolynomialFeatures,
        rescaler: LabelScaling,
        multiplier: float,
    ) -> None:
        self.model = model
        self.poly = poly
        self.rescaler = rescaler
        self.multiplier = multiplier

    def predict_single(self, coefficients: np.ndarray) -> Label:
        X_ = self.poly.fit_transform(coefficients.reshape(1, -1) * self.multiplier)
        return Label(*self.model.predict(X_)[0], self.rescaler)  # type: ignore

    def predict(self, coefficients: np.ndarray) -> Labels:
        X_ = self.poly.fit_transform(coefficients * self.multiplier)
        return Labels(self.model.predict(X_), self.rescaler)


class RidgeCVRegressor(CoefficientRegressor):

    def __init__(self, degree: int = 2, multiplier=100, **kwargs) -> None:
        self.degree = degree
        self.kwargs = kwargs
        self.multiplier = multiplier

    def train(self, training: RegressorTraining, **kwargs) -> TrainedRidgeCVRegressor:
        Y = training.labels.array
        poly = PolynomialFeatures(degree=self.degree, **kwargs)
        X_ = poly.fit_transform(training.x * self.multiplier)
        model = linear_model.RidgeCV(**self.kwargs).fit(X_, Y)
        return TrainedRidgeCVRegressor(
            model, poly, training.labels.rescaler, self.multiplier
        )


class TrainedRidgeCVRegressor(TrainedCoefficientRegressor):

    def __init__(
        self,
        model: linear_model.RidgeCV,
        poly: PolynomialFeatures,
        rescaler: LabelScaling,
        multiplier: float,
    ) -> None:
        self.model = model
        self.poly = poly
        self.rescaler = rescaler
        self.multiplier = multiplier

    def predict_single(self, coefficients: np.ndarray) -> Label:
        X_ = self.poly.fit_transform(coefficients.reshape(1, -1) * self.multiplier)
        return Label(*self.model.predict(X_)[0], self.rescaler)  # type: ignore

    def predict(self, coefficients: np.ndarray) -> Labels:
        X_ = self.poly.fit_transform(coefficients * self.multiplier)
        return Labels(self.model.predict(X_), self.rescaler)


class PolynomialRegressorVsini(CoefficientRegressor):

    def __init__(self, degree: int = 2) -> None:
        self.degree = degree

    def train(
        self, training: RegressorTraining, **kwargs
    ) -> TrainedPolynomialRegressorVsini:
        Y = training.labels.array[:, :-1]  # Don't include vsini
        poly = PolynomialFeatures(degree=self.degree, **kwargs)
        X_ = poly.fit_transform(training.x)
        model = linear_model.LinearRegression().fit(X_, Y)
        return TrainedPolynomialRegressorVsini(model, poly, training.labels.rescaler)


class TrainedPolynomialRegressorVsini(TrainedCoefficientRegressor):

    def __init__(
        self,
        model: linear_model.LinearRegression,
        poly: PolynomialFeatures,
        rescaler: LabelScaling,
    ) -> None:
        self.model = model
        self.poly = poly
        self.rescaler = rescaler

    def predict_single(self, coefficients: np.ndarray) -> Label:
        vsini = coefficients[0]
        coeffs = coefficients[1:]
        X_ = self.poly.fit_transform(coeffs.reshape(1, -1))
        teff, m, logg = self.model.predict(X_)[0]
        return Label(teff, m, logg, vsini * 1e-2, self.rescaler)

    def predict(self, coefficients: np.ndarray) -> Labels:
        vsini = coefficients[:, 0]
        coeffs = coefficients[:, 1:]
        X_ = self.poly.fit_transform(coeffs)
        predictions = np.zeros((coefficients.shape[0], 4))
        predictions[:, :3] = self.model.predict(X_)
        predictions[:, 3] = vsini * 1e-2
        return Labels(predictions, self.rescaler)


class GPRegressor(CoefficientRegressor):

    def __init__(self, kernel: Kernel = RBF(), **kwargs) -> None:
        self.gaussianProcessRegressor = GaussianProcessRegressor(
            kernel=kernel, **kwargs
        )

    def train(self, training: RegressorTraining, **kwargs) -> TrainedGPRegressor:
        Y = training.labels.array
        model = self.gaussianProcessRegressor.fit(training.x, Y, **kwargs)
        return TrainedGPRegressor(model, training.labels.rescaler)


class GPRegressorAlpha(CoefficientRegressor):
    def __init__(
        self, kernel: Kernel = RBF(), guess=0.1, bounds=[1e-5, 1], **kwargs
    ) -> None:
        self.kernel = kernel
        self.guess = guess
        self.bounds = bounds
        self.init_kwargs = kwargs

    def train_fixed_alpha(self, alpha: float, training: RegressorTraining, **kwargs):
        Y = training.labels.array
        model = GaussianProcessRegressor(
            kernel=self.kernel, alpha=alpha, **self.init_kwargs
        ).fit(training.x, Y, **kwargs)
        return TrainedGPRegressor(model, training.labels.rescaler)

    def train(
        self,
        training: RegressorTraining,
        evaluation: RegressorTraining,
        print_opt=False,
        **kwargs,
    ) -> TrainedGPRegressor:
        def chi2(alpha: float) -> float:
            model = self.train_fixed_alpha(alpha, training, **kwargs)
            score = model.score(evaluation)
            return np.sum(score)

        result = opt.minimize(chi2, [self.guess], bounds=[self.bounds])
        if print_opt:
            print(result)

        alpha = result.x

        model = self.train_fixed_alpha(alpha, training, **kwargs)
        return model


class TrainedGPRegressor(TrainedCoefficientRegressor):

    def __init__(self, model: GaussianProcessRegressor, rescaler: LabelScaling) -> None:
        self.model = model
        self.rescaler = rescaler

    def predict_single(self, coefficients: np.ndarray) -> Label:
        return Label(
            *self.model.predict(coefficients.reshape(1, -1), return_std=False)[0]
        )

    def predict(self, coefficients: np.ndarray) -> Labels:
        return Labels(
            np.array(self.model.predict(coefficients, return_std=False)), self.rescaler
        )


class NNRegressor(CoefficientRegressor):

    def __init__(self, hidden_layer_sizes: tuple[int, ...], **kwargs) -> None:
        self.nn = MLPRegressor(hidden_layer_sizes, **kwargs)

    def train(self, training: RegressorTraining) -> TrainedCoefficientRegressor:
        self.nn.fit(training.x, training.labels.array)
        return TrainedNNRegressor(self.nn, training.labels.rescaler)


class TrainedNNRegressor(TrainedCoefficientRegressor):

    def __init__(self, nn: MLPRegressor, rescaler: LabelScaling) -> None:
        self.nn = nn
        self.rescaler = rescaler

    def predict_single(self, coefficients: np.ndarray) -> Label:
        return Label(*self.nn.predict(coefficients.reshape(1, -1)))

    def predict(self, coefficients: np.ndarray) -> Labels:
        return Labels(self.nn.predict(coefficients), self.rescaler)


def plot_accuracy(
    labels: Labels, predictions: Labels, axes: Optional[np.ndarray] = None, set_lim=True
) -> np.ndarray:
    if axes is None:
        fig, axess = plt.subplots(2, 2, figsize=(10, 10))
        axes_ = axess.ravel()
    else:
        axes_ = axes

    axes_[0].scatter(labels.teff, predictions.teff, s=5)
    T_min = min(labels.teff)
    T_max = max(labels.teff)
    axes_[0].plot([T_min, T_max], [T_min, T_max], "r--")
    axes_[0].set_xlabel("Teff (K)")
    axes_[0].set_ylabel("Predicted Teff (K)")
    if set_lim:
        l = T_max - T_min
        axes_[0].set_xlim(T_min - l / 10, T_max + l / 10)
        axes_[0].set_ylim(T_min - l / 10, T_max + l / 10)

    axes_[1].scatter(labels.mh, predictions.mh, s=5)
    M_min = min(labels.mh)
    M_max = max(labels.mh)
    axes_[1].plot([M_min, M_max], [M_min, M_max], "r--")
    axes_[1].set_xlabel("Metallicity (dex)")
    axes_[1].set_ylabel("Predicted Metallicity (dex)")
    if set_lim:
        l = M_max - M_min
        axes_[1].set_xlim(M_min - l / 10, M_max + l / 10)
        axes_[1].set_ylim(M_min - l / 10, M_max + l / 10)

    axes_[2].scatter(labels.logg, predictions.logg, s=5)
    g_min = min(labels.logg)
    g_max = max(labels.logg)
    axes_[2].plot([g_min, g_max], [g_min, g_max], "r--")
    axes_[2].set_xlabel("logg (dex)")
    axes_[2].set_ylabel("Predicted logg (dex)")
    if set_lim:
        l = g_max - g_min
        axes_[2].set_xlim(g_min - l / 10, g_max + l / 10)
        axes_[2].set_ylim(g_min - l / 10, g_max + l / 10)

    axes_[3].scatter(labels.vsini, predictions.vsini, s=5)
    vsini_min = min(labels.vsini)
    vsini_max = max(labels.vsini)
    axes_[3].plot([vsini_min, vsini_max], [vsini_min, vsini_max], "r--")
    axes_[3].set_xlabel("vsini (km/s)")
    axes_[3].set_ylabel("Predicted vsini (km/s)")
    if set_lim:
        l = vsini_max - vsini_min
        axes_[3].set_xlim(vsini_min - l / 10, vsini_max + l / 10)
        axes_[3].set_ylim(vsini_min - l / 10, vsini_max + l / 10)

    return axes_
