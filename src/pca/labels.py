from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Sequence, overload

import numpy as np


class Scaler:
    """Rescaling values for a single label."""

    def __init__(self, _min, _max, log=False):
        self._min = _min
        self._max = _max
        self.log = log

    def scale(self, x: float | np.ndarray):
        if self.log:
            return (np.log(x) - np.log(self._min)) / (
                np.log(self._max) - np.log(self._min)
            )
        else:
            return (x - self._min) / (self._max - self._min)

    def unscale(self, x: float | np.ndarray):
        if self.log:
            return np.exp(
                x * (np.log(self._max) - np.log(self._min)) + np.log(self._min)
            )
        else:
            return x * (self._max - self._min) + self._min


class LabelScaling:
    """Rescaling that is applied to labels before using in regression."""

    def __init__(self, teff: Scaler, mh: Scaler, logg: Scaler, vsini: Scaler) -> None:
        self.teff = teff
        self.mh = mh
        self.logg = logg
        self.vsini = vsini

    def scale(
        self, teff: float, mh: float, logg: float, vsini: float
    ) -> tuple[float, float, float, float]:
        return (
            self.teff.scale(teff),
            self.mh.scale(mh),
            self.logg.scale(logg),
            self.vsini.scale(vsini),
        )

    def unscale(
        self, teff: float, mh: float, logg: float, vsini: float
    ) -> tuple[float, float, float, float]:
        return (
            self.teff.unscale(teff),
            self.mh.unscale(mh),
            self.logg.unscale(logg),
            self.vsini.unscale(vsini),
        )


linear_scaler = LabelScaling(
    Scaler(6000, 30_000), Scaler(-1, 1), Scaler(3.0, 5), Scaler(0, 600)
)

# Linear in teff, mh, logg and logarithmic in vsini
log_vsini_scaler = LabelScaling(
    Scaler(6000, 30_000), Scaler(-1, 1), Scaler(3.0, 5), Scaler(1, 600, log=True)
)

# Logarithmic in teff and vsini, linear in mh and logg
log_teff_and_vsini_scaler = LabelScaling(
    Scaler(6000, 30_000, log=True),
    Scaler(-1, 1),
    Scaler(3.0, 5),
    Scaler(1, 600, log=True),
)


class Label:
    """Label of a spectrum. Contains teff, mh, logg and vsini."""

    def __init__(
        self,
        _teff: float,
        _mh: float,
        _logg: float,
        _vsini: float,
        scaler: LabelScaling,
    ) -> None:
        self._teff = _teff
        self._mh = _mh
        self._logg = _logg
        self._vsini = _vsini
        self.scaler = scaler

    @classmethod
    def from_unscaled(
        cls,
        teff: float,
        mh: float,
        logg: float,
        vsini: float,
        scaler: LabelScaling = linear_scaler,
    ) -> Label:
        return Label(*scaler.scale(teff, mh, logg, vsini), scaler)

    def with_rescaler(self, rescaler: LabelScaling) -> Label:
        return Label.from_unscaled(self.teff, self.mh, self.logg, self.vsini, rescaler)

    def difference(self, other: Label) -> tuple[float, float, float, float]:
        """Difference between two labels."""
        return (
            self.teff - other.teff,
            self.mh - other.mh,
            self.logg - other.logg,
            self.vsini - other.vsini,
        )

    def relative_difference(self, other: Label) -> tuple[float, float, float, float]:
        """Relative difference between two labels."""
        return (
            (self.teff - other.teff) / self.teff,
            (self.mh - other.mh) / self.mh,
            (self.logg - other.logg) / self.logg,
            (self.vsini - other.vsini) / self.vsini,
        )

    def difference_relative_in_teff_vsini(
        self, other: Label
    ) -> tuple[float, float, float, float]:
        """Relative difference in teff and vsini, and absolute difference in mh and logg."""
        return (
            (self.teff - other.teff) / self.teff,
            (self.mh - other.mh),
            (self.logg - other.logg),
            (self.vsini - other.vsini) / self.vsini,
        )

    @property
    def teff(self) -> float:
        return self.scaler.teff.unscale(self._teff)

    @property
    def mh(self) -> float:
        return self.scaler.mh.unscale(self._mh)

    @property
    def logg(self) -> float:
        return self.scaler.logg.unscale(self._logg)

    @property
    def vsini(self) -> float:
        return self.scaler.vsini.unscale(self._vsini)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self._teff, self._mh, self._logg, self._vsini

    def __str__(self) -> str:
        return f"teff = {round(self.teff)} K, mh = {self.mh:.3f} dex, logg = {self.logg:.3f} dex, vsini = {self.vsini:.1f} km/s"

    def __repr__(self) -> str:
        return f"Label(teff={self.teff}, mh={self.mh}, logg={self.logg}, vsini={self.vsini})"

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Label):
            return False
        return self.as_tuple() == __value.as_tuple()


class Labels:
    """List of labels."""

    def __init__(self, array: np.ndarray, rescaler: LabelScaling) -> None:
        self.array = array
        self.rescaler = rescaler

    def with_rescaler(self, rescaler: LabelScaling) -> Labels:
        """Use a different rescaling"""
        return Labels.from_list([label.with_rescaler(rescaler) for label in self])

    @classmethod
    def from_list(cls, labels: list[Label]) -> Labels:
        return Labels(
            np.array([label.as_tuple() for label in labels], dtype=np.float64),
            labels[0].scaler,
        )

    @property
    def teff(self) -> np.ndarray:
        return self.rescaler.teff.unscale(self.array[:, 0])

    @property
    def mh(self) -> np.ndarray:
        return self.rescaler.mh.unscale(self.array[:, 1])

    @property
    def logg(self) -> np.ndarray:
        return self.rescaler.logg.unscale(self.array[:, 2])

    @property
    def vsini(self) -> np.ndarray:
        return self.rescaler.vsini.unscale(self.array[:, 3])

    def hashes(self) -> np.ndarray:
        return np.array([hash(l) for l in self])

    def difference(self, other: Labels) -> np.ndarray:
        return np.array([l.difference(o) for l, o in zip(self, other)])

    def relative_difference(self, other: Labels) -> np.ndarray:
        return np.array([l.relative_difference(o) for l, o in zip(self, other)])

    def difference_relative_in_teff_vsini(self, other: Labels) -> np.ndarray:
        return np.array(
            [l.difference_relative_in_teff_vsini(o) for l, o in zip(self, other)]
        )

    def RMSE(self, other: Labels) -> dict:
        """Root mean square error."""
        diff = self.difference(other)
        rmse = np.sqrt(np.nanmean(diff**2, axis=0))
        return {
            "teff": rmse[0],
            "mh": rmse[1],
            "logg": rmse[2],
            "vsini": rmse[3],
        }

    def RMSRE(self, other: Labels) -> dict:
        """Root mean square relative error."""
        diff = self.relative_difference(other)
        rmse = np.sqrt(np.nanmean(diff**2, axis=0))
        return {
            "teff": rmse[0],
            "mh": rmse[1],
            "logg": rmse[2],
            "vsini": rmse[3],
        }

    def RMSE_RMSRE_teff_vsini(self, other: Labels) -> dict:
        """RMSRE in teff and vsini, RMSE in mh and logg."""
        diff = self.difference_relative_in_teff_vsini(other)
        rmse = np.sqrt(np.nanmean(diff**2, axis=0))
        return {
            "teff": rmse[0],
            "mh": rmse[1],
            "logg": rmse[2],
            "vsini": rmse[3],
        }

    @overload
    def __getitem__(self, i: slice | Sequence | np.ndarray) -> Labels: ...

    @overload
    def __getitem__(self, i: int) -> Label: ...

    def __getitem__(self, i):
        if isinstance(i, (slice, list, np.ndarray)):
            return Labels(self.array[i], self.rescaler)
        elif isinstance(i, int):
            return Label(*self.array[i], self.rescaler)  # type: ignore
        else:
            raise TypeError(f"Invalid index type {type(i)}")

    def __iter__(self) -> Generator[Label, None, None]:
        return (Label(*label, self.rescaler) for label in self.array)  # type: ignore

    def __len__(self) -> int:
        return len(self.array)


@dataclass(frozen=True)
class ObservedLabel:
    """Labels of an observed spectrum, with RV and uncertainties."""

    unseq: str
    teff: float
    dteff: float
    mh: float
    dmh: float
    logg: float
    dlogg: float
    vsini: float
    dvsini: float
    rv: float
    drv: float
    flag: int

    def as_tuple(self):
        return (
            self.unseq,
            self.teff,
            self.dteff,
            self.mh,
            self.dmh,
            self.logg,
            self.dlogg,
            self.vsini,
            self.dvsini,
            self.rv,
            self.drv,
            self.flag,
        )

    def to_label(self, rescaler: LabelScaling = linear_scaler) -> Label:
        """Remove uncertainty and RV and convert to Label object."""
        return Label.from_unscaled(self.teff, self.mh, self.logg, self.vsini, rescaler)

    @classmethod
    def from_label(cls, label: Label) -> ObservedLabel:
        """
        Convert a Label object to an ObservedLabel object.
        Uncertainties and RV are set to 0.
        """
        return cls(
            "",
            label.teff,
            0,
            label.mh,
            0,
            label.logg,
            0,
            label.vsini,
            0,
            0,
            0,
            0,
        )

    def relative_difference(self, other: Label) -> tuple[float, float, float, float]:
        """Difference with another label, relative to the uncertainties."""
        return (
            (self.teff - other.teff) / self.dteff,
            (self.mh - other.mh) / self.dmh,
            (self.logg - other.logg) / self.dlogg,
            (self.vsini - other.vsini) / self.dvsini,
        )

    def difference_rel_teff_vsini(
        self, other: Label
    ) -> tuple[float, float, float, float]:
        """Difference with another labels, relative in teff and vsini, absolute in mh and logg."""
        return (
            (self.teff - other.teff) / self.teff,
            (self.mh - other.mh),
            (self.logg - other.logg),
            (self.vsini - other.vsini) / self.vsini,
        )

    def difference(self, other: Label) -> tuple[float, float, float, float]:
        """Difference with another label."""
        return (
            self.teff - other.teff,
            self.mh - other.mh,
            self.logg - other.logg,
            self.vsini - other.vsini,
        )

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Label):
            return False
        return self.as_tuple() == __value.as_tuple()

    def __str__(self) -> str:
        return ", ".join(
            [
                f"Teff=({round(self.teff)} +- {round(self.dteff)}) K",
                f"[Fe/H]={round(self.mh, 2)} +- {round(self.dmh, 2)}",
                f"logg={round(self.logg, 2)} +- {round(self.dlogg, 2)}",
                f"vsini={round(self.vsini, 2)} +- {round(self.dvsini, 2)} km/s",
                f"RV={round(self.rv, 2)} +- {round(self.drv, 2)} km/s",
                f"ID={int(self.unseq):08d}",
                f"flag={self.flag}",
            ]
        )


class ObservedLabels:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    @classmethod
    def from_list(cls, labels: list[ObservedLabel]) -> ObservedLabels:
        return cls(np.array([label.as_tuple() for label in labels], dtype=object))

    @classmethod
    def from_labels(cls, labels: Labels) -> ObservedLabels:
        return cls.from_list([ObservedLabel.from_label(label) for label in labels])

    def filter_unseq(
        self, unseqs: list[float] | np.ndarray, invert=False
    ) -> ObservedLabels:
        """Filter by unseq. If invert is True, keep all unseqs not in the list."""
        if invert:
            return self.from_list([l for l in self if int(l.unseq) not in unseqs])
        else:
            return self.from_list([l for l in self if int(l.unseq) in unseqs])

    def filter(self, include_list: list[bool] | np.ndarray) -> ObservedLabels:
        """Filter by include_list."""
        return self.from_list([l for l, include in zip(self, include_list) if include])

    def find(self, unseq: int) -> ObservedLabel:
        """Find label with a specific unseq."""
        return next(l for l in self if int(l.unseq) == unseq)

    def to_labels(
        self,
        rescaler: LabelScaling = linear_scaler,
        include_list: Optional[list[bool]] = None,
    ) -> Labels:
        """Convert to Labels object. If include_list is given, only include labels where include_list is True."""
        if not include_list:
            include_list = [True] * len(self)
        return Labels.from_list(
            [l.to_label(rescaler) for l, include in zip(self, include_list) if include]
        )

    def relative_difference(self, other: Labels) -> np.ndarray:
        """Relative differences between two sets of labels."""
        return np.array([l.relative_difference(o) for l, o in zip(self, other)])

    def difference_rel_teff_vsini(self, other: Labels) -> np.ndarray:
        """Relative difference in teff and vsini, and absolute difference in mh and logg."""
        return np.array([l.difference_rel_teff_vsini(o) for l, o in zip(self, other)])

    def difference(self, other: Labels) -> np.ndarray:
        """Differences between two sets of labels."""
        return np.array([l.difference(o) for l, o in zip(self, other)])

    def RMSE(self, other: Labels) -> dict:
        """Root mean square error."""
        diff = self.difference(other)
        rmse = np.sqrt(np.nanmean(diff**2, axis=0))
        return {
            "teff": rmse[0],
            "mh": rmse[1],
            "logg": rmse[2],
            "vsini": rmse[3],
        }

    def RMSRE(self, other: Labels) -> dict:
        """Root mean square relative error."""
        diff = self.relative_difference(other)
        rmse = np.sqrt(np.nanmean(diff**2, axis=0))
        return {
            "teff": rmse[0],
            "mh": rmse[1],
            "logg": rmse[2],
            "vsini": rmse[3],
        }

    def RMSE_rel_teff_vsini(self, other: Labels) -> dict:
        """RMSRE in teff and vsini, RMSE in mh and logg."""
        diff = self.difference_rel_teff_vsini(other)
        rmse = np.sqrt(np.nanmean(diff**2, axis=0))
        return {
            "teff": rmse[0],
            "mh": rmse[1],
            "logg": rmse[2],
            "vsini": rmse[3],
        }

    @property
    def teff(self) -> np.ndarray:
        return np.array([l.teff for l in self])

    @property
    def dteff(self) -> np.ndarray:
        return np.array([l.dteff for l in self])

    @property
    def mh(self) -> np.ndarray:
        return np.array([l.mh for l in self])

    @property
    def dmh(self) -> np.ndarray:
        return np.array([l.dmh for l in self])

    @property
    def logg(self) -> np.ndarray:
        return np.array([l.logg for l in self])

    @property
    def dlogg(self) -> np.ndarray:
        return np.array([l.dlogg for l in self])

    @property
    def vsini(self) -> np.ndarray:
        return np.array([l.vsini for l in self])

    @property
    def dvsini(self) -> np.ndarray:
        return np.array([l.dvsini for l in self])

    @property
    def RV(self) -> np.ndarray:
        return np.array([l.rv for l in self])

    @property
    def dRV(self) -> np.ndarray:
        return np.array([l.drv for l in self])

    def __len__(self) -> int:
        return len(self.array)

    @overload
    def __getitem__(self, i: slice | Sequence | np.ndarray) -> ObservedLabels: ...

    @overload
    def __getitem__(self, i: int) -> ObservedLabel: ...

    def __getitem__(self, i):
        if isinstance(i, (slice, list, np.ndarray)):
            return ObservedLabels(self.array[i])
        elif isinstance(i, int):
            return ObservedLabel(*self.array[i])
        else:
            raise TypeError(f"Invalid index type {type(i)}")

    def __iter__(self) -> Generator[ObservedLabel, None, None]:
        return (ObservedLabel(*label) for label in self.array)

    def __repr__(self) -> str:
        return f"Labels(array={self.array})"

    def __hash__(self) -> int:
        return hash(tuple(self.array))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ObservedLabels):
            return False
        return tuple(self.array) == tuple(__value.array)
