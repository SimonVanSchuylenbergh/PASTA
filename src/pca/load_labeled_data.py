from multiprocessing import Pool
from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from definitions import HERMESNET, PROJECT_ROOT
from Literature.simbad_data import SimbadData
from nmf.observed_data import (
    ObservedLabel,
    ObservedLabels,
    ObservedSpectra,
    ObservedTrainingData,
)
from typing import Optional, Literal
from pca.spectrum import (
    Label,
    Labels,
    ObservedSpectra,
    ObservedSpectrum,
    TrainingData,
    Spectra,
    Spectrum,
)

LABEL_FILE = PROJECT_ROOT / "src/nmf/training_labels_final.txt"


def read_spectrum(path: Path) -> ObservedSpectrum:
    with fits.open(path) as image:
        flux = image[0].data  # type: ignore
        wl = np.exp(
            np.linspace(
                image[0].header["CRVAL1"],  # type: ignore
                image[0].header["CDELT1"] * (len(flux) - 1) + image[0].header["CRVAL1"],  # type: ignore
                len(flux),
            )
        )
    with fits.open(str(path).replace("merged", "mergedVar")) as image:
        var = image[0].data  # type: ignore

    return ObservedSpectrum(wl, flux, var)


def read_labels(file: Path) -> ObservedLabels:
    labels = np.loadtxt(file, delimiter=" ")[:, :12]
    labels[:, 11] = 0
    return ObservedLabels(labels)


class LabelHandlerFits:
    def __init__(self, spectra_dir: Optional[Path]):
        self.simbad_data = SimbadData()
        self.spectra_dir = spectra_dir

    def __call__(self, label: ObservedLabel) -> tuple[ObservedSpectrum, ObservedLabel]:
        index = int(label.unseq)
        if self.spectra_dir is None:
            file = Path(self.simbad_data[str(index)].filepath())
        else:
            file = (
                self.spectra_dir
                / f"{index:08d}_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits"
            )
        return (read_spectrum(file), label)


class ObservedLabelHandlerNpy:
    def __init__(self, spectra_dir: Path, filetype: str):
        self.spectra_dir = spectra_dir
        self.filetype = filetype

    def __call__(self, label: ObservedLabel) -> tuple[ObservedSpectrum, ObservedLabel]:
        index = int(label.unseq)
        wl = np.load(self.spectra_dir / f"{index:08d}_wl.npy")
        y = np.load(self.spectra_dir / f"{index:08d}_{self.filetype}.npy")
        if self.filetype == "flux":
            var = np.load(self.spectra_dir / f"{index:08d}_var.npy")
            return (ObservedSpectrum(wl, y, var), label)
        elif self.filetype == "norm":
            var = np.load(self.spectra_dir / f"{index:08d}_norm_var.npy")
            return (ObservedSpectrum(wl, y, var), label)
        else:
            raise ValueError(f"Invalid filetype {self.filetype}")


class SynthLabelHandlerNpy:
    def __init__(self, spectra_dir: Path, filetype: str):
        self.spectra_dir = spectra_dir
        self.filetype = filetype

    def __call__(self, label: ObservedLabel) -> tuple[Spectrum, Label]:
        index = int(label.unseq)
        wl = np.load(self.spectra_dir / f"{index:08d}_wl.npy")
        y = np.load(self.spectra_dir / f"{index:08d}_{self.filetype}.npy")
        return (Spectrum(wl, y), label.to_label())


class SynthNoiseLabelHandlerNpy:
    def __init__(self, spectra_dir: Path, filetype: str):
        self.spectra_dir = spectra_dir
        self.filetype = filetype

    def __call__(self, label: ObservedLabel) -> tuple[ObservedSpectrum, ObservedLabel]:
        index = int(label.unseq)
        wl = np.load(self.spectra_dir / f"{index:08d}_wl.npy")
        synth_flux = np.load(self.spectra_dir / f"{index:08d}_{self.filetype}.npy")
        if self.filetype == "synth":
            var = np.load(self.spectra_dir / f"{index:08d}_var.npy")
            var[var == 0] = np.nan
            original_flux = np.load(self.spectra_dir / f"{index:08d}_flux.npy")
            noise = synth_flux / original_flux * np.sqrt(var)
            cleaned_noise = np.clip(np.nan_to_num(noise, nan=0), 0, None)
            new_flux = np.random.normal(synth_flux, cleaned_noise)
            return (ObservedSpectrum(wl, new_flux, noise**2), label)
        else:
            raise ValueError(f"Invalid filetype {self.filetype}")


def load_training_fits(
    label_file=LABEL_FILE,
    spectra_dir=None,
    min_n: int = 0,
    max_n: int = -1,
    n_cores=None,
    silent=False,
) -> ObservedTrainingData:
    labels = read_labels(label_file)[min_n:max_n]
    handle = LabelHandlerFits(spectra_dir)
    if n_cores == 1:
        if silent:
            out = [handle(file) for file in labels]
        else:
            out = [handle(file) for file in tqdm(labels)]
    else:
        with Pool(n_cores) as p:
            if silent:
                out = p.map(handle, labels)
            else:
                out = list(tqdm(p.imap(handle, labels), total=len(labels)))
    spectra = [x[0] for x in out]
    labels = [x[1] for x in out]
    return ObservedTrainingData(
        ObservedSpectra(spectra), ObservedLabels.from_list(labels)
    )


test_labels = np.loadtxt(PROJECT_ROOT / "src/nmf/testlabels.txt")
train_labels = np.loadtxt(PROJECT_ROOT / "src/nmf/trainlabels.txt")


def load_observed_npy(
    group: Literal["train", "test"],
    label_file=LABEL_FILE,
    spectra_dir=HERMESNET / "observed_npy",
    filetype: str = "flux",
    n_cores=None,
    silent=False,
) -> ObservedTrainingData:
    labels = read_labels(label_file).filter_unseq(
        train_labels if group == "train" else test_labels
    )
    handle = ObservedLabelHandlerNpy(spectra_dir, filetype)
    if n_cores == 1:
        if silent:
            out = [handle(file) for file in labels]
        else:
            out = [handle(file) for file in tqdm(labels)]
    else:
        with Pool(n_cores) as p:
            if silent:
                out = p.map(handle, labels)
            else:
                out = list(tqdm(p.imap(handle, labels), total=len(labels)))
    spectra = [x[0] for x in out]
    labels = [x[1] for x in out]
    return ObservedTrainingData(
        ObservedSpectra(spectra), ObservedLabels.from_list(labels)
    )


def load_synthetic_npy(
    group: Literal["train", "test"],
    label_file=LABEL_FILE,
    spectra_dir=HERMESNET / "observed_npy",
    filetype: str = "synth",
    n_cores=None,
    silent=False,
) -> TrainingData[Spectra]:
    labels = read_labels(label_file).filter_unseq(
        train_labels if group == "train" else test_labels
    )
    handle = SynthLabelHandlerNpy(spectra_dir, filetype)
    if n_cores == 1:
        if silent:
            out = [handle(file) for file in labels]
        else:
            out = [handle(file) for file in tqdm(labels)]
    else:
        with Pool(n_cores) as p:
            if silent:
                out = p.map(handle, labels)
            else:
                out = list(tqdm(p.imap(handle, labels), total=len(labels)))
    spectra = [x[0] for x in out]
    labels = [x[1] for x in out]
    return TrainingData(Spectra.from_list(spectra), Labels.from_list(labels))


def load_noisy_synthetic_npy(
    group: Literal["train", "test"],
    label_file=LABEL_FILE,
    spectra_dir=HERMESNET / "observed_npy",
    filetype: str = "synth",
    n_cores=None,
    silent=False,
) -> ObservedTrainingData:
    labels = read_labels(label_file).filter_unseq(
        train_labels if group == "train" else test_labels
    )

    handle = SynthNoiseLabelHandlerNpy(spectra_dir, filetype)
    if n_cores == 1:
        if silent:
            out = [handle(file) for file in labels]
        else:
            out = [handle(file) for file in tqdm(labels)]
    else:
        with Pool(n_cores) as p:
            if silent:
                out = p.map(handle, labels)
            else:
                out = list(tqdm(p.imap(handle, labels), total=len(labels)))
    spectra = [x[0] for x in out]
    labels = [x[1] for x in out]
    return ObservedTrainingData(
        ObservedSpectra(spectra), ObservedLabels.from_list(labels)
    )


def sort_files(files: list[Path]) -> list[Path]:
    return sorted(files, key=lambda x: int(x.name.split("_")[0]))


class NewFileHandler:
    def __init__(self, wl: np.ndarray):
        self.wl = wl

    @staticmethod
    def label_from_filename(filename: str) -> Label:
        teff, logg, m, vsini = filename.split("_")
        return Label.from_unscaled(
            float(teff), float(logg) / 1e3, float(m) / 1e3, float(vsini)
        )

    def __call__(self, path: Path) -> tuple[Spectrum, Label]:
        flux = np.load(path)
        return Spectrum(self.wl, flux), self.label_from_filename(path.stem)


def load_new_synth_training(
    location, min_n: int = 0, max_n: int = -1, n_cores=None, silent=False
) -> TrainingData[Spectra]:
    dir_ = Path(location)
    all_files = sort_files([f for f in dir_.glob("*.npy") if f.name != "wl.npy"])
    rng = np.random.default_rng(123)
    randomized = list(rng.permutation(np.array(all_files, dtype=object)))
    selection = randomized[min_n:max_n]
    handle = NewFileHandler(np.load(dir_ / "wl.npy"))
    if n_cores == 1:
        if silent:
            out = [handle(file) for file in selection]
        else:
            out = [handle(file) for file in tqdm(selection)]
    else:
        with Pool(n_cores) as p:
            if silent:
                out = p.map(handle, selection)
            else:
                out = list(tqdm(p.imap(handle, selection), total=len(selection)))
    spectra = [x[0] for x in out]
    labels = [x[1] for x in out]
    return TrainingData(
        Spectra(spectra[0].wl, np.array([s.flux for s in spectra])),
        Labels.from_list(labels),
    )


def load_all_synth_training(
    location, group: Literal["train", "test"], n_cores=None, silent=False
) -> TrainingData[Spectra]:
    dir_ = Path(location)
    wl = np.load(dir_ / "wl.npy")
    spectra: list[Spectrum] = []
    labels: list[Label] = []
    test_label_list = read_labels(
        PROJECT_ROOT / "rust-normalization/pso_05_44_100_latest.txt"
    ).filter_unseq(train_labels if group == "train" else test_labels)
    for label in test_label_list:
        labels.append(label.to_label())
        flux = np.load(dir_ / f"{int(label.unseq):08d}_synthflux.npy")
        spectra.append(Spectrum(wl, flux))

    return TrainingData(
        Spectra.from_list(spectra),
        Labels.from_list(labels),
    )
