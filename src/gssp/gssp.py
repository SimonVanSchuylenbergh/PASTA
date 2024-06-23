from __future__ import annotations

import subprocess as sp
from abc import ABC
from copy import deepcopy
from dataclasses import astuple, dataclass, fields
from multiprocessing import Pool, cpu_count
from pathlib import Path
from shutil import rmtree
from typing import ClassVar

import numpy as np
from IVS_Pipeline.filesystem import module_load
from matplotlib import pyplot as plt
from spectrum.spectrum import ObservedSpectrum, Spectrum
from scipy.spatial import Delaunay
from tqdm import tqdm

from definitions import GSSP_DIR, PROJECT_ROOT
from General.StringUtils import preface_char
from gssp.gssp_inputfile import GSSPInput


def run_gssp(
    working_dir: Path,
    input_file: Path,
    n_cores: int,
    gssp_dir=GSSP_DIR,
    timeout=None,
    **kwargs,
) -> int:
    """
    Run the gssp executable from working_dir with the given input file.
    Meant to be used on the IVS system. The mpi module is loaded automatically.
    n_cores is passed to the mpirun command.
    """
    mpi_env = module_load("mpi/mpich-x86_64")
    wd = str(working_dir.absolute())
    exe = str((gssp_dir / "GSSP_single").absolute())
    infile = str(input_file.absolute())
    cmd = ["mpirun", "-n", str(n_cores), exe, infile]
    # p = sp.run(cmd, cwd=wd, env=mpi_env, **kwargs)
    p = sp.Popen(cmd, cwd=wd, env=mpi_env, **kwargs)
    try:
        return p.wait(timeout=timeout)
    except KeyboardInterrupt:
        p.kill()
        p.wait()
        raise KeyboardInterrupt

    # return p


@dataclass
class GSSPConfig:
    """Configuration of GSSP input files and folders"""
    working_dir: Path
    scratch_path: Path
    abundances: Path
    atmosphere_models: Path
    linelist: Path
    gssp_dir: Path = GSSP_DIR


@dataclass
class GridPoint:
    metallicity: float
    teff: float
    logg: float
    vmic: float
    vsini: float


@dataclass
class ParameterGrid:
    """
    Grid of parameters for GSSP input files. Ranges given as (first, step, last).
    """
    metallicity: tuple[float, float, float]
    teff: tuple[float, float, float]
    logg: tuple[float, float, float]
    vmic: tuple[float, float, float]
    vsini: tuple[float, float, float]

    def get_range(self, parameter: str) -> np.ndarray:
        """Retrieve a list of all possible values of one parameter."""
        first, step, last = getattr(self, parameter)
        if step == 0:
            raise ValueError("Step size cannot be zero")
        if last < first:
            raise ValueError("Last value must be larger than first value")
        if last == first:
            return np.array([first])
        if np.isclose((last - first) % step, 0):
            if isinstance(last, int):
                last += step
            else:
                last += step / 2
        return np.arange(first, last, step)

    def meshgrid(self) -> tuple[np.ndarray, ...]:
        """Create a numpy meshgrid of all parameters."""
        field_names = [field.name for field in fields(self)]
        return np.meshgrid(*[self.get_range(field) for field in field_names])

    def list_combinations(self) -> list[GridPoint]:
        """Cartesian product of all parameters, i.e. all possible combinations."""
        params = [x.flatten() for x in self.meshgrid()]
        return [GridPoint(*x) for x in zip(*params)]


@dataclass(kw_only=True)
class GSSPSetup(ABC):
    """All parameters for a single GSSP run. The mode is either 'grid' or 'fit'."""
    run_id: str
    config: GSSPConfig
    parameter_grid: ParameterGrid
    vmac: tuple[float, float, float, float]
    atmosphere_model_params: tuple[float, float]
    atmosphere_model_composition: str
    wavelength_regions: tuple[int, float]
    wl_range: tuple[float, float]
    mode: ClassVar[str] = ""
    observed_spectrum: ClassVar[Path] = Path("dev/null")
    rv_params: ClassVar[tuple[float, float, float, str]] = (0, 0, 0, "fixed")

    def to_inputfile(self) -> GSSPInput:
        """Build a GSSP input file from the setup."""
        return GSSPInput(
            run_id=self.run_id,
            scratch_path=self.config.scratch_path,
            teff=self.parameter_grid.teff,
            logg=self.parameter_grid.logg,
            vmic=self.parameter_grid.vmic,
            vsini=self.parameter_grid.vsini,
            dilution=("skip", 0, 0, 0),
            metallicity=(
                "skip",
                self.parameter_grid.metallicity[0],
                self.parameter_grid.metallicity[1],
                self.parameter_grid.metallicity[2],
            ),
            element_abu=("Fe", 0, 0, 0),
            vmac=self.vmac,
            abundances=self.config.abundances,
            atmosphere_models=self.config.atmosphere_models,
            linelist=self.config.linelist,
            atmosphere_model_params=self.atmosphere_model_params,
            atmosphere_model_composition=self.atmosphere_model_composition,
            wavelength_regions=(
                self.wavelength_regions[0],
                self.wavelength_regions[1],
                self.mode,
            ),
            observed_spectrum=self.observed_spectrum,
            rv_params=self.rv_params,
            wl_range=self.wl_range,
        )


class GridSetup(GSSPSetup):
    """Setup for a GSSP grid run with mode set to grid. Used to create a grid of model spectra."""
    mode = "grid"

    def compute(
        self, n_cores: int, remove_existing=False, timeout=None
    ) -> ComputedGrid:
        if remove_existing:
            rgs_files = self.config.working_dir / "rgs_files" / self.run_id
            output_files = self.config.working_dir / "output_files" / self.run_id
            scratch = self.config.scratch_path / self.run_id
            if rgs_files.exists():
                print(f"Removing {rgs_files}")
                rmtree(rgs_files)
            if output_files.exists():
                print(f"Removing {output_files}")
                rmtree(output_files)
            if scratch.exists():
                print(f"Removing {scratch}")
                rmtree(scratch)

        file = self.config.working_dir / "grid_input.inp"
        self.to_inputfile().save(file, comments=None)
        run_gssp(self.config.working_dir, file, n_cores, timeout=timeout)

        return ComputedGrid(self)


@dataclass(kw_only=True)
class FitSetup(GSSPSetup):
    """Setup for a GSSP fit run with mode set to fit. Used to fit a model spectrum to an observed spectrum."""
    mode: ClassVar[str] = "fit"
    observed_spectrum: Path
    rv_params: tuple[float, float, float, str]

    def compute(
        self, n_cores: int, remove_existing=False, timeout=None
    ) -> GSSPChi2Result:
        output_files = self.config.working_dir / "output_files" / self.run_id
        if output_files.exists():
            if remove_existing:
                print(f"Removing {output_files}")
                rmtree(output_files)
            else:
                raise FileExistsError(f"Output files {output_files} already exists")

        file = self.config.working_dir / "fit_input.inp"
        self.to_inputfile().save(file, comments=None)
        exitcode = run_gssp(self.config.working_dir, file, n_cores, timeout=timeout)
        if exitcode != 0:
            raise RuntimeError(f"Non-zero exit code {exitcode}")
        fitresult = GSSPChi2Result.from_file(output_files / "Chi2_table.dat")
        fitresult.check_gridpoint_correspondence(
            self.parameter_grid.list_combinations()
        )
        return fitresult


def get_rgs_filename(
    teff: float, logg: float, vmic: float, vsini: float, metallicity: float
) -> str:
    """Create a filename for a model spectrum based on its parameters."""
    met_abs = preface_char(str(int(abs(metallicity) * 10)), 4)
    met_str = "p" + met_abs if metallicity >= 0 else "m" + met_abs

    t_str = preface_char(str(int(teff)), 5)
    logg_str = preface_char(str(int(logg * 100)), 4)
    vmic_str = preface_char(str(int(vmic * 10)), 4)
    vsini_str = preface_char(str(int(vsini)), 4)
    return f"l{met_str}_{t_str}_{logg_str}_{vmic_str}_0000_Vsini_{vsini_str}.rgs"


def read_rgs_file(filename: Path) -> Spectrum:
    """Read a model spectrum from an rgs file."""
    return Spectrum(*np.loadtxt(filename, dtype=np.float32)[:, :2].T)


class ComputedGrid:
    """A computed grid of model spectra. Used to check the output of a grid run."""
    def __init__(self, setup: GridSetup, check=True):
        self.setup = deepcopy(setup)
        self.working_dir = self.setup.config.working_dir
        self.parameter_grid = self.setup.parameter_grid
        self.run_id = self.setup.run_id
        if check:
            self.check_model_overview()

    def check_model_overview(self, error=False):
        """Check the ModelOverview.txt file for failed models."""
        with open(
            self.working_dir / "output_files" / self.run_id / "ModelOverview.txt"
        ) as f:
            models = [x.split() for x in f.readlines()]
        for model in models:
            if model[-1] != "OK":
                if error:
                    raise ValueError(f"Model {model} failed")
                else:
                    print(f"Model {model[0]} failed")


@dataclass
class GridOutput:
    """Output of a grid run. Used to retrieve and load output rgs files."""
    rgs_dir: Path
    filenames: list[Path]
    models: list[GridPoint]

    def __post_init__(self):
        points = np.array(
            [astuple(gridpoint) for gridpoint in self.models], dtype=np.float64
        )

        constant_dimensions = {}
        for i in range(5):
            u = np.unique(points[:, i])
            if len(u) == 1:
                constant_dimensions[i] = u[0]

        bounds = [
            (np.min(points[:, i]).astype(float), np.max(points[:, i]).astype(float))
            for i in range(5)
            if i not in constant_dimensions.keys()
        ]

        self.bounds = bounds
        self.constant_dimensions = constant_dimensions

    @classmethod
    def from_output(cls, rgs_dir: Path) -> GridOutput:
        """Create a GridOutput object from a directory of rgs files."""
        if not rgs_dir.exists():
            raise FileNotFoundError(f"Directory {rgs_dir} doesn't exist")
        models: list[GridPoint] = []
        filenames = []
        for file in rgs_dir.glob("*.rgs"):
            met_str, teff_str, logg_str, v_mic, _, _, vsini_str = file.stem.split("_")
            met_abs = float(met_str[2:]) / 10
            mettalicity = met_abs if met_str[1] == "p" else -met_abs
            teff = float(teff_str)
            logg = float(logg_str) / 100
            vmic = float(v_mic) / 10
            vsini = float(vsini_str)
            models.append(GridPoint(mettalicity, teff, logg, vmic, vsini))
            filenames.append(file)
        return GridOutput(rgs_dir, filenames, models)

    def gridpoint_to_vec(self, gridPoint: GridPoint) -> list[float]:
        """Represent gridpoint as list of floats, excluding fixed parameters."""
        return [
            astuple(gridPoint)[i]
            for i in range(5)
            if i not in self.constant_dimensions.keys()
        ]

    def vec_to_gridpoint(self, vec: list[float]) -> GridPoint:
        """Create a GridPoint object from a list of floats, filling in fixed parameters."""
        v = list(vec)
        return GridPoint(
            *[self.constant_dimensions.get(i) or v.pop(0) for i in range(5)]
        )

    def get_file(self, gridPoint: GridPoint) -> Path:
        """Get the filename of a model gridpoint."""
        return self.rgs_dir / get_rgs_filename(
            gridPoint.teff,
            gridPoint.logg,
            gridPoint.vmic,
            gridPoint.vsini,
            gridPoint.metallicity,
        )

    def read_model_index(self, index: int) -> Spectrum:
        """Read a model spectrum from the grid output."""
        return read_rgs_file(self.filenames[index])

    def load_flux(self, index: int) -> np.ndarray:
        """Load the flux of a model spectrum from the grid output."""
        return read_rgs_file(self.filenames[index]).flux

    def read_model_gridpoint(self, gridPoint: GridPoint) -> Spectrum:
        """Read a model spectrum from the grid output."""
        return read_rgs_file(self.get_file(gridPoint))

    def load_all_fluxes(self, cpus: int | None = None) -> list[np.ndarray]:
        """Load the flux of all model spectra into a list of arrays."""
        if cpus is None:
            cpus = cpu_count() - 1 or 1
        with Pool(cpus) as p:
            fluxes = list(
                tqdm(
                    p.imap(self.load_flux, range(len(self.filenames))),
                    total=len(self.filenames),
                )
            )
        return fluxes

    def load_all_spectra(self, cpus: int | None = None) -> list[Spectrum]:
        """Load all model spectra into a list of Spectrum objects."""
        wavelengths = self.read_model_index(0).wl
        return [
            Spectrum(wavelengths, flux) for flux in self.load_all_fluxes(cpus)
        ]

    def triangulate(self) -> Delaunay:
        """Create a Delaunay triangulation of the varying parameters."""
        points = np.array(
            [astuple(gridpoint) for gridpoint in self.models], dtype=np.float64
        )
        varying_points = np.delete(
            points, list(self.constant_dimensions.keys()), axis=1
        )
        triangulation = Delaunay(varying_points)
        return triangulation


class ChiSquareGrid:
    """Rectangular grid of chi2 values for all parameters."""
    parameters = ["teff", "logg", "vmic", "vsini", "metal"]
    chi2_paramters = ["chi2_inter", "contin_factor", "reduced_chi2", "chi2_1sigma"]

    def __init__(self, paramlist: list[np.ndarray], chi2grid: np.ndarray):
        if len(paramlist) != 5:
            raise ValueError("Parameter list must have 5 elements")
        if chi2grid.ndim != 6:
            raise ValueError("Chi2 must be 5D")
        *s2, N = chi2grid.shape
        if N != 4:
            raise ValueError("Chi2 must have 4 columns")

        param_dims = [len(x) for x in paramlist]
        if param_dims != list(chi2grid.shape[:-1]):
            raise ValueError("Dimensions of parameter list and chi2 grid must match")

        self.paramlist = paramlist
        self.chi2grid = chi2grid

    def get_parameter_index(self, value: float, parameter: str | int):
        if isinstance(parameter, str):
            parameter = self.parameters.index(parameter)
        return np.argwhere(self.paramlist[parameter] == value)[0, 0]

    def get_index(
        self, teff: float, logg: float, vmic: float, vsini: float, metallicity: float
    ):
        return (
            self.get_parameter_index(teff, "teffs"),
            self.get_parameter_index(logg, "loggs"),
            self.get_parameter_index(vmic, "vmics"),
            self.get_parameter_index(vsini, "vsinis"),
            self.get_parameter_index(metallicity, "metals"),
        )

    def get_marginal_min(
        self, parameter: str | int, value: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(parameter, str):
            parameter = self.parameters.index(parameter)
        if isinstance(value, str):
            value = self.chi2_paramters.index(value)
        axis = [0, 1, 2, 3, 4]
        axis.pop(parameter)
        return self.paramlist[parameter], np.min(
            self.chi2grid[..., value], axis=tuple(axis)
        )

    def plot_landscape(self, axes: np.ndarray | None = None):
        if axes is None:
            fig, axes = plt.subplots(2, 3)
            assert axes is not None
            fig.delaxes(axes[1, 2])
        for parameter, ax in zip(self.parameters, axes.flatten()):
            ax.set_xlabel(parameter)
            ax.set_ylabel("Reduced chi2")
            ax.scatter(*self.get_marginal_min(parameter, "reduced_chi2"))
        return axes


class GSSPChi2Result:
    """
    Result of a GSSP fit run. Contains chi2 values for all gridpoints.
    Points do not need to be in a rectangular grid. Points are assumed to be sorted by chi2.
    """
    def __init__(self, gridpoints: list[GridPoint], chi2: np.ndarray):
        if chi2.ndim != 2:
            raise ValueError("Chi2 must be 2D")
        n, m = chi2.shape
        if m != 4:
            raise ValueError("Chi2 must have 4 columns")
        if n != len(gridpoints):
            raise ValueError("Number of gridpoints must match number of chi2 values")

        self.gridpoints = gridpoints
        self.chi2 = chi2

    @classmethod
    def from_file(cls, filename: Path) -> GSSPChi2Result:
        data = np.loadtxt(filename)
        m, n = data.shape
        if n != 10:
            raise ValueError("Input file must have 10 columns")
        gridpoints = [
            GridPoint(met, teff, logg, vmic, vsini)
            for met, teff, logg, vmic, abu, vsini in data[:, :6]
        ]
        chi2 = data[:, 6:]
        return cls(gridpoints, chi2)

    def check_gridpoint_correspondence(self, gridpoints: list[GridPoint]):
        if len(gridpoints) != len(self.gridpoints):
            raise ValueError("Number of gridpoints doesn't match")
        for gp1 in self.gridpoints:
            if gp1 not in gridpoints:
                raise ValueError(f"Gridpoint {gp1} not found in input gridpoints")
        for gp2 in gridpoints:
            if gp2 not in self.gridpoints:
                raise ValueError(f"Gridpoint {gp2} not found in output gridpoints")

    @property
    def best_gridpoint(self) -> GridPoint:
        return self.gridpoints[0]

    def to_chi2_grid(self) -> ChiSquareGrid:
        """Assume that the gridpoints form a rectangular grid and convert to a ChiSquareGrid."""
        teffs = np.sort(np.unique([g.teff for g in self.gridpoints]))
        loggs = np.sort(np.unique([g.logg for g in self.gridpoints]))
        vmics = np.sort(np.unique([g.vmic for g in self.gridpoints]))
        vsinis = np.sort(np.unique([g.vsini for g in self.gridpoints]))
        metals = np.sort(np.unique([g.metallicity for g in self.gridpoints]))

        paramlist = [teffs, loggs, vmics, vsinis, metals]
        chi2grid = np.zeros(
            (len(teffs), len(loggs), len(vmics), len(vsinis), len(metals), 4)
        )
        for gridpoint, chi2 in zip(self.gridpoints, self.chi2):
            i = np.argwhere(np.sort(np.unique(teffs)) == gridpoint.teff)[0, 0]
            j = np.argwhere(np.sort(np.unique(loggs)) == gridpoint.logg)[0, 0]
            k = np.argwhere(np.sort(np.unique(vmics)) == gridpoint.vmic)[0, 0]
            l = np.argwhere(np.sort(np.unique(vsinis)) == gridpoint.vsini)[0, 0]
            m = np.argwhere(np.sort(np.unique(metals)) == gridpoint.metallicity)[0, 0]
            chi2grid[i, j, k, l, m] = chi2

        return ChiSquareGrid(paramlist, chi2grid)


GSSP_CONFIG = GSSPConfig(
    PROJECT_ROOT / "gssp_workdir",
    Path("/scratch/simonv"),
    GSSP_DIR / "abundances/",
    GSSP_DIR / "LLmodels/",
    GSSP_DIR / "Radiative_transfer/VALD2012.lns",
)
