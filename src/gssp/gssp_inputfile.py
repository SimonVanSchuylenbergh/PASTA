from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import zip_longest
from pathlib import Path


def read_double(line: str) -> tuple[float, float]:
    a, b, *_ = line.split()
    return float(a), float(b)


def read_triple(line: str) -> tuple[float, float, float]:
    a, b, c, *_ = line.split()
    return float(a), float(b), float(c)


def read_quadruple(line: str) -> tuple[float, float, float, float]:
    a, b, c, d, *_ = line.split()
    return float(a), float(b), float(c), float(d)


def read_flagged_triple(line: str) -> tuple[str, float, float, float]:
    flag, a, b, c, *_ = line.split()
    return flag, float(a), float(b), float(c)


def field_to_string(x: str | tuple | Path, comment: str | None = None) -> str:
    if isinstance(x, str):
        s = x
    elif isinstance(x, tuple):
        s = " ".join([str(y) for y in x])
    elif isinstance(x, Path):
        if x.is_dir() or not x.exists():
            s = str(x) + "/"
        else:
            s = str(x)
    else:
        raise ValueError(f"Cannot convert {x} to string")
    if comment is None:
        return s
    else:
        return s + "     ! " + comment


default_comments = [
    "Run ID",
    "path",
    "effective temperature",
    "surface gravity",
    "microturbulent velocity",
    "projected rotational velocity",
    "dilution factor",
    "metallicity and switch for individual abundances",
    "individual abundance (He 0.0783 0.0005 0.0813) (Fe -4.99 0.05 -4.39); element ID",
    "macroturbulent velocity, resolving power",
    "abundance path",
    "atmosphere model path",
    "SynthV line list",
    "atmosphere model vmicro and mass",
    "model atmosphere chemical composition ID (ST-standard, CNm and CNh-moderately and heavily CN-cycled)",
    "Number of wavelength regions; step in wavelength; 'fit' or 'grid'",
    "Observed spectrum path",
    "RV scaling factor, continuum cutoff factor, RV value, RV option",
    "wavelength range in Angstroms",
]


@dataclass
class GSSPInput:
    """Represents a GSSP input file. The fields are named after the GSSP input file format."""
    run_id: str
    scratch_path: Path
    teff: tuple[float, float, float]
    logg: tuple[float, float, float]
    vmic: tuple[float, float, float]
    vsini: tuple[float, float, float]
    dilution: tuple[str, float, float, float]
    metallicity: tuple[str, float, float, float]
    element_abu: tuple[str, float, float, float]
    vmac: tuple[float, float, float, float]
    abundances: Path
    atmosphere_models: Path
    linelist: Path
    atmosphere_model_params: tuple[float, float]
    atmosphere_model_composition: str
    wavelength_regions: tuple[int, float, str]
    observed_spectrum: Path
    rv_params: tuple[float, float, float, str]
    wl_range: tuple[float, float]

    @classmethod
    def from_str(cls, inp: str) -> GSSPInput:
        lines = [
            line.split("!")[0].strip()
            for line in inp.split("\n")
            if line.split("!")[0].strip() != ""
        ]

        nranges, wave_step, mode, *_ = lines[15].split()
        RV_factor, contin_factor, RV, RV_flag, *_ = lines[17].split()

        return cls(
            run_id=lines[0],
            scratch_path=Path(lines[1]),
            teff=read_triple(lines[2]),
            logg=read_triple(lines[3]),
            vmic=read_triple(lines[4]),
            vsini=read_triple(lines[5]),
            dilution=read_flagged_triple(lines[6]),
            metallicity=read_flagged_triple(lines[7]),
            element_abu=read_flagged_triple(lines[8]),  #  Individual abundance
            vmac=read_quadruple(lines[9]),
            abundances=Path(lines[10]),
            atmosphere_models=Path(lines[11]),
            linelist=Path(lines[12]),
            atmosphere_model_params=read_double(lines[13]),
            atmosphere_model_composition=lines[14],
            wavelength_regions=(int(nranges), float(wave_step), mode),
            observed_spectrum=Path(lines[16]),
            rv_params=(float(RV_factor), float(contin_factor), float(RV), RV_flag),
            wl_range=read_double(lines[18]),
        )

    def to_string(self, comments: list[str] | None = default_comments) -> str:
        if comments is None:
            comments = []
        return (
            "\n".join(
                [
                    field_to_string(getattr(self, field.name), comment)
                    for field, comment in zip_longest(fields(self), comments)
                ]
            )
            + "\n"
        )

    @classmethod
    def from_file(cls, filename: Path) -> GSSPInput:
        with open(filename, "r") as f:
            inp = f.read()
        return cls.from_str(inp)

    def save(self, filename: Path, comments: list[str] | None = default_comments):
        with open(filename, "w") as f:
            f.write(self.to_string(comments))


if __name__ == "__main__":
    test_input = """
test ! Run ID
/scratch/ragnarv/   ! path
8000 100 8100       ! effective temperature
4.0 0.1 4.0               ! surface gravity
2.0 0.2 2.0               ! microturbulent velocity 
100 5 100                  ! projected rotational velocity
skip 0.90 0.01 0.97         ! dilution factor (unconstrained fitting of the disentangled spectra; skip = single star spectrum mode)
skip -0.1 0.1 0.0	         ! metallicity and switch for individual abundances
Cr -5.75 0.05 -5.30 ! individual abundance (He 0.0783 0.0005 0.0813) (Fe -4.99 0.05 -4.39); element ID
0.0 20.0 0.0 60000                 ! macroturbulent velocity, resolving power
/home/ragnarv/Documents/gssptest/abundances/ ! abundance path
/home/ragnarv/Documents/gssptest/LLmodels/    ! atmosphere model path
/home/ragnarv/Documents/gssptest/Radiative_transfer/VALD2012.lns  ! SynthV line list
2 1                       ! atmosphere model vmicro and mass
ST                        ! model atmosphere chemical composition ID (ST-standard, CNm and CNh-moderately and heavily CN-cycled)
1 0.02 grid ! Number of wavelength regions; step in wavelength; "fit" or "grid"
observed_spectra/KIC4150611/KIC4150611_primary_fdb_res.sep ! Observed spectrum path
0.5 100.97 -25.0 fixed ! Scaling factor for RV determination, number of points for running mean calculation (n*2+1), a cutoff for the continuum correction
4000.0 6000.0 ! KIC4150611
"""
    test = GSSPInput.from_str(test_input)
    print(test.to_string())
