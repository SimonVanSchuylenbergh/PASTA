from pathlib import Path
from subprocess import run

from definitions import hermesAnalyses, hermesVR_location, nights_folder
from General.StringUtils import preface_char


def module_load(module: str) -> dict[str, str]:
    """
    Returns a dictionary with the environment variables that would be set by running `module load {module}`
    """
    code = run(
        ["/usr/share/lmod/lmod/libexec/lmod", "python", "load", module],
        capture_output=True,
    ).stdout.decode()
    new_code = code.replace("os.", "").replace(";", "")
    locals = {"environ": {}}
    exec(new_code, {}, locals)
    return locals["environ"]


def get_index_str(index: int) -> str:
    return preface_char(str(index), 8)


def get_VR_file(index: int) -> Path:
    return hermesAnalyses / (get_index_str(index) + "_AllCCF.data")


def clean_VR_folder():
    for file in hermesAnalyses.glob("*_ALLCCF.*"):
        file.unlink()


def get_hermes_logfile(index: int) -> Path:
    return hermesAnalyses / f"DRS_{get_index_str(index)}.log"


def at_IVS() -> bool:
    return hermesVR_location.exists()


def get_hermes_spectrum_path(
    index: int, kind: str, night: str | None = None, check_exists=True
) -> Path:
    """
    Find the filepath of a HERMES spectrum on IVS. Observation index and type
    (e.g. HRF_OBJ_ext_CosmicsRemoved_log_mergedVar_cf) are required.
    If no night is given, all nights will be searched.

    ## Parameters
    * index, int, index of the observation
    * type, str, type of the spectrum, e.g. HRF_OBJ_ext_log_merged
    * night, str | None, night of the observation.
    """
    if not at_IVS() and (night is None or not check_exists):
        raise ValueError("Cannot search files when not at IVS")

    index_str = get_index_str(index)
    night_folder = None
    if night is not None and check_exists:
        # Check if the night exists and the index is in there somewhere
        night_folder = nights_folder / night / "reduced"
        if not night_folder.exists():
            raise FileNotFoundError(f"Night {night} was not found")
        available_files = list(night_folder.glob(f"{index_str}_*.fits"))
        if len(available_files) == 0:
            raise FileNotFoundError(f"Index {index_str} was not found in night {night}")
    else:
        # Search all nights for the index
        for attempted_night in nights_folder.glob("*"):
            for file in (attempted_night / "raw").glob("*.fits"):
                if file.name.startswith(index_str):
                    night_folder = attempted_night / "reduced"
                    break
            if night_folder is not None:
                break
        if night_folder is None:
            raise FileNotFoundError(f"Index {index_str} was not found in any night")

    file = night_folder / f"{index_str}_{kind}.fits"
    # Check if requested type exists
    if not file.exists() and check_exists:
        available_files = list(night_folder.glob(f"{index_str}_*.fits"))
        raise FileNotFoundError(
            f"File {file.name} was not found, but the index was found in the night {night_folder.name}.\n"
            + "Available files are: \n"
            + "\n".join([str(x.name) for x in available_files])
        )
    return file
