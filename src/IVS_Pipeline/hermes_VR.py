from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess, run

from definitions import hermesVR_location
from IVS_Pipeline.filesystem import get_VR_file, module_load

environ = module_load("hermesVR")


def run_hermesVR(options: str) -> CompletedProcess[bytes]:
    """
    Run hermesVR with supplied options
    """
    cmd = "python " + str(hermesVR_location) + " " + options
    p = run(cmd, shell=True, env=environ, check=True)
    return p


def parse_VR_output(filename: Path) -> tuple[float, float]:
    """
    Retrieve RV and uncertainty value from hermesVR output file
    """
    with open(filename, "r") as f:
        for line in f:
            if "Vr from orders 55-74" in line:
                VR_string = line.split(":")[1].strip()
                VR, _, dVR, *_ = VR_string.split(" ")
                return float(VR), float(dVR)

    raise ValueError(f"No RV found in file {filename}")


def compute_RV(
    index: int,
    night: str | None = None,
    mask: str | None = None,
    logfile: str | None = None,
    overwrite=False,
    use_existing=True,
) -> tuple[float, float]:
    """
    Run hermesVR pipeline for given index, night and logfile and return RV value
    """
    output_file = get_VR_file(index)
    if output_file.exists() and not overwrite and not use_existing:
        raise Exception(f"Output file {output_file} already exists")

    elif not output_file.exists() or overwrite:
        options = ""
        options += " -i " + str(index)
        options += " -g " + str(logfile)

        if night is not None:
            options += " -n " + night

        if mask is not None:
            options += " -m " + mask

        run_hermesVR(options)
    return parse_VR_output(output_file)
