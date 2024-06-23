from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

MELCHIORS_SPECTRA_DIR = PROJECT_ROOT / "MELCHIORS_Spectra"

hermesAnalyses = Path.home() / "hermesRun/HermesAnalyses"
HERMESNET = Path("/STER/hermesnet")
hermesVR_location = Path(
    "/STER/mercator/mercator/Hermes/releases/p3/hermes/pipeline/run/hermesVR.py"
)
nights_folder = Path("/STER/mercator/hermes")

GSSP_DIR = HERMESNET /"gssp"
