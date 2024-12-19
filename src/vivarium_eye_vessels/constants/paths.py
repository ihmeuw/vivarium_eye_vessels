from pathlib import Path

import vivarium_eye_vessels
from vivarium_eye_vessels.constants import metadata

BASE_DIR = Path(vivarium_eye_vessels.__file__).resolve().parent

ARTIFACT_ROOT = Path(
    f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/"
)
