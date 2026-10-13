import os
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = ML_DIR.parent

SHARED_DIR = Path(os.environ.get("VASCORA_SHARED_DIR", REPO_DIR / "shared"))
DATA_DIR = Path(os.environ.get("VASCORA_DATA_DIR", ML_DIR / "data"))
ARTIFACTS_DIR = Path(os.environ.get("VASCORA_ARTIFACTS_DIR", ML_DIR / "artifacts"))
REPORTS_DIR = ML_DIR / "reports"

SCHEMA_PATH = SHARED_DIR / "feature_schema.json"
REGISTRY_PATH = SHARED_DIR / "vessel_registry.json"
DATASET_FILE = "extention of Z-Alizadeh sani dataset.xlsx"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/411/extention+of+z+alizadeh+sani+dataset.zip"
)

SEED = 42
TEST_SIZE = 0.2
