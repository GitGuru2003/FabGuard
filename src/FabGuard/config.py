"""Project-wide paths and constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"
MODELS_DIR = ROOT_DIR / "Models"
REPORTS_DIR = ROOT_DIR / "Reports"
FIGURES_DIR = REPORTS_DIR / "Figures"
TABLES_DIR = REPORTS_DIR / "Tables"

SECOM_DATA_FILE = RAW_DIR / "secom.data"
SECOM_LABELS_FILE = RAW_DIR / "secom_labels.data"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Preprocessing defaults (tune during the Sep 28 - Oct 11 phase)
MAX_MISSING_FRACTION = 0.5
