import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

ROOT_DIR = Path(__file__).parent.parent

GIT_SHA = "a50c6a780691165282b0aefee51aa8bcabeb1217"  # updated by post-commit


DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = ROOT_DIR / "logs"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
WANDB_DIR = ROOT_DIR  # Strangely enough, this will resolve to ROOT_DIR / wandb...

CSV_DELIM = ","
