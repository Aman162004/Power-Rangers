import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import run_ingestion


if __name__ == "__main__":
    manifest = run_ingestion("config/config.yaml")
    print("Ingestion run completed successfully.")
    print(json.dumps(manifest["window"], indent=2))
