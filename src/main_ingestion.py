import json

from src.ingestion.pipeline import run_ingestion


if __name__ == "__main__":
    manifest = run_ingestion("config/config.yaml")
    print("Ingestion run completed successfully.")
    print(json.dumps(manifest["window"], indent=2))
