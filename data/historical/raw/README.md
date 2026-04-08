# Historical Raw Lane

This folder holds the raw source data for the locked historical corpus.

## Layout

- `demand_load/`: load-only source files
- `weather/`: weather source files
- `calendar/`: holiday and calendar reference files
- `electricity_demand_*.csv`: combined raw dataset built from the source folders, including `is_holiday`
- `ingestion_manifest_*.json`: snapshot metadata and checksums

## Policy

- Keep source files separated by origin.
- Strip weather out of demand/load inputs.
- Refresh weather independently when source quality needs to be corrected.
- Preserve the combined CSV at the root as the merge target for processing.
