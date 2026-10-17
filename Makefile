# make data | split | eda | propensity | ate | memory | test | lint | format

.PHONY: data split eda propensity ate memory test lint format

data:            ## Download Criteo-UPLIFT (checksum verified) and stream it into Parquet
	./scripts/download_data.sh
	uv run python -m uplift.data.ingest

split:           ## Stratified train/val/test split (needs `make data` and src/uplift/config.py)
	uv run python -m uplift.data.split

eda:             ## Ties from float32 and duplicated feature vectors (needs `make split` and data/raw)
	uv run python -m uplift.data.eda

propensity:      ## Balance and propensity check, before any ATE (needs `make split`)
	uv run python -m uplift.propensity_check

ate:             ## ATE on visit and conversion with its three sensitivity versions (needs `make split`)
	uv run python -m uplift.ate

memory:          ## Peak memory of ingest, split and a full-train fit, 2 threads (needs `make data`)
	uv run python scripts/peak_memory.py

test:            ## The full suite, on synthetic data: no dataset needed
	uv run pytest -q

lint:            ## The same gate CI runs: lint and formatting
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:          ## Apply the formatting instead of checking it
	uv run ruff format src tests scripts
