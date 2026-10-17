# make data | split | test | lint | format

.PHONY: data split test lint format

data:            ## Download Criteo-UPLIFT (checksum verified) and stream it into Parquet
	./scripts/download_data.sh
	uv run python -m uplift.data.ingest

split:           ## Stratified train/val/test split (needs `make data` and src/uplift/config.py)
	uv run python -m uplift.data.split

test:            ## The full suite, on synthetic data: no dataset needed
	uv run pytest -q

lint:            ## The same gate CI runs: lint and formatting
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:          ## Apply the formatting instead of checking it
	uv run ruff format src tests scripts
