.PHONY: setup data results test lint all

setup:            ## create .venv from uv.lock
	uv sync --locked

data:             ## download the Cricsheet IPL JSON (one request, skipped if present)
	uv run python scripts/download_data.py

results: data     ## rebuild every table in results/ from the raw JSON (MCMC, ~5 min)
	uv run python -m cricket_ba.pipeline

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check src tests scripts

all: setup lint test results
