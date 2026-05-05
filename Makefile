.PHONY: install install-gnn data synthetic pipeline train test lint format typecheck notebooks clean

install:
	poetry install --with dev

install-gnn:
	poetry install --with dev,gnn

data:
	poetry run python scripts/download_data.py

synthetic:
	poetry run python scripts/generate_synthetic_data.py

pipeline:
	poetry run python scripts/run_pipeline.py --use-synthetic-if-missing

train:
	poetry run python scripts/train_model.py --use-synthetic-if-missing

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

typecheck:
	poetry run mypy src

notebooks:
	poetry run jupyter lab

clean:
	rm -rf data/interim/* data/processed/* models/* reports/figures/*.png
	touch data/interim/.gitkeep data/processed/.gitkeep
