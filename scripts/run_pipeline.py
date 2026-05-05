"""Run the end-to-end pipeline."""

from __future__ import annotations

import argparse

from graph_fraud.pipeline import run_pipeline


def main() -> None:
    """Run the pipeline and print metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-synthetic-if-missing", action="store_true")
    args = parser.parse_args()

    metrics = run_pipeline(use_synthetic_if_missing=args.use_synthetic_if_missing)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
