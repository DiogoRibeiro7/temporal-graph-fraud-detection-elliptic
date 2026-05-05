# Fraud Risk Report

This is a scaffold report. The pipeline writes metrics and scored transactions to `data/processed`.

## Intended use

Prioritize transactions for analyst review.

## Not intended use

Do not use the score as proof of illicit activity or as an automated enforcement mechanism.

## Required evaluation

- temporal validation
- PR-AUC
- precision@K
- recall@K
- false-positive review
- calibration analysis

## Human oversight

All high-risk transactions require analyst review.
