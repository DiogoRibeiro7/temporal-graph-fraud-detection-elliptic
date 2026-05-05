# Temporal Graph Fraud Detection — Elliptic Bitcoin Transactions

This repository focuses on one problem:

> Can we detect suspicious Bitcoin transactions by using both transaction-level features and the structure of the transaction graph?

The project is not another generic fraud classifier. It is designed to showcase graph analytics, temporal validation, class imbalance handling, and investigation-oriented risk scoring.

```text
transaction features
→ transaction graph
→ graph-derived signals
→ temporal validation
→ risk scores
→ analyst review workflow
```

## Dataset

The project expects the Kaggle dataset:

```text
ellipticco/elliptic-data-set
```

The raw data is not committed to the repository. Download it with:

```bash
make data
```

You need Kaggle credentials configured locally.

## What this repo showcases

- graph data loading from nodes and edges
- transaction-feature baseline models
- graph feature engineering with NetworkX
- leakage-aware temporal train/test splits
- precision@K and recall@K for analyst capacity
- optional graph neural network scaffold
- investigation workflow with risk tiers
- governance and model-risk documentation

## Repository structure

```text
notebooks/
├── 01_problem_framing_graph_fraud_detection.ipynb
├── 02_graph_data_exploration_nodes_edges_labels.ipynb
├── 03_baseline_transaction_feature_model.ipynb
├── 04_graph_feature_engineering_network_metrics.ipynb
├── 05_graph_neural_network_model.ipynb
├── 06_temporal_validation_and_leakage_control.ipynb
├── 07_risk_scores_and_investigation_workflow.ipynb
└── 08_limitations_governance_and_model_risk.ipynb
```

## Install

```bash
poetry install --with dev
```

For the optional GNN notebook:

```bash
poetry install --with dev,gnn
```

## Run with synthetic data

The repo includes a synthetic graph generator so the code can run without the Kaggle files.

```bash
make synthetic
make pipeline
make test
```

## Run with Kaggle data

```bash
make data
make pipeline
```

## Important caution

A high-risk score is not proof of illicit activity. Graph proximity is a signal, not evidence. This project is educational and analytical, not an enforcement system.
