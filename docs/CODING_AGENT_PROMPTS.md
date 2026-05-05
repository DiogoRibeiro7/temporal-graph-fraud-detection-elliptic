# Coding Agent Prompts

## Prompt 1 — Rolling-origin backtesting

Implement rolling-origin temporal validation. Report PR-AUC, ROC-AUC, precision@K, recall@K, and labelled coverage for each fold.

## Prompt 2 — Leakage-safe graph features

Add graph features that only use labels available up to the training cutoff. Add tests that prove future labels are not used.

## Prompt 3 — GNN training loop

Complete the PyTorch GCN training loop using labelled-node masks. Keep unknown labels in message passing, but not in supervised loss.

## Prompt 4 — Analyst queue simulation

Create an analyst-capacity simulation. Rank transactions by risk score and report how many illicit transactions are surfaced at different review budgets.

## Prompt 5 — Fraud risk report

Generate a Markdown report with metrics, risk tiers, top transaction groups, temporal validation results, and governance cautions.
