"""Generate synthetic raw files matching the expected Elliptic filenames."""

from __future__ import annotations

from graph_fraud.config import RANDOM_SEED, RAW_DATA_DIR
from graph_fraud.synthetic import make_synthetic_transaction_graph


def main() -> None:
    """Generate synthetic graph CSV files."""
    nodes, edges = make_synthetic_transaction_graph(
        n_nodes=800,
        n_features=16,
        random_state=RANDOM_SEED,
    )
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    nodes.drop(columns=["label"]).rename(columns={"tx_id": "txId"}).to_csv(
        RAW_DATA_DIR / "elliptic_txs_features.csv",
        index=False,
    )

    classes = nodes[["tx_id", "label"]].rename(
        columns={"tx_id": "txId", "label": "class"}
    )
    classes["class"] = classes["class"].map({1.0: "1", 0.0: "2"}).fillna("unknown")
    classes.to_csv(RAW_DATA_DIR / "elliptic_txs_classes.csv", index=False)

    edges.rename(columns={"source": "txId1", "target": "txId2"}).to_csv(
        RAW_DATA_DIR / "elliptic_txs_edgelist.csv",
        index=False,
    )

    print(f"Synthetic data written to {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
