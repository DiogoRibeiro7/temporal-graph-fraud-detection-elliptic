# Data Source

Expected Kaggle dataset slug:

```text
ellipticco/elliptic-data-set
```

Expected raw files:

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

Labels are mapped as:

```text
1 or illicit  -> 1
2 or licit    -> 0
unknown       -> missing
```

Unknown labels are excluded from supervised evaluation, but the nodes can still exist in the graph for neighbourhood structure.
