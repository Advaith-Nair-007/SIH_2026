# SIH 2026 — SIH26146 Group B Starter

## Goal
Build the Graph & Intelligence pipeline:

CSV -> transaction graph -> address features -> Isolation Forest -> ranked anomalies

## Current data
- blockchain_transactions.csv
- network_events.csv
- ip_enrichment.csv
- group_b_features.csv

## Quick start

```bash
pip install -r requirements.txt
python src/build_graph.py
python src/build_features.py
python src/detect_anomalies.py
streamlit run app/dashboard.py
```

Outputs are written to `outputs/`.

## Important modelling choice
The authoritative transaction representation is:

Address -> Transaction -> Address

This preserves multi-input/multi-output Bitcoin transactions. We also derive an address-level graph for graph analytics. We do NOT invent a specific input-to-output amount mapping when a transaction has multiple inputs and outputs.

Coinbase transactions are kept in the transaction graph but do not create normal input-address edges because they have no inputs.

## First MVP features
- input/output transaction counts
- unique incoming/outgoing counterparties
- incoming/outgoing BTC volume
- average input/output amount
- degree
- PageRank
- betweenness centrality
- simple transaction velocity
- correlated network features from `group_b_features.csv`

Ground-truth labels are deliberately kept separate from model features. Add them later for evaluation if Group A supplies them.
