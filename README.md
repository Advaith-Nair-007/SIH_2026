from pathlib import Path

readme = r'''<div align="center">

# ChainSentinel

### AI-Powered Bitcoin Transaction Intelligence and Anomaly Detection

**Smart India Hackathon 2026 | SIH26146**

[Overview](#overview) ·
[Features](#features) ·
[Architecture](#architecture) ·
[ML Pipeline](#ml-pipeline) ·
[Installation](#installation) ·
[Dashboard](#dashboard)

</div>

---

## Overview

ChainSentinel is an AI-powered blockchain intelligence platform developed for **Smart India Hackathon 2026 – Problem Statement SIH26146**.

The system processes Bitcoin transaction and network metadata, constructs transaction graphs, extracts behavioural and graph-based features, and applies machine learning techniques to identify anomalous activity.

The resulting intelligence is presented through an interactive Streamlit dashboard that allows analysts to examine suspicious wallets, transaction patterns, network relationships, and anomaly scores.

### Core Pipeline

```text
Raw Blockchain Data
        |
        v
Data Processing
        |
        v
Transaction Graph
        |
        v
Feature Engineering
        |
        v
Anomaly Detection
        |
        v
Ranked Anomalies
        |
        v
Investigation Dashboard
```

---

## Problem Statement

Bitcoin's pseudonymous architecture presents challenges for investigators attempting to identify suspicious transaction behaviour.

Large-scale blockchain datasets contain millions of transactions, addresses, and relationships, making manual analysis difficult.

This project aims to develop an offline intelligence pipeline capable of:

- Detecting anomalous transaction behaviour
- Representing Bitcoin activity as a transaction graph
- Extracting behavioural and network-level features
- Applying machine learning for anomaly detection
- Ranking suspicious entities for further investigation
- Correlating blockchain and network observations
- Providing an interactive investigation interface

---

## Features

### Transaction Graph

Bitcoin activity is represented using the relationship:

```text
Address → Transaction → Address
```

This preserves the original multi-input and multi-output structure of Bitcoin transactions.

An additional address-level graph is derived for graph-based analysis.

### Anomaly Detection

The current pipeline uses **Isolation Forest** to identify observations whose behaviour differs significantly from the normal population.

The model operates without requiring labelled data during the initial anomaly detection stage.

### Graph-Based Features

The system extracts network characteristics including:

- Transaction degree
- PageRank
- Betweenness centrality
- Unique counterparties
- Transaction frequency
- Incoming transaction count
- Outgoing transaction count
- Bitcoin transaction volume
- Average transaction value

### Network Correlation

Blockchain observations can be correlated with network-level information to identify relationships between transaction activity and network behaviour.

```text
Blockchain Data              Network Data
       |                          |
       |                          |
       +------------+-------------+
                    |
                    v
             Correlated Features
                    |
                    v
             Anomaly Detection
```

### Investigation Dashboard

The Streamlit dashboard provides:

- Anomaly rankings
- Wallet-level analysis
- Transaction statistics
- Behavioural features
- Graph relationships
- Anomaly scores
- Investigation reports

---

## Architecture

```text
                         ┌──────────────────────┐
                         │     Input Data       │
                         │                      │
                         │ Blockchain           │
                         │ Network Events       │
                         │ IP Enrichment        │
                         └──────────┬───────────┘
                                    |
                                    v
                         ┌──────────────────────┐
                         │   Data Processing    │
                         │                      │
                         │ Cleaning             │
                         │ Parsing              │
                         │ Validation           │
                         └──────────┬───────────┘
                                    |
                                    v
                         ┌──────────────────────┐
                         │  Transaction Graph   │
                         │                      │
                         │ Address → TX →       │
                         │ Address              │
                         └──────────┬───────────┘
                                    |
                                    v
                    ┌──────────────────────────────┐
                    │     Feature Engineering     │
                    │                              │
                    │ Transaction Features        │
                    │ Behavioural Features         │
                    │ Graph Features               │
                    │ Network Features             │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │      Isolation Forest        │
                    │                              │
                    │     Anomaly Detection        │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │     Ranked Anomalies         │
                    │                              │
                    │ Wallets / Transactions       │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │    Investigation Dashboard  │
                    │          Streamlit           │
                    └──────────────────────────────┘
```

---

## ML Pipeline

The current machine learning pipeline follows:

```text
Dataset
   |
   v
Graph Construction
   |
   v
Feature Engineering
   |
   v
Feature Processing
   |
   v
Isolation Forest
   |
   v
Anomaly Score
   |
   v
Ranked Investigation Leads
```

### Feature Categories

| Category | Examples |
|---|---|
| Transaction | Input count, output count |
| Behavioural | Transaction frequency, velocity |
| Financial | BTC inflow, BTC outflow, transaction volume |
| Network | Unique counterparties |
| Graph | Degree, PageRank |
| Centrality | Betweenness centrality |
| Network Correlation | Correlated network observations |

---

## Dataset

The project works with blockchain and network-level datasets.

```text
data/
├── blockchain_transactions.csv
├── network_events.csv
├── ip_enrichment.csv
└── group_b_features.csv
```

### Blockchain Transactions

Transaction-level blockchain information used to construct the transaction graph and generate wallet-level features.

### Network Events

Network observations that can be correlated with blockchain activity.

### IP Enrichment

Additional information associated with network-level observations.

### Group B Features

Correlated features used during the feature engineering and analysis stages.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/CapNovaa/SIH_2026.git
cd SIH_2026
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Build the Transaction Graph

```bash
python src/build_graph.py
```

### Generate Features

```bash
python src/build_features.py
```

### Run Anomaly Detection

```bash
python src/detect_anomalies.py
```

### Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

Generated files are stored in:

```text
outputs/
```

---

## Project Structure

```text
SIH_2026/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── blockchain_transactions.csv
│   ├── network_events.csv
│   ├── ip_enrichment.csv
│   └── group_b_features.csv
│
├── src/
│   ├── build_graph.py
│   ├── build_features.py
│   └── detect_anomalies.py
│
├── outputs/
│   └── generated analysis files
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Transaction Representation

Bitcoin transactions can contain multiple inputs and multiple outputs.

Rather than creating an artificial direct mapping between inputs and outputs, the system maintains the transaction as an intermediate entity:

```text
Address
   |
   v
Transaction
   |
   v
Address
```

This preserves the actual structure of the transaction.

Coinbase transactions are also represented in the transaction graph, while standard input-address relationships are not created for their coinbase inputs.

---

## Investigation Workflow

```text
Load Dataset
     |
     v
Build Transaction Graph
     |
     v
Generate Features
     |
     v
Detect Anomalies
     |
     v
Rank Anomalous Entities
     |
     v
Select Wallet
     |
     v
Investigate Behaviour
     |
     v
Analyse Transactions
     |
     v
Explore Graph Relationships
```

The system is intended to identify unusual activity and provide investigative leads. An anomaly score alone does not establish malicious or illegal behaviour.

---

## Dashboard

The Streamlit dashboard provides an interface for moving from high-level anomaly detection to individual wallet investigation.

Suggested screenshots:

```markdown
![Dashboard Overview](assets/dashboard.png)

![Wallet Investigation](assets/investigation.png)

![Transaction Graph](assets/transaction_graph.png)
```

For a polished repository, screenshots should be placed inside an `assets/` directory.

---

## Model Evaluation

Ground-truth labels are kept separate from the model features to avoid target leakage.

When labelled data is available, the detection pipeline can be evaluated using:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix

The evaluation should consider the highly imbalanced nature of anomaly detection datasets rather than relying solely on accuracy.

---

## Why Isolation Forest?

Isolation Forest provides an unsupervised baseline for identifying unusual observations.

Instead of learning a conventional classification boundary, the algorithm attempts to isolate observations that differ from the majority of the dataset.

This makes it suitable as an initial anomaly detection method when reliable labels are limited.

The architecture can later be extended with graph-based and temporal models.

---

## Roadmap

### Current

- [x] Bitcoin transaction graph
- [x] Address-level feature engineering
- [x] Transaction-level features
- [x] Graph centrality features
- [x] Isolation Forest anomaly detection
- [x] Anomaly ranking
- [x] Streamlit dashboard

### Planned

- [ ] Improved anomaly scoring
- [ ] Advanced graph visualization
- [ ] Model explainability
- [ ] Ground-truth based evaluation
- [ ] Temporal anomaly detection
- [ ] Community detection
- [ ] Multi-hop wallet investigation
- [ ] Entity clustering
- [ ] Real-time analysis pipeline
- [ ] Automated investigation reports

---

## Smart India Hackathon 2026

**Problem Statement:** SIH26146

**Domain:** Blockchain and Cybersecurity

**Objective:** AI-powered monitoring and analysis of Bitcoin transaction traffic.

The project focuses on processing blockchain and network metadata, correlating observations, detecting anomalous activity using AI/ML techniques, and presenting the resulting intelligence through an investigation interface.

---

## Contributing

Contributions and suggestions are welcome.

```bash
git checkout -b feature/new-feature
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature
```

Create a pull request after pushing the branch.

---

## License

This project is developed for educational, research, and hackathon purposes.

---

<div align="center">

**ChainSentinel**

AI-powered analysis of Bitcoin transaction activity.

**Smart India Hackathon 2026**

</div>
'''

path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(f"Created: {path}")
