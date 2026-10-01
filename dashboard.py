import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

PREDICTIONS = "outputs/anomaly_scores.csv"
GROUND_TRUTH = "data/wallet_ground_truth.csv"


# ============================================================
# 2. LOAD FILES
# ============================================================

pred = pd.read_csv(PREDICTIONS)
truth = pd.read_csv(GROUND_TRUTH)

print("=" * 70)
print("              SIH-2026 MODEL ACCURACY EVALUATION")
print("=" * 70)

print(f"\nPrediction rows   : {len(pred):,}")
print(f"Ground truth rows : {len(truth):,}")


# ============================================================
# 3. DISPLAY COLUMNS
# ============================================================

print("\nPrediction columns:")
print(pred.columns.tolist())

print("\nGround truth columns:")
print(truth.columns.tolist())


# ============================================================
# 4. COLUMN NAMES
# ============================================================

# Model output
MODEL_ADDRESS = "address"
MODEL_LABEL = "anomaly_label"
MODEL_SCORE = "anomaly_score"

# Ground truth
GROUND_TRUTH_ADDRESS = "wallet_id"
GROUND_TRUTH_LABEL = "is_anomaly"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_prediction_columns = [
    MODEL_ADDRESS,
    MODEL_LABEL,
    MODEL_SCORE
]

required_ground_truth_columns = [
    GROUND_TRUTH_ADDRESS,
    GROUND_TRUTH_LABEL
]

for column in required_prediction_columns:

    if column not in pred.columns:

        raise ValueError(
            f"\nERROR: '{column}' not found in prediction file."
        )


for column in required_ground_truth_columns:

    if column not in truth.columns:

        raise ValueError(
            f"\nERROR: '{column}' not found in ground truth file."
        )


# ============================================================
# 6. CONVERT ISOLATION FOREST LABEL
# ============================================================

# Isolation Forest returns:
#
#     1  = NORMAL
#    -1  = ANOMALY
#
# Convert this into:
#
#     0  = NORMAL
#     1  = ANOMALY

pred["model_prediction"] = (
    pred[MODEL_LABEL] == -1
).astype(int)


# ============================================================
# 7. PREPARE GROUND TRUTH
# ============================================================

truth["actual"] = (
    pd.to_numeric(
        truth[GROUND_TRUTH_LABEL],
        errors="coerce"
    )
)

# Remove rows where the ground-truth label is invalid.
truth = truth.dropna(
    subset=["actual"]
).copy()

truth["actual"] = truth["actual"].astype(int)


# Rename wallet_id -> address so that both files
# can be matched using the same column.

truth = truth.rename(
    columns={
        GROUND_TRUTH_ADDRESS: MODEL_ADDRESS
    }
)


# ============================================================
# 8. CLEAN ADDRESS VALUES
# ============================================================

pred[MODEL_ADDRESS] = (
    pred[MODEL_ADDRESS]
    .astype(str)
    .str.strip()
)

truth[MODEL_ADDRESS] = (
    truth[MODEL_ADDRESS]
    .astype(str)
    .str.strip()
)


# ============================================================
# 9. MERGE PREDICTIONS WITH GROUND TRUTH
# ============================================================

evaluation = pred[
    [
        MODEL_ADDRESS,
        "model_prediction",
        MODEL_SCORE
    ]
].merge(
    truth[
        [
            MODEL_ADDRESS,
            "actual"
        ]
    ],
    on=MODEL_ADDRESS,
    how="inner"
)


# ============================================================
# 10. DATASET MATCHING
# ============================================================

print("\n" + "-" * 70)
print("DATASET MATCHING")
print("-" * 70)

print(
    f"Prediction wallets       : {len(pred):,}"
)

print(
    f"Ground-truth wallets     : {len(truth):,}"
)

print(
    f"Matched wallets          : {len(evaluation):,}"
)

print(
    f"Unmatched predictions    : "
    f"{len(pred) - len(evaluation):,}"
)

print(
    f"Unmatched ground truth   : "
    f"{len(truth) - len(evaluation):,}"
)


if len(evaluation) == 0:

    print("\nERROR: No wallet addresses matched.")

    print(
        "\nExample prediction addresses:"
    )

    print(
        pred[MODEL_ADDRESS].head(5).to_string(
            index=False
        )
    )

    print(
        "\nExample ground-truth wallet IDs:"
    )

    print(
        truth[MODEL_ADDRESS].head(5).to_string(
            index=False
        )
    )

    raise SystemExit


# ============================================================
# 11. ACTUAL ANOMALIES
# ============================================================

actual = evaluation["actual"].astype(int)

predicted = (
    evaluation["model_prediction"]
    .astype(int)
)

scores = evaluation[MODEL_SCORE]


actual_anomalies = int(
    (actual == 1).sum()
)

actual_normal = int(
    (actual == 0).sum()
)

predicted_anomalies = int(
    (predicted == 1).sum()
)

predicted_normal = int(
    (predicted == 0).sum()
)


# ============================================================
# 12. ANOMALY COUNTS
# ============================================================

print("\n" + "=" * 70)
print("                    ANOMALY COUNTS")
print("=" * 70)

print(
    f"\nTotal labelled wallets : "
    f"{len(evaluation):,}"
)

print(
    f"Actual anomalies       : "
    f"{actual_anomalies:,}"
)

print(
    f"Actual normal          : "
    f"{actual_normal:,}"
)

print(
    f"\nModel predicted anomalies : "
    f"{predicted_anomalies:,}"
)

print(
    f"Model predicted normal    : "
    f"{predicted_normal:,}"
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    actual,
    predicted,
    labels=[0, 1]
).ravel()


print("\n" + "=" * 70)
print("                    CONFUSION MATRIX")
print("=" * 70)

print(
    "\n                         ACTUAL"
)

print(
    "                    Normal    Anomaly"
)

print(
    f"Predicted Normal   {tn:8d}  {fn:9d}"
)

print(
    f"Predicted Anomaly  {fp:8d}  {tp:9d}"
)


# ============================================================
# 14. METRICS
# ============================================================

accuracy = accuracy_score(
    actual,
    predicted
)

precision = precision_score(
    actual,
    predicted,
    zero_division=0
)

recall = recall_score(
    actual,
    predicted,
    zero_division=0
)

f1 = f1_score(
    actual,
    predicted,
    zero_division=0
)


print("\n" + "=" * 70)
print("                    MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.4%}"
)

print(
    f"Precision : {precision:.4%}"
)

print(
    f"Recall    : {recall:.4%}"
)

print(
    f"F1 Score  : {f1:.4%}"
)


# ============================================================
# 15. ROC-AUC
# ============================================================

if actual.nunique() == 2:

    roc_auc = roc_auc_score(
        actual,
        scores
    )

    avg_precision = average_precision_score(
        actual,
        scores
    )

    print(
        f"ROC-AUC   : {roc_auc:.4%}"
    )

    print(
        f"Avg Prec. : {avg_precision:.4%}"
    )

else:

    print(
        "\nROC-AUC cannot be calculated "
        "because only one class exists."
    )


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("                 CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        actual,
        predicted,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. TRUE POSITIVES
# ============================================================

true_positives = evaluation[
    (evaluation["actual"] == 1) &
    (evaluation["model_prediction"] == 1)
].sort_values(
    MODEL_SCORE,
    ascending=False
)


print("\n" + "=" * 70)
print("                 TRUE POSITIVES")
print("=" * 70)

print(
    f"\nCorrectly detected anomalies: "
    f"{len(true_positives)}"
)

if len(true_positives) > 0:

    print(
        true_positives[
            [
                MODEL_ADDRESS,
                MODEL_SCORE
            ]
        ].head(20).to_string(
            index=False
        )
    )


# ============================================================
# 18. FALSE POSITIVES
# ============================================================

false_positives = evaluation[
    (evaluation["actual"] == 0) &
    (evaluation["model_prediction"] == 1)
].sort_values(
    MODEL_SCORE,
    ascending=False
)


print("\n" + "=" * 70)
print("                 FALSE POSITIVES")
print("=" * 70)

print(
    f"\nNormal wallets incorrectly flagged: "
    f"{len(false_positives)}"
)

if len(false_positives) > 0:

    print(
        false_positives[
            [
                MODEL_ADDRESS,
                MODEL_SCORE
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 19. FALSE NEGATIVES
# ============================================================

false_negatives = evaluation[
    (evaluation["actual"] == 1) &
    (evaluation["model_prediction"] == 0)
].sort_values(
    MODEL_SCORE,
    ascending=False
)


print("\n" + "=" * 70)
print("                 FALSE NEGATIVES")
print("=" * 70)

print(
    f"\nAnomalies missed by model: "
    f"{len(false_negatives)}"
)

if len(false_negatives) > 0:

    print(
        false_negatives[
            [
                MODEL_ADDRESS,
                MODEL_SCORE
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 20. ADD RESULT CATEGORY
# ============================================================

evaluation["result"] = np.select(
    [
        (evaluation["actual"] == 1) &
        (evaluation["model_prediction"] == 1),

        (evaluation["actual"] == 0) &
        (evaluation["model_prediction"] == 1),

        (evaluation["actual"] == 1) &
        (evaluation["model_prediction"] == 0),

        (evaluation["actual"] == 0) &
        (evaluation["model_prediction"] == 0)
    ],
    [
        "True Positive",
        "False Positive",
        "False Negative",
        "True Negative"
    ],
    default="Unknown"
)


# ============================================================
# 21. SAVE EVALUATION
# ============================================================

output_file = "outputs/model_evaluation.csv"

evaluation.to_csv(
    output_file,
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("                       FINAL SUMMARY")
print("=" * 70)

print(
    f"""
Total labelled wallets : {len(evaluation):,}

Actual anomalies       : {actual_anomalies:,}
Detected anomalies     : {predicted_anomalies:,}

True Positives         : {tp:,}
True Negatives         : {tn:,}
False Positives        : {fp:,}
False Negatives        : {fn:,}

Accuracy               : {accuracy:.2%}
Precision              : {precision:.2%}
Recall                 : {recall:.2%}
F1 Score               : {f1:.2%}
"""
)

if actual.nunique() == 2:

    print(
        f"ROC-AUC                : "
        f"{roc_auc:.2%}"
    )

    print(
        f"Average Precision      : "
        f"{avg_precision:.2%}"
    )


print(
    "\nDetailed wallet-level results:"
)

print(
    output_file
)

print("=" * 70)