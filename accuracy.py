import pandas as pd

truth = pd.read_csv("data/ground_truth.csv")

print("========== TARGET TYPES ==========")
print(truth["target_type"].value_counts())

print("\n========== IS ANOMALY ==========")
print(truth["is_anomaly"].value_counts())

print("\n========== TARGET ID SAMPLE ==========")
print(truth[[
    "target_type",
    "target_id",
    "is_anomaly",
    "severity",
    "scenario_id"
]].head(20).to_string(index=False))

print("\n========== UNIQUE TARGET IDs ==========")
print("Total:", truth["target_id"].nunique())

print("\n========== SEVERITIES ==========")
print(truth["severity"].value_counts())