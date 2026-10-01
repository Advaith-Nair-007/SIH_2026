import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"
EDA = OUT / "eda"
EDA.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(OUT / "address_features.csv")
pred = pd.read_csv(OUT / "anomaly_scores.csv")
truth = pd.read_csv(DATA / "wallet_ground_truth_V3.csv")

# Your real ground-truth schema
truth = truth[["wallet_id", "is_anomaly"]].rename(
    columns={"wallet_id": "address", "is_anomaly": "actual"}
)
truth["actual"] = pd.to_numeric(truth["actual"], errors="coerce")
truth = truth.dropna(subset=["actual"])
truth["actual"] = truth["actual"].astype(int)

df = features.merge(truth, on="address", how="inner")
pred["model_prediction"] = (pred["anomaly_label"] == -1).astype(int)

df = df.merge(
    pred[["address", "model_prediction", "anomaly_score",
          "anomaly_percentile", "risk_priority"]],
    on="address", how="left"
)

numeric = [
    c for c in features.columns
    if c != "address" and pd.api.types.is_numeric_dtype(features[c])
]

print("=" * 70)
print("              SIH-2026 MODEL-ORIENTED EDA")
print("=" * 70)
print(f"Feature rows       : {len(features):,}")
print(f"Labelled wallets   : {len(df):,}")
print(f"Actual anomalies   : {(df.actual == 1).sum():,}")

# 1. Data quality
quality = pd.DataFrame({
    "feature": numeric,
    "missing": [features[c].isna().sum() for c in numeric],
    "unique_values": [features[c].nunique() for c in numeric],
    "mean": [features[c].mean() for c in numeric],
    "median": [features[c].median() for c in numeric],
    "std": [features[c].std() for c in numeric],
    "min": [features[c].min() for c in numeric],
    "max": [features[c].max() for c in numeric],
})
quality.to_csv(EDA / "feature_quality.csv", index=False)

# 2. Correlation
X = features[numeric].replace([np.inf, -np.inf], 0).fillna(0)
corr = X.corr()

pairs = []
for i, a in enumerate(numeric):
    for b in numeric[i+1:]:
        r = abs(corr.loc[a, b])
        if r >= 0.85:
            pairs.append([a, b, r])

high_corr = pd.DataFrame(
    pairs,
    columns=["feature_1", "feature_2", "absolute_correlation"]
).sort_values("absolute_correlation", ascending=False)

high_corr.to_csv(EDA / "high_correlation_pairs.csv", index=False)

plt.figure(figsize=(14, 11))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(numeric)), numeric, rotation=90)
plt.yticks(range(len(numeric)), numeric)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(EDA / "correlation_heatmap.png", dpi=180)
plt.close()

# 3. Normal vs anomaly statistics
normal = df[df.actual == 0]
anomaly = df[df.actual == 1]

stats = []
for c in numeric:
    nmed = normal[c].median()
    amed = anomaly[c].median()
    stats.append({
        "feature": c,
        "normal_median": nmed,
        "anomaly_median": amed,
        "normal_mean": normal[c].mean(),
        "anomaly_mean": anomaly[c].mean(),
        "median_difference": abs(amed - nmed),
        "median_ratio": amed / nmed if nmed != 0 else np.nan
    })

stats = pd.DataFrame(stats).sort_values(
    "median_difference", ascending=False
)
stats.to_csv(EDA / "normal_vs_anomaly_features.csv", index=False)

# 4. Mutual information with ground truth
mi_features = [c for c in numeric if df[c].nunique() > 1]
mi = mutual_info_classif(
    df[mi_features].replace([np.inf, -np.inf], 0).fillna(0),
    df["actual"],
    random_state=42
)
mi_df = pd.DataFrame({
    "feature": mi_features,
    "mutual_information": mi
}).sort_values("mutual_information", ascending=False)
mi_df.to_csv(EDA / "mutual_information.csv", index=False)

plt.figure(figsize=(10, 7))
top = mi_df.head(15).sort_values("mutual_information")
plt.barh(top["feature"], top["mutual_information"])
plt.xlabel("Mutual Information")
plt.title("Top Features Associated With Ground Truth")
plt.tight_layout()
plt.savefig(EDA / "mutual_information_top15.png", dpi=180)
plt.close()

# 5. False-positive analysis
fp = df[(df.actual == 0) & (df.model_prediction == 1)]
fn = df[(df.actual == 1) & (df.model_prediction == 0)]
tp = df[(df.actual == 1) & (df.model_prediction == 1)]

print("\n" + "-" * 70)
print("FALSE POSITIVE / FALSE NEGATIVE ANALYSIS")
print("-" * 70)
print(f"True positives  : {len(tp):,}")
print(f"False positives : {len(fp):,}")
print(f"False negatives : {len(fn):,}")

fp_stats = []
for c in numeric:
    fp_stats.append({
        "feature": c,
        "normal_median": normal[c].median(),
        "false_positive_median": fp[c].median() if len(fp) else np.nan,
        "difference": abs(
            (fp[c].median() if len(fp) else np.nan) -
            normal[c].median()
        )
    })
pd.DataFrame(fp_stats).sort_values(
    "difference", ascending=False
).to_csv(EDA / "false_positive_feature_profile.csv", index=False)

df[df["model_prediction"].notna()].to_csv(
    EDA / "labelled_predictions_for_eda.csv", index=False
)

# 6. Print the most useful findings
print("\n" + "-" * 70)
print("TOP FEATURES BY MUTUAL INFORMATION")
print("-" * 70)
print(mi_df.head(15).to_string(index=False))

print("\n" + "-" * 70)
print("HIGHLY CORRELATED FEATURES")
print("-" * 70)
if len(high_corr):
    print(high_corr.head(20).to_string(index=False))
else:
    print("No pairs with |correlation| >= 0.85")

print("\n" + "-" * 70)
print("NORMAL VS ANOMALY")
print("-" * 70)
print(stats.head(15).to_string(index=False))

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
print(f"Results saved in: {EDA}")
print("\nMost important files:")
print("  mutual_information.csv")
print("  high_correlation_pairs.csv")
print("  normal_vs_anomaly_features.csv")
print("  false_positive_feature_profile.csv")
print("  correlation_heatmap.png")
print("  mutual_information_top15.png")
