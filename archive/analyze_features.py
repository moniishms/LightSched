import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ============================================================
# LOAD DATA
# ============================================================

FEATURE_FILE = "lightsched_ml_features.csv"
LABEL_FILE = "lightsched_labeled_dataset_v2.csv"

features_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)

df = pd.merge(
    features_df,
    labels_df[["workload_id", "best_scheduler"]],
    on="workload_id",
    how="inner"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "num_processes",
    "avg_arrival_time",
    "arrival_time_variance",
    "avg_burst_time",
    "burst_time_variance",
    "avg_priority",
    "avg_io_frequency",
    "arrival_rate"
]


# ============================================================
# BASIC STATISTICS BY SCHEDULER
# ============================================================

print("=" * 70)
print("FEATURE STATISTICS BY BEST SCHEDULER")
print("=" * 70)

group_stats = df.groupby("best_scheduler")[FEATURE_COLUMNS].mean()

print(
    group_stats.round(3).to_string()
)


# ============================================================
# MEDIAN BY SCHEDULER
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MEDIANS BY BEST SCHEDULER")
print("=" * 70)

group_medians = df.groupby("best_scheduler")[FEATURE_COLUMNS].median()

print(
    group_medians.round(3).to_string()
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

X = df[FEATURE_COLUMNS]

encoder = LabelEncoder()

y = encoder.fit_transform(
    df["best_scheduler"]
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X, y)

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(
    importance_df.round(4).to_string(index=False)
)


# ============================================================
# FEATURE CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CORRELATION WITH ENCODED TARGET")
print("=" * 70)

temp_df = df[FEATURE_COLUMNS].copy()

temp_df["target"] = y

correlations = (
    temp_df.corr()["target"]
    .drop("target")
    .sort_values(
        key=abs,
        ascending=False
    )
)

print(
    correlations.round(4).to_string()
)


# ============================================================
# RANGE OF EACH FEATURE BY CLASS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE RANGE BY SCHEDULER")
print("=" * 70)

for scheduler in ["RR", "SJF", "Priority", "MLFQ"]:

    subset = df[
        df["best_scheduler"] == scheduler
    ]

    print("\n" + scheduler)
    print("-" * 40)

    for feature in FEATURE_COLUMNS:

        minimum = subset[feature].min()
        maximum = subset[feature].max()

        print(
            f"{feature:25s}: "
            f"{minimum:.3f} → {maximum:.3f}"
        )