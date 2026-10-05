import pandas as pd
import numpy as np
df = pd.read_csv("lightsched_labeled_dataset_v2.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


print("\n" + "=" * 60)
print("BEST SCHEDULER DISTRIBUTION")
print("=" * 60)

print(
    df["best_scheduler"].value_counts()
)

print("\nPercentages:")

print(
    df["best_scheduler"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\n" + "=" * 60)
print("SCHEDULER SCORE STATISTICS")
print("=" * 60)

score_columns = [
    "RR_score",
    "SJF_score",
    "Priority_score",
    "MLFQ_score"
]

print(
    df[score_columns].describe()
)


print("\n" + "=" * 60)
print("BEST SCORE STATISTICS")
print("=" * 60)

df["best_score"] = df[score_columns].max(axis=1)

print(
    df["best_score"].describe()
)


print("\n" + "=" * 60)
print("HOW CLOSE WERE THE TOP TWO SCHEDULERS?")
print("=" * 60)

scores = df[score_columns].values

sorted_scores = np.sort(scores, axis=1)[:, ::-1]

margin = sorted_scores[:, 0] - sorted_scores[:, 1]

df["score_margin"] = margin

print("\n" + "=" * 60)
print("EXACT TIES")
print("=" * 60)

tie_count = (df["score_margin"] == 0).sum()

print("Number of exact ties:", tie_count)

print(
    "Percentage of exact ties:",
    round(tie_count / len(df) * 100, 2),
    "%"
)


print("\nTie workloads:")

tie_df = df[df["score_margin"] == 0]

print(
    tie_df[
        [
            "workload_id",
            "RR_score",
            "SJF_score",
            "Priority_score",
            "MLFQ_score",
            "best_scheduler"
        ]
    ].head(20)
)

print(df["score_margin"].describe())


print("\nSmallest margins:")

print(
    df[
        [
            "workload_id",
            "best_scheduler",
            "best_score",
            "score_margin"
        ]
    ]
    .sort_values("score_margin")
    .head(20)
)


print("\n" + "=" * 60)
print("AVERAGE SCORE BY WINNER")
print("=" * 60)

print(
    df.groupby("best_scheduler")[score_columns]
    .mean()
    .round(4)
)