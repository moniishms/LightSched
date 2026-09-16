"""
test_priority_variance_feature.py

============================================================
WHY THIS SCRIPT EXISTS
============================================================
analyze_regret_by_class.py found that Priority-true workloads get
25% accuracy and 16.78% average regret from the tuned RF -- more
than double the overall average -- and account for 21% of total
regret while being only ~10% of the population. The mechanism:
avg_priority is a MEAN, so a workload where roughly half the
processes have priority 1-2 and half have priority 9-10 (exactly
what generate_priority_heavy() in workload_generation.py creates)
can average out to a middle value indistinguishable from a workload
where every process has moderate priority. The model has never been
given a feature that can see that split.

This script adds ONE new feature -- priority_variance (population
variance of per-process priority within a workload) -- and tests,
head to head, whether it actually improves Priority-class
performance, using the SAME 500-workload unseen set and SAME fixed
RF hyperparameters for both the 8-feature and 9-feature model, so
any difference is attributable to the new feature and not to
something else changing.

============================================================
METHODOLOGY NOTE
============================================================
Both models here use the fixed hyperparameters from tune_models.py
(n_estimators=200, max_depth=12, min_samples_leaf=5,
class_weight='balanced') rather than re-tuning for each feature set.
This is a deliberate simplification for a fast, fair A/B comparison:
re-tuning separately for each feature set would let hyperparameter
search itself explain part of any difference, muddying the question
"does this one feature help." If the 9-feature version wins here, a
full re-tune on top of it would only be expected to help further,
not to change the qualitative conclusion.
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from schedulers_v2 import (
    round_robin,
    sjf,
    priority_scheduling,
    mlfq,
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RAW_FILE = "lightsched_workload_dataset.csv"
TRAIN_LABEL_FILE = "lightsched_labeled_dataset_v2.csv"

UNSEEN_RAW_FILE = "lightsched_unseen_workload_dataset.csv"

OUTPUT_FILE = "priority_variance_feature_test_results.csv"

ORIGINAL_FEATURES = [
    "num_processes",
    "avg_arrival_time",
    "arrival_time_variance",
    "avg_burst_time",
    "burst_time_variance",
    "avg_priority",
    "avg_io_frequency",
    "arrival_rate",
]

NEW_FEATURE = "priority_variance"

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
)

WEIGHTS = {
    "waiting": 0.25,
    "turnaround": 0.25,
    "response": 0.20,
    "throughput": 0.15,
    "cpu_utilization": 0.15,
}

SCHEDULERS = {
    "RR": round_robin,
    "SJF": sjf,
    "Priority": priority_scheduling,
    "MLFQ": mlfq,
}

CLASS_LABELS = ["RR", "SJF", "Priority", "MLFQ"]


# ============================================================
# FEATURE EXTRACTION (original 8 + new priority_variance)
# ============================================================

def extract_features_v3(group):

    num_processes = len(group)

    avg_arrival_time = group["arrival_time"].mean()
    arrival_time_variance = group["arrival_time"].var(ddof=0)

    avg_burst_time = group["burst_time"].mean()
    burst_time_variance = group["burst_time"].var(ddof=0)

    avg_priority = group["priority"].mean()
    priority_variance = group["priority"].var(ddof=0)

    avg_io_frequency = group["io_frequency"].mean()

    arrival_span = (
        group["arrival_time"].max() - group["arrival_time"].min() + 1
    )
    arrival_rate = num_processes / arrival_span

    return pd.Series(
        {
            "num_processes": num_processes,
            "avg_arrival_time": avg_arrival_time,
            "arrival_time_variance": arrival_time_variance,
            "avg_burst_time": avg_burst_time,
            "burst_time_variance": burst_time_variance,
            "avg_priority": avg_priority,
            "priority_variance": priority_variance,
            "avg_io_frequency": avg_io_frequency,
            "arrival_rate": arrival_rate,
        }
    )


def build_v3_features(raw_file):

    df = pd.read_csv(raw_file)

    features = (
        df.groupby("workload_id")
        .apply(extract_features_v3, include_groups=False)
        .reset_index()
    )

    return features.fillna(0.0)


# ============================================================
# TRAINING SET (v3 features + existing V2 labels)
# ============================================================

def build_training_set():

    print("Building v3 features from raw training data (this groups "
          "~87k process rows into 5000 workloads, may take a moment)...")

    features_v3 = build_v3_features(TRAIN_RAW_FILE)

    labels_df = pd.read_csv(TRAIN_LABEL_FILE)

    train_df = pd.merge(
        features_v3,
        labels_df[["workload_id", "best_scheduler"]],
        on="workload_id",
        how="inner",
    )

    print(f"Training set shape (with priority_variance): {train_df.shape}")

    return train_df


# ============================================================
# UNSEEN SET GROUND TRUTH (mirrors evaluate_unseen.py)
# ============================================================

def normalize_lower_is_better(values):
    values = np.array(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.ones(len(values))
    return (hi - values) / (hi - lo)


def normalize_higher_is_better(values):
    values = np.array(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.ones(len(values))
    return (values - lo) / (hi - lo)


def convert_workload(group):
    processes = []
    for _, row in group.iterrows():
        processes.append(
            {
                "pid": int(row["pid"]),
                "arrival_time": int(row["arrival_time"]),
                "burst_time": int(row["burst_time"]),
                "priority": int(row["priority"]),
                "io_frequency": int(row["io_frequency"]),
            }
        )
    return processes


def calculate_best_scheduler(results):

    names = list(SCHEDULERS.keys())

    waiting = [results[s][1]["avg_waiting_time"] for s in names]
    turnaround = [results[s][1]["avg_turnaround_time"] for s in names]
    response = [results[s][1]["avg_response_time"] for s in names]
    throughput = [results[s][1]["throughput"] for s in names]
    utilization = [results[s][1]["cpu_utilization"] for s in names]

    scores = {}
    for i, name in enumerate(names):
        scores[name] = (
            WEIGHTS["waiting"] * normalize_lower_is_better(waiting)[i]
            + WEIGHTS["turnaround"] * normalize_lower_is_better(turnaround)[i]
            + WEIGHTS["response"] * normalize_lower_is_better(response)[i]
            + WEIGHTS["throughput"] * normalize_higher_is_better(throughput)[i]
            + WEIGHTS["cpu_utilization"] * normalize_higher_is_better(utilization)[i]
        )

    best = max(scores, key=scores.get)
    return best, scores


def build_unseen_ground_truth():

    print("\nBuilding v3 features + running simulator for ground truth "
          "on the 500-workload unseen set...")

    raw_df = pd.read_csv(UNSEEN_RAW_FILE)

    unseen_features = build_v3_features(UNSEEN_RAW_FILE)

    rows = []

    workload_groups = raw_df.groupby("workload_id")
    total = len(workload_groups)

    for index, (workload_id, group) in enumerate(workload_groups, start=1):

        processes = convert_workload(group)

        scheduler_results = {}
        for name, fn in SCHEDULERS.items():
            scheduler_results[name] = fn(processes)

        actual_best, scores = calculate_best_scheduler(scheduler_results)

        row = {"workload_id": workload_id, "actual_best_scheduler": actual_best}
        for name, score in scores.items():
            row[f"{name}_score"] = score

        rows.append(row)

        if index % 100 == 0 or index == total:
            print(f"  simulated {index}/{total} unseen workloads")

    ground_truth_df = pd.DataFrame(rows)

    unseen_df = pd.merge(
        unseen_features, ground_truth_df, on="workload_id", how="inner"
    )

    return unseen_df


# ============================================================
# TRAIN + EVALUATE ONE VARIANT
# ============================================================

def train_and_evaluate(train_df, unseen_df, feature_columns, label):

    X_train = train_df[feature_columns]
    y_train = train_df["best_scheduler"]

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    X_unseen = unseen_df[feature_columns]
    y_true = unseen_df["actual_best_scheduler"]

    y_pred = model.predict(X_unseen)

    result_df = unseen_df[["workload_id", "actual_best_scheduler"]].copy()
    result_df["predicted_scheduler"] = y_pred
    result_df["variant"] = label

    # Vectorized score lookup: build a workload_id -> {scheduler: score}
    # dict once, then look up both the predicted and oracle score per row.
    # (Avoids an O(n^2) apply-with-boolean-mask pattern.)
    score_cols = {s: f"{s}_score" for s in SCHEDULERS}
    scores_by_workload = unseen_df.set_index("workload_id")[
        list(score_cols.values())
    ].to_dict(orient="index")

    def lookup_score(workload_id, scheduler_name):
        return scores_by_workload[workload_id][score_cols[scheduler_name]]

    result_df["lightsched_score"] = [
        lookup_score(wid, pred)
        for wid, pred in zip(result_df["workload_id"], result_df["predicted_scheduler"])
    ]
    result_df["oracle_score"] = [
        lookup_score(wid, actual)
        for wid, actual in zip(result_df["workload_id"], result_df["actual_best_scheduler"])
    ]
    result_df["regret_pct"] = np.where(
        result_df["oracle_score"] != 0,
        (result_df["oracle_score"] - result_df["lightsched_score"])
        / result_df["oracle_score"]
        * 100.0,
        0.0,
    )
    result_df["correct"] = y_true.values == y_pred

    overall = {
        "variant": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "avg_regret_pct": result_df["regret_pct"].mean(),
    }

    priority_mask = y_true == "Priority"
    priority_subset = result_df[priority_mask.values]

    priority_stats = {
        "variant": label,
        "n_priority_workloads": int(priority_mask.sum()),
        "priority_accuracy": float(priority_subset["correct"].mean()),
        "priority_avg_regret_pct": float(priority_subset["regret_pct"].mean()),
    }

    return model, result_df, overall, priority_stats


# ============================================================
# MAIN
# ============================================================

def main():

    train_df = build_training_set()
    unseen_df = build_unseen_ground_truth()

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING BOTH VARIANTS")
    print("=" * 70)

    print("\n[1/2] Original 8 features (no priority_variance)...")
    _, results_old, overall_old, priority_old = train_and_evaluate(
        train_df, unseen_df, ORIGINAL_FEATURES, "original_8_features"
    )

    print("[2/2] 9 features (+ priority_variance)...")
    new_features = ORIGINAL_FEATURES + [NEW_FEATURE]
    _, results_new, overall_new, priority_new = train_and_evaluate(
        train_df, unseen_df, new_features, "plus_priority_variance"
    )

    combined_results = pd.concat([results_old, results_new], ignore_index=True)
    combined_results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved per-workload results to: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)
    overall_df = pd.DataFrame([overall_old, overall_new]).set_index("variant")
    print(overall_df.round(4).to_string())

    print("\n" + "=" * 70)
    print("PRIORITY-CLASS COMPARISON (the actual hypothesis test)")
    print("=" * 70)
    priority_df = pd.DataFrame([priority_old, priority_new]).set_index("variant")
    print(priority_df.round(4).to_string())

    acc_delta = (
        priority_new["priority_accuracy"] - priority_old["priority_accuracy"]
    )
    regret_delta = (
        priority_new["priority_avg_regret_pct"]
        - priority_old["priority_avg_regret_pct"]
    )

    print(f"\nPriority accuracy change   : {acc_delta:+.4f}")
    print(f"Priority avg regret change : {regret_delta:+.2f} pp")

    if acc_delta > 0.05 or regret_delta < -2.0:
        verdict = (
            "priority_variance appears to meaningfully help Priority-class "
            "prediction. Worth keeping as a permanent feature."
        )
    elif acc_delta < -0.02 or regret_delta > 2.0:
        verdict = (
            "priority_variance appears to HURT Priority-class prediction "
            "(possibly adding noise given the ~500-1500 training samples "
            "per class). The hypothesis was wrong as stated -- worth "
            "checking a fraction-of-extreme-priority feature instead."
        )
    else:
        verdict = (
            "priority_variance made little difference either way. The "
            "Priority weakness likely needs a different fix (e.g. a "
            "feature that directly counts low- vs high-priority "
            "processes, rather than a single variance summary)."
        )

    print(f"\nVERDICT: {verdict}")

    print("\n" + "=" * 70)
    print("FEATURE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()