"""
rule_based_baseline.py

Tests whether the tuned Random Forest is doing something a simple,
human-written decision rule couldn't do in a handful of lines.

============================================================
WHY THIS SCRIPT EXISTS
============================================================
SJF is the best scheduler for ~75% of workloads in the training
population. A model that mostly predicts SJF can look reasonable on
regret (picking SJF when SJF isn't optimal is often still a decent
pick) without doing any real workload-aware discrimination. This
script builds the simplest plausible alternative -- a four-branch
if/elif rule using the same 8 features the RF sees -- and evaluates
it on the exact same unseen workload set, ground truth, and metrics
as evaluate_unseen.py. If the RF doesn't clearly beat this, that's
an important, reportable finding, not a bug.

============================================================
HOW THE RULE WAS CONSTRUCTED
============================================================
Thresholds are quartiles (P25/P75) of each feature computed from the
TRAINING population (lightsched_ml_features.csv merged with the V2
labels) -- not hand-guessed numbers, and not computed from the
unseen set itself (that would leak the test set into the rule).
This mirrors what the RF had access to: the same population
statistics, just used by a human instead of a tree ensemble.

The rule, in order:
  1. Near-simultaneous arrivals + uniform burst lengths (both in the
     bottom quartile) -> RR. This is the classic "fairness" workload
     signature: nothing to gain by reordering, so round-robin is a
     natural fit.
  2. Burst-time variance in the top quartile -> MLFQ. Wide spread in
     job lengths is exactly what multilevel feedback's demotion
     mechanism is designed to exploit.
  3. Average priority in the bottom or top quartile (i.e. workload
     leans toward a priority extreme) -> Priority scheduling.
  4. Otherwise -> SJF (the default / majority fallback).

Known limitation, stated up front: avg_priority is a MEAN, so a
workload with a genuinely bimodal priority split (many very-low +
many very-high priority processes, as generate_priority_heavy()
in workload_generation.py creates) can still average out to a
middle value and be invisible to branch 3. The rule can't fix this
without a priority-variance feature, which isn't in the current
feature set -- worth noting as a possible future feature addition
regardless of how this comparison turns out.
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
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

TRAIN_FEATURE_FILE = "lightsched_ml_features.csv"
TRAIN_LABEL_FILE = "lightsched_labeled_dataset_v2.csv"

UNSEEN_FILE = "lightsched_unseen_workload_dataset.csv"

OUTPUT_FILE = "lightsched_rule_baseline_evaluation.csv"

FEATURE_COLUMNS = [
    "num_processes",
    "avg_arrival_time",
    "arrival_time_variance",
    "avg_burst_time",
    "burst_time_variance",
    "avg_priority",
    "avg_io_frequency",
    "arrival_rate",
]

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
# STEP 1: DERIVE RULE THRESHOLDS FROM THE TRAINING POPULATION
# ============================================================

def derive_thresholds():

    features_df = pd.read_csv(TRAIN_FEATURE_FILE)
    labels_df = pd.read_csv(TRAIN_LABEL_FILE)

    train_df = pd.merge(
        features_df,
        labels_df[["workload_id", "best_scheduler"]],
        on="workload_id",
        how="inner",
    )

    thresholds = {
        "arrival_var_p25": train_df["arrival_time_variance"].quantile(0.25),
        "burst_var_p25": train_df["burst_time_variance"].quantile(0.25),
        "burst_var_p75": train_df["burst_time_variance"].quantile(0.75),
        "priority_p25": train_df["avg_priority"].quantile(0.25),
        "priority_p75": train_df["avg_priority"].quantile(0.75),
    }

    print("=" * 70)
    print("RULE THRESHOLDS (derived from training population quartiles)")
    print("=" * 70)

    for name, value in thresholds.items():
        print(f"{name:20s}: {value:.4f}")

    return thresholds


# ============================================================
# STEP 2: THE RULE ITSELF
# ============================================================

def rule_predict(row, t):

    low_arrival_var = row["arrival_time_variance"] <= t["arrival_var_p25"]
    low_burst_var = row["burst_time_variance"] <= t["burst_var_p25"]
    high_burst_var = row["burst_time_variance"] >= t["burst_var_p75"]
    extreme_priority = (
        row["avg_priority"] <= t["priority_p25"]
        or row["avg_priority"] >= t["priority_p75"]
    )

    if low_arrival_var and low_burst_var:
        return "RR"

    if high_burst_var:
        return "MLFQ"

    if extreme_priority:
        return "Priority"

    return "SJF"


# ============================================================
# STEP 3: FEATURE EXTRACTION + GROUND TRUTH (mirrors
# evaluate_unseen.py exactly, so results are comparable)
# ============================================================

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


def extract_features(group):

    num_processes = len(group)

    avg_arrival_time = group["arrival_time"].mean()
    arrival_time_variance = group["arrival_time"].var()

    avg_burst_time = group["burst_time"].mean()
    burst_time_variance = group["burst_time"].var()

    avg_priority = group["priority"].mean()

    avg_io_frequency = group["io_frequency"].mean()

    arrival_span = (
        group["arrival_time"].max() - group["arrival_time"].min()
    )

    if arrival_span == 0:
        arrival_rate = num_processes
    else:
        arrival_rate = num_processes / arrival_span

    return {
        "num_processes": num_processes,
        "avg_arrival_time": avg_arrival_time,
        "arrival_time_variance": arrival_time_variance,
        "avg_burst_time": avg_burst_time,
        "burst_time_variance": burst_time_variance,
        "avg_priority": avg_priority,
        "avg_io_frequency": avg_io_frequency,
        "arrival_rate": arrival_rate,
    }


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


def calculate_best_scheduler(results):

    names = list(SCHEDULERS.keys())

    waiting = [results[s][1]["avg_waiting_time"] for s in names]
    turnaround = [results[s][1]["avg_turnaround_time"] for s in names]
    response = [results[s][1]["avg_response_time"] for s in names]
    throughput = [results[s][1]["throughput"] for s in names]
    utilization = [results[s][1]["cpu_utilization"] for s in names]

    waiting_score = normalize_lower_is_better(waiting)
    turnaround_score = normalize_lower_is_better(turnaround)
    response_score = normalize_lower_is_better(response)
    throughput_score = normalize_higher_is_better(throughput)
    utilization_score = normalize_higher_is_better(utilization)

    scores = {}

    for i, name in enumerate(names):
        scores[name] = (
            WEIGHTS["waiting"] * waiting_score[i]
            + WEIGHTS["turnaround"] * turnaround_score[i]
            + WEIGHTS["response"] * response_score[i]
            + WEIGHTS["throughput"] * throughput_score[i]
            + WEIGHTS["cpu_utilization"] * utilization_score[i]
        )

    best = max(scores, key=scores.get)

    return best, scores


# ============================================================
# STEP 4: RUN THE RULE ON THE UNSEEN SET
# ============================================================

def run_evaluation(thresholds):

    df = pd.read_csv(UNSEEN_FILE)

    print("\n" + "=" * 70)
    print("RULE-BASED BASELINE: UNSEEN WORKLOAD EVALUATION")
    print("=" * 70)

    print(f"Loaded unseen dataset shape : {df.shape}")
    print(f"Number of workloads         : {df['workload_id'].nunique()}")

    results = []

    workload_groups = df.groupby("workload_id")
    total = len(workload_groups)

    for index, (workload_id, group) in enumerate(workload_groups, start=1):

        processes = convert_workload(group)
        feature_dict = extract_features(group)

        predicted_scheduler = rule_predict(feature_dict, thresholds)

        scheduler_results = {}
        for name, fn in SCHEDULERS.items():
            scheduler_results[name] = fn(processes)

        actual_best, scores = calculate_best_scheduler(scheduler_results)

        lightsched_score = scores[predicted_scheduler]
        oracle_score = scores[actual_best]
        score_gap = oracle_score - lightsched_score

        regret_pct = (
            (score_gap / oracle_score) * 100.0 if oracle_score != 0 else 0.0
        )

        results.append(
            {
                "workload_id": workload_id,
                "predicted_scheduler": predicted_scheduler,
                "actual_best_scheduler": actual_best,
                "prediction_correct": predicted_scheduler == actual_best,
                "lightsched_score": lightsched_score,
                "oracle_score": oracle_score,
                "score_gap": score_gap,
                "regret_pct": regret_pct,
            }
        )

        if index % 50 == 0 or index == total:
            print(f"Processed {index}/{total} workloads")

    return pd.DataFrame(results)


# ============================================================
# STEP 5: METRICS + REPORT
# ============================================================

def report(results_df):

    y_true = results_df["actual_best_scheduler"]
    y_pred = results_df["predicted_scheduler"]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    avg_regret = results_df["regret_pct"].mean()

    print("\n" + "=" * 70)
    print("RULE-BASED BASELINE RESULTS")
    print("=" * 70)

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}")
    print(f"Weighted F1        : {weighted_f1:.4f}")
    print(f"Average regret %   : {avg_regret:.2f}%")

    print(
        f"\nCorrect predictions: "
        f"{int(results_df['prediction_correct'].sum())}/{len(results_df)}"
    )

    print("\nPrediction distribution:")
    print(y_pred.value_counts().to_string())

    print("\nActual best-scheduler distribution:")
    print(y_true.value_counts().to_string())

    print("\nClassification report:")
    print(
        classification_report(
            y_true, y_pred, labels=CLASS_LABELS, zero_division=0
        )
    )

    print("Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {s}" for s in CLASS_LABELS],
        columns=[f"Pred {s}" for s in CLASS_LABELS],
    )
    print(cm_df)

    print("\n" + "=" * 70)
    print("REFERENCE: TUNED RF ON THE SAME UNSEEN SET (from prior run)")
    print("=" * 70)
    print("Accuracy           : 0.6740")
    print("Balanced Accuracy  : 0.5783")
    print("Macro F1           : 0.5033")
    print("Average regret %   : 8.28%")
    print(
        "\n(These RF numbers are restated from your existing "
        "lightsched_unseen_evaluation.csv / "
        "lightsched_performance_evaluation.csv runs, not "
        "recomputed here -- compare against the rule's numbers "
        "above directly.)"
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "avg_regret_pct": avg_regret,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    thresholds = derive_thresholds()

    results_df = run_evaluation(thresholds)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved per-workload results to: {OUTPUT_FILE}")

    report(results_df)

    print("\n" + "=" * 70)
    print("RULE-BASED BASELINE EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()