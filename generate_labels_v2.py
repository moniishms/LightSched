import pandas as pd
import numpy as np

from schedulers_v2 import (
    round_robin,
    sjf,
    priority_scheduling,
    mlfq
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "lightsched_workload_dataset.csv"
OUTPUT_FILE = "lightsched_labeled_dataset_v2.csv"

# Weights for combined scheduler score
WEIGHTS = {
    "waiting": 0.25,
    "turnaround": 0.25,
    "response": 0.20,
    "throughput": 0.15,
    "cpu_utilization": 0.15
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Loaded dataset:")
print("Shape:", df.shape)
print("Number of workloads:", df["workload_id"].nunique())


# ============================================================
# SCHEDULERS
# ============================================================

schedulers = {
    "RR": round_robin,
    "SJF": sjf,
    "Priority": priority_scheduling,
    "MLFQ": mlfq
}


# ============================================================
# PROCESS WORKLOAD
# ============================================================

def convert_workload(group):

    processes = []

    for _, row in group.iterrows():

        processes.append({
            "pid": int(row["pid"]),
            "arrival_time": int(row["arrival_time"]),
            "burst_time": int(row["burst_time"]),
            "priority": int(row["priority"]),
            "io_frequency": int(row["io_frequency"])
        })

    return processes


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_lower_is_better(values):

    values = np.array(values, dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.ones(len(values))

    return (maximum - values) / (maximum - minimum)


def normalize_higher_is_better(values):

    values = np.array(values, dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.ones(len(values))

    return (values - minimum) / (maximum - minimum)


# ============================================================
# GENERATE LABELS
# ============================================================

all_results = []

workload_count = df["workload_id"].nunique()

for index, (workload_id, group) in enumerate(
    df.groupby("workload_id"),
    start=1
):

    processes = convert_workload(group)

    scheduler_metrics = {}

    # --------------------------------------------------------
    # Run all schedulers
    # --------------------------------------------------------

    for scheduler_name, scheduler_function in schedulers.items():

        _, summary = scheduler_function(processes)

        scheduler_metrics[scheduler_name] = summary

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    scheduler_names = list(schedulers.keys())

    waiting_values = [
        scheduler_metrics[s]["avg_waiting_time"]
        for s in scheduler_names
    ]

    turnaround_values = [
        scheduler_metrics[s]["avg_turnaround_time"]
        for s in scheduler_names
    ]

    response_values = [
        scheduler_metrics[s]["avg_response_time"]
        for s in scheduler_names
    ]

    throughput_values = [
        scheduler_metrics[s]["throughput"]
        for s in scheduler_names
    ]

    utilization_values = [
        scheduler_metrics[s]["cpu_utilization"]
        for s in scheduler_names
    ]

    # --------------------------------------------------------
    # Normalize metrics
    #
    # Lower waiting/turnaround/response = better
    # Higher throughput/utilization = better
    # --------------------------------------------------------

    waiting_score = normalize_lower_is_better(
        waiting_values
    )

    turnaround_score = normalize_lower_is_better(
        turnaround_values
    )

    response_score = normalize_lower_is_better(
        response_values
    )

    throughput_score = normalize_higher_is_better(
        throughput_values
    )

    utilization_score = normalize_higher_is_better(
        utilization_values
    )

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    combined_scores = {}

    for i, scheduler_name in enumerate(scheduler_names):

        score = (
            WEIGHTS["waiting"] * waiting_score[i]
            + WEIGHTS["turnaround"] * turnaround_score[i]
            + WEIGHTS["response"] * response_score[i]
            + WEIGHTS["throughput"] * throughput_score[i]
            + WEIGHTS["cpu_utilization"] * utilization_score[i]
        )

        combined_scores[scheduler_name] = score

    # --------------------------------------------------------
    # Select best scheduler
    # --------------------------------------------------------

    best_scheduler = max(
        combined_scores,
        key=combined_scores.get
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    result = {
        "workload_id": workload_id
    }

    for scheduler_name in scheduler_names:

        metrics = scheduler_metrics[scheduler_name]

        result[f"{scheduler_name}_waiting"] = (
            metrics["avg_waiting_time"]
        )

        result[f"{scheduler_name}_turnaround"] = (
            metrics["avg_turnaround_time"]
        )

        result[f"{scheduler_name}_response"] = (
            metrics["avg_response_time"]
        )

        result[f"{scheduler_name}_throughput"] = (
            metrics["throughput"]
        )

        result[f"{scheduler_name}_cpu_utilization"] = (
            metrics["cpu_utilization"]
        )

        result[f"{scheduler_name}_score"] = (
            combined_scores[scheduler_name]
        )

    result["best_scheduler"] = best_scheduler

    all_results.append(result)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 100 == 0 or index == workload_count:

        print(
            f"Processed {index}/{workload_count} workloads"
        )


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(all_results)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LABEL GENERATION COMPLETE")
print("=" * 60)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nShape:")
print(result_df.shape)

print("\nBest scheduler distribution:")
print(
    result_df["best_scheduler"]
    .value_counts()
)

print("\nBest scheduler percentage:")
print(
    result_df["best_scheduler"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nAverage scores:")

score_columns = [
    "RR_score",
    "SJF_score",
    "Priority_score",
    "MLFQ_score"
]

print(
    result_df[score_columns]
    .mean()
    .sort_values(ascending=False)
)

print("\nFirst 5 rows:")
print(result_df.head())