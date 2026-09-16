import pandas as pd
import numpy as np

from schedulers import (
    round_robin,
    sjf,
    priority_scheduling,
    mlfq
)


# ============================================================
# 1. LOAD RAW WORKLOAD DATA
# ============================================================

df = pd.read_csv("lightsched_workload_dataset.csv")

print("Raw dataset shape:", df.shape)
print("Number of workloads:", df["workload_id"].nunique())


# ============================================================
# 2. DEFINE SCHEDULERS
# ============================================================

schedulers = {
    "Round Robin": round_robin,
    "SJF": sjf,
    "Priority": priority_scheduling,
    "MLFQ": mlfq
}


# ============================================================
# 3. WEIGHTS FOR COMBINED SCORE
# ============================================================

WEIGHTS = {
    "waiting_time": 0.25,
    "turnaround_time": 0.25,
    "response_time": 0.20,
    "throughput": 0.15,
    "cpu_utilization": 0.15
}


# ============================================================
# 4. NORMALIZATION FUNCTIONS
# ============================================================

def normalize_lower_is_better(values):
    """
    Lower value = better score.
    """
    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return [1.0] * len(values)

    return [
        (max_value - value) / (max_value - min_value)
        for value in values
    ]


def normalize_higher_is_better(values):
    """
    Higher value = better score.
    """
    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return [1.0] * len(values)

    return [
        (value - min_value) / (max_value - min_value)
        for value in values
    ]


# ============================================================
# 5. PROCESS EACH WORKLOAD
# ============================================================

all_results = []

workload_ids = df["workload_id"].unique()

total_workloads = len(workload_ids)

for index, workload_id in enumerate(workload_ids, start=1):

    workload_df = df[df["workload_id"] == workload_id]

    # Convert dataframe rows into process dictionaries
    processes = []

    for _, row in workload_df.iterrows():

        process = {
            "pid": int(row["pid"]),
            "arrival_time": int(row["arrival_time"]),
            "burst_time": int(row["burst_time"]),
            "priority": int(row["priority"])
        }

        processes.append(process)


    # --------------------------------------------------------
    # Run all four schedulers
    # --------------------------------------------------------

    scheduler_metrics = {}

    for scheduler_name, scheduler_function in schedulers.items():

        results, summary = scheduler_function(processes)

        scheduler_metrics[scheduler_name] = {
            "waiting_time": summary["avg_waiting_time"],
            "turnaround_time": summary["avg_turnaround_time"],
            "response_time": summary["avg_response_time"],
            "throughput": summary["throughput"],
            "cpu_utilization": summary["cpu_utilization"]
        }


    # --------------------------------------------------------
    # Extract values for normalization
    # --------------------------------------------------------

    waiting_values = [
        scheduler_metrics[name]["waiting_time"]
        for name in schedulers
    ]

    turnaround_values = [
        scheduler_metrics[name]["turnaround_time"]
        for name in schedulers
    ]

    response_values = [
        scheduler_metrics[name]["response_time"]
        for name in schedulers
    ]

    throughput_values = [
        scheduler_metrics[name]["throughput"]
        for name in schedulers
    ]

    utilization_values = [
        scheduler_metrics[name]["cpu_utilization"]
        for name in schedulers
    ]


    # --------------------------------------------------------
    # Normalize metrics
    # --------------------------------------------------------

    waiting_scores = normalize_lower_is_better(waiting_values)

    turnaround_scores = normalize_lower_is_better(
        turnaround_values
    )

    response_scores = normalize_lower_is_better(
        response_values
    )

    throughput_scores = normalize_higher_is_better(
        throughput_values
    )

    utilization_scores = normalize_higher_is_better(
        utilization_values
    )


    # --------------------------------------------------------
    # Calculate combined score
    # --------------------------------------------------------

    combined_scores = {}

    scheduler_names = list(schedulers.keys())

    for i, scheduler_name in enumerate(scheduler_names):

        score = (
            WEIGHTS["waiting_time"] * waiting_scores[i]
            + WEIGHTS["turnaround_time"] * turnaround_scores[i]
            + WEIGHTS["response_time"] * response_scores[i]
            + WEIGHTS["throughput"] * throughput_scores[i]
            + WEIGHTS["cpu_utilization"] * utilization_scores[i]
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

    row = {
        "workload_id": workload_id,

        # Round Robin
        "rr_waiting_time": scheduler_metrics["Round Robin"]["waiting_time"],
        "rr_turnaround_time": scheduler_metrics["Round Robin"]["turnaround_time"],
        "rr_response_time": scheduler_metrics["Round Robin"]["response_time"],
        "rr_throughput": scheduler_metrics["Round Robin"]["throughput"],
        "rr_cpu_utilization": scheduler_metrics["Round Robin"]["cpu_utilization"],

        # SJF
        "sjf_waiting_time": scheduler_metrics["SJF"]["waiting_time"],
        "sjf_turnaround_time": scheduler_metrics["SJF"]["turnaround_time"],
        "sjf_response_time": scheduler_metrics["SJF"]["response_time"],
        "sjf_throughput": scheduler_metrics["SJF"]["throughput"],
        "sjf_cpu_utilization": scheduler_metrics["SJF"]["cpu_utilization"],

        # Priority
        "priority_waiting_time": scheduler_metrics["Priority"]["waiting_time"],
        "priority_turnaround_time": scheduler_metrics["Priority"]["turnaround_time"],
        "priority_response_time": scheduler_metrics["Priority"]["response_time"],
        "priority_throughput": scheduler_metrics["Priority"]["throughput"],
        "priority_cpu_utilization": scheduler_metrics["Priority"]["cpu_utilization"],

        # MLFQ
        "mlfq_waiting_time": scheduler_metrics["MLFQ"]["waiting_time"],
        "mlfq_turnaround_time": scheduler_metrics["MLFQ"]["turnaround_time"],
        "mlfq_response_time": scheduler_metrics["MLFQ"]["response_time"],
        "mlfq_throughput": scheduler_metrics["MLFQ"]["throughput"],
        "mlfq_cpu_utilization": scheduler_metrics["MLFQ"]["cpu_utilization"],

        # Combined scores
        "rr_score": combined_scores["Round Robin"],
        "sjf_score": combined_scores["SJF"],
        "priority_score": combined_scores["Priority"],
        "mlfq_score": combined_scores["MLFQ"],

        # ML target
        "best_scheduler": best_scheduler
    }

    all_results.append(row)


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 100 == 0 or index == total_workloads:
        print(
            f"Processed {index}/{total_workloads} workloads"
        )


# ============================================================
# 6. CREATE FINAL DATASET
# ============================================================

results_df = pd.DataFrame(all_results)


# ============================================================
# 7. SAVE DATASET
# ============================================================

output_file = "lightsched_labeled_dataset.csv"

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 8. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LABEL GENERATION COMPLETE")
print("=" * 60)

print("\nFinal dataset shape:")
print(results_df.shape)

print("\nBest scheduler distribution:")
print(
    results_df["best_scheduler"].value_counts()
)

print("\nPercentage distribution:")
print(
    results_df["best_scheduler"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nSaved to:")
print(output_file)