import pandas as pd
import numpy as np
import joblib

from schedulers_v2 import (
    round_robin,
    sjf,
    priority_scheduling,
    mlfq
)


# ============================================================
# SETTINGS
# ============================================================

WORKLOAD_FILE = "lightsched_unseen_workload_dataset.csv"

MODEL_FILE = "models/random_forest_tuned.joblib"

OUTPUT_FILE = "lightsched_performance_evaluation.csv"

WEIGHTS = {
    "waiting": 0.25,
    "turnaround": 0.25,
    "response": 0.20,
    "throughput": 0.15,
    "cpu_utilization": 0.15
}

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
# LOAD DATA
# ============================================================

df = pd.read_csv(WORKLOAD_FILE)

model = joblib.load(MODEL_FILE)

print("=" * 70)
print("LIGHTSCHED END-TO-END PERFORMANCE EVALUATION")
print("=" * 70)

print("\nWorkloads:", df["workload_id"].nunique())
print("Process records:", len(df))


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
# CONVERT WORKLOAD
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
# FEATURE EXTRACTION
# ============================================================

def extract_features(group):

    num_processes = len(group)

    avg_arrival_time = group["arrival_time"].mean()

    arrival_time_variance = group["arrival_time"].var()

    avg_burst_time = group["burst_time"].mean()

    burst_time_variance = group["burst_time"].var()

    avg_priority = group["priority"].mean()

    avg_io_frequency = group["io_frequency"].mean()

    arrival_span = (
        group["arrival_time"].max()
        - group["arrival_time"].min()
    )

    if arrival_span == 0:
        arrival_rate = num_processes
    else:
        arrival_rate = num_processes / arrival_span

    return pd.DataFrame([{
        "num_processes": num_processes,
        "avg_arrival_time": avg_arrival_time,
        "arrival_time_variance": arrival_time_variance,
        "avg_burst_time": avg_burst_time,
        "burst_time_variance": burst_time_variance,
        "avg_priority": avg_priority,
        "avg_io_frequency": avg_io_frequency,
        "arrival_rate": arrival_rate
    }])


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
# CALCULATE COMPOSITE SCORES
# ============================================================

def calculate_scores(scheduler_results):

    scheduler_names = list(schedulers.keys())

    waiting = [
        scheduler_results[s][1]["avg_waiting_time"]
        for s in scheduler_names
    ]

    turnaround = [
        scheduler_results[s][1]["avg_turnaround_time"]
        for s in scheduler_names
    ]

    response = [
        scheduler_results[s][1]["avg_response_time"]
        for s in scheduler_names
    ]

    throughput = [
        scheduler_results[s][1]["throughput"]
        for s in scheduler_names
    ]

    utilization = [
        scheduler_results[s][1]["cpu_utilization"]
        for s in scheduler_names
    ]

    waiting_score = normalize_lower_is_better(waiting)

    turnaround_score = normalize_lower_is_better(
        turnaround
    )

    response_score = normalize_lower_is_better(
        response
    )

    throughput_score = normalize_higher_is_better(
        throughput
    )

    utilization_score = normalize_higher_is_better(
        utilization
    )

    scores = {}

    for i, scheduler in enumerate(scheduler_names):

        scores[scheduler] = (

            WEIGHTS["waiting"]
            * waiting_score[i]

            + WEIGHTS["turnaround"]
            * turnaround_score[i]

            + WEIGHTS["response"]
            * response_score[i]

            + WEIGHTS["throughput"]
            * throughput_score[i]

            + WEIGHTS["cpu_utilization"]
            * utilization_score[i]
        )

    return scores


# ============================================================
# PROCESS WORKLOADS
# ============================================================

evaluation_results = []

workload_groups = df.groupby("workload_id")

total = len(workload_groups)

for index, (workload_id, group) in enumerate(
    workload_groups,
    start=1
):

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = extract_features(group)

    # --------------------------------------------------------
    # LightSched prediction
    # --------------------------------------------------------

    predicted_scheduler = model.predict(
        features[FEATURE_COLUMNS]
    )[0]

    # --------------------------------------------------------
    # Convert workload
    # --------------------------------------------------------

    processes = convert_workload(group)

    # --------------------------------------------------------
    # Run all schedulers
    # --------------------------------------------------------

    scheduler_results = {}

    for scheduler_name, scheduler_function in schedulers.items():

        scheduler_results[scheduler_name] = (
            scheduler_function(processes)
        )

    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    scores = calculate_scores(
        scheduler_results
    )

    # --------------------------------------------------------
    # Find oracle
    # --------------------------------------------------------

    oracle_scheduler = max(
        scores,
        key=scores.get
    )

    # --------------------------------------------------------
    # Get LightSched metrics
    # --------------------------------------------------------

    lightsched_metrics = (
        scheduler_results[predicted_scheduler][1]
    )

    # --------------------------------------------------------
    # Get oracle metrics
    # --------------------------------------------------------

    oracle_metrics = (
        scheduler_results[oracle_scheduler][1]
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    lightsched_score = scores[
        predicted_scheduler
    ]

    oracle_score = scores[
        oracle_scheduler
    ]

    # --------------------------------------------------------
    # Score gap / regret
    # --------------------------------------------------------

    score_gap = (
        oracle_score
        - lightsched_score
    )

    if oracle_score != 0:

        regret_percentage = (
            score_gap
            / oracle_score
        ) * 100

    else:

        regret_percentage = 0

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {

        "workload_id": workload_id,

        "predicted_scheduler":
            predicted_scheduler,

        "oracle_scheduler":
            oracle_scheduler,

        "prediction_correct":
            predicted_scheduler == oracle_scheduler,

        "lightsched_score":
            lightsched_score,

        "oracle_score":
            oracle_score,

        "score_gap":
            score_gap,

        "regret_percentage":
            regret_percentage,

        "lightsched_waiting":
            lightsched_metrics[
                "avg_waiting_time"
            ],

        "lightsched_turnaround":
            lightsched_metrics[
                "avg_turnaround_time"
            ],

        "lightsched_response":
            lightsched_metrics[
                "avg_response_time"
            ],

        "lightsched_throughput":
            lightsched_metrics[
                "throughput"
            ],

        "lightsched_cpu_utilization":
            lightsched_metrics[
                "cpu_utilization"
            ],

        "oracle_waiting":
            oracle_metrics[
                "avg_waiting_time"
            ],

        "oracle_turnaround":
            oracle_metrics[
                "avg_turnaround_time"
            ],

        "oracle_response":
            oracle_metrics[
                "avg_response_time"
            ],

        "oracle_throughput":
            oracle_metrics[
                "throughput"
            ],

        "oracle_cpu_utilization":
            oracle_metrics[
                "cpu_utilization"
            ]
    }

    # --------------------------------------------------------
    # Store every fixed scheduler's score
    # --------------------------------------------------------

    for scheduler in schedulers:

        result[
            f"{scheduler}_score"
        ] = scores[scheduler]

    evaluation_results.append(result)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 50 == 0 or index == total:

        print(
            f"Processed {index}/{total} workloads"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    evaluation_results
)


# ============================================================
# OVERALL RESULTS
# ============================================================

correct = results_df[
    "prediction_correct"
].sum()

total_workloads = len(results_df)

prediction_accuracy = (
    correct / total_workloads
) * 100


average_lightsched_score = results_df[
    "lightsched_score"
].mean()

average_oracle_score = results_df[
    "oracle_score"
].mean()

average_score_gap = results_df[
    "score_gap"
].mean()

average_regret = results_df[
    "regret_percentage"
].mean()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LIGHTSCHED PERFORMANCE RESULTS")
print("=" * 70)

print(
    f"\nCorrect scheduler selections:"
    f" {correct}/{total_workloads}"
)

print(
    f"Scheduler selection accuracy:"
    f" {prediction_accuracy:.2f}%"
)

print(
    f"\nAverage LightSched score:"
    f" {average_lightsched_score:.4f}"
)

print(
    f"Average Oracle score:"
    f" {average_oracle_score:.4f}"
)

print(
    f"Average score gap:"
    f" {average_score_gap:.4f}"
)

print(
    f"Average regret:"
    f" {average_regret:.2f}%"
)


# ============================================================
# LIGHTSCHED VS ORACLE METRICS
# ============================================================

print("\n" + "=" * 70)
print("LIGHTSCHED VS ORACLE")
print("=" * 70)

print(
    f"\nWaiting Time:"
)

print(
    f"LightSched : "
    f"{results_df['lightsched_waiting'].mean():.4f}"
)

print(
    f"Oracle     : "
    f"{results_df['oracle_waiting'].mean():.4f}"
)


print(
    f"\nTurnaround Time:"
)

print(
    f"LightSched : "
    f"{results_df['lightsched_turnaround'].mean():.4f}"
)

print(
    f"Oracle     : "
    f"{results_df['oracle_turnaround'].mean():.4f}"
)


print(
    f"\nResponse Time:"
)

print(
    f"LightSched : "
    f"{results_df['lightsched_response'].mean():.4f}"
)

print(
    f"Oracle     : "
    f"{results_df['oracle_response'].mean():.4f}"
)


print(
    f"\nThroughput:"
)

print(
    f"LightSched : "
    f"{results_df['lightsched_throughput'].mean():.4f}"
)

print(
    f"Oracle     : "
    f"{results_df['oracle_throughput'].mean():.4f}"
)


print(
    f"\nCPU Utilization:"
)

print(
    f"LightSched : "
    f"{results_df['lightsched_cpu_utilization'].mean():.4f}"
)

print(
    f"Oracle     : "
    f"{results_df['oracle_cpu_utilization'].mean():.4f}"
)


# ============================================================
# HOW OFTEN EACH SCHEDULER WAS SELECTED
# ============================================================

print("\n" + "=" * 70)
print("LIGHTSCHED PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    results_df[
        "predicted_scheduler"
    ].value_counts()
)

print("\nOracle distribution:")

print(
    results_df[
        "oracle_scheduler"
    ].value_counts()
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nSaved to:"
)

print(OUTPUT_FILE)