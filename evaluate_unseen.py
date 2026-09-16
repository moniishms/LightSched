import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

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

print("=" * 70)
print("LIGHTSCHED UNSEEN WORKLOAD EVALUATION")
print("=" * 70)

print("\nLoaded unseen dataset:")
print("Shape:", df.shape)

print(
    "Number of workloads:",
    df["workload_id"].nunique()
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print("\nLoaded model:")
print(MODEL_FILE)


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

    return {
        "num_processes": num_processes,
        "avg_arrival_time": avg_arrival_time,
        "arrival_time_variance": arrival_time_variance,
        "avg_burst_time": avg_burst_time,
        "burst_time_variance": burst_time_variance,
        "avg_priority": avg_priority,
        "avg_io_frequency": avg_io_frequency,
        "arrival_rate": arrival_rate
    }


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_lower_is_better(values):

    values = np.array(values, dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return np.ones(len(values))

    return (
        (maximum - values)
        / (maximum - minimum)
    )


def normalize_higher_is_better(values):

    values = np.array(values, dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return np.ones(len(values))

    return (
        (values - minimum)
        / (maximum - minimum)
    )


# ============================================================
# CALCULATE TRUE BEST SCHEDULER
# ============================================================

def calculate_best_scheduler(results):

    scheduler_names = list(schedulers.keys())

    waiting_values = [
        results[s][1]["avg_waiting_time"]
        for s in scheduler_names
    ]

    turnaround_values = [
        results[s][1]["avg_turnaround_time"]
        for s in scheduler_names
    ]

    response_values = [
        results[s][1]["avg_response_time"]
        for s in scheduler_names
    ]

    throughput_values = [
        results[s][1]["throughput"]
        for s in scheduler_names
    ]

    utilization_values = [
        results[s][1]["cpu_utilization"]
        for s in scheduler_names
    ]

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

    best_scheduler = max(
        scores,
        key=scores.get
    )

    return best_scheduler, scores


# ============================================================
# PROCESS ALL UNSEEN WORKLOADS
# ============================================================

results = []

workload_groups = df.groupby("workload_id")

total_workloads = len(workload_groups)


for index, (workload_id, group) in enumerate(
    workload_groups,
    start=1
):

    # --------------------------------------------------------
    # Convert workload
    # --------------------------------------------------------

    processes = convert_workload(group)

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    feature_dict = extract_features(group)

    feature_df = pd.DataFrame(
        [feature_dict]
    )

    # --------------------------------------------------------
    # LightSched prediction
    # --------------------------------------------------------

    predicted_scheduler = model.predict(
        feature_df[FEATURE_COLUMNS]
    )[0]

    # --------------------------------------------------------
    # Prediction probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        feature_df[FEATURE_COLUMNS]
    )[0]

    class_names = model.classes_

    prediction_probability = probabilities[
        list(class_names).index(
            predicted_scheduler
        )
    ]

    # --------------------------------------------------------
    # Run ALL schedulers
    # --------------------------------------------------------

    scheduler_results = {}

    for scheduler_name, scheduler_function in schedulers.items():

        scheduler_results[scheduler_name] = (
            scheduler_function(processes)
        )

    # --------------------------------------------------------
    # Find actual best scheduler
    # --------------------------------------------------------

    actual_best, scores = calculate_best_scheduler(
        scheduler_results
    )

    # --------------------------------------------------------
    # Get metrics for predicted scheduler
    # --------------------------------------------------------

    predicted_metrics = (
        scheduler_results[predicted_scheduler][1]
    )

    # --------------------------------------------------------
    # Get metrics for actual best scheduler
    # --------------------------------------------------------

    best_metrics = (
        scheduler_results[actual_best][1]
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {

        "workload_id": workload_id,

        "predicted_scheduler": predicted_scheduler,

        "actual_best_scheduler": actual_best,

        "prediction_probability": prediction_probability,

        "predicted_waiting": (
            predicted_metrics["avg_waiting_time"]
        ),

        "predicted_turnaround": (
            predicted_metrics["avg_turnaround_time"]
        ),

        "predicted_response": (
            predicted_metrics["avg_response_time"]
        ),

        "predicted_throughput": (
            predicted_metrics["throughput"]
        ),

        "predicted_cpu_utilization": (
            predicted_metrics["cpu_utilization"]
        ),

        "best_waiting": (
            best_metrics["avg_waiting_time"]
        ),

        "best_turnaround": (
            best_metrics["avg_turnaround_time"]
        ),

        "best_response": (
            best_metrics["avg_response_time"]
        ),

        "best_throughput": (
            best_metrics["throughput"]
        ),

        "best_cpu_utilization": (
            best_metrics["cpu_utilization"]
        ),

        "prediction_correct": (
            predicted_scheduler == actual_best
        )
    }

    # Add all scheduler scores

    for scheduler in schedulers:

        result[
            f"{scheduler}_score"
        ] = scores[scheduler]

    results.append(result)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 50 == 0 or index == total_workloads:

        print(
            f"Processed "
            f"{index}/{total_workloads} workloads"
        )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

y_true = results_df[
    "actual_best_scheduler"
]

y_pred = results_df[
    "predicted_scheduler"
]


accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


# ============================================================
# DISPLAY CLASSIFICATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("UNSEEN WORKLOAD CLASSIFICATION RESULTS")
print("=" * 70)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy  : {balanced_accuracy:.4f}"
)

print(
    f"Macro F1           : {macro_f1:.4f}"
)

print(
    f"Weighted F1        : {weighted_f1:.4f}"
)


print("\nActual Best Scheduler Distribution:")

print(
    y_true.value_counts()
)


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[
            "RR",
            "SJF",
            "Priority",
            "MLFQ"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        "RR",
        "SJF",
        "Priority",
        "MLFQ"
    ]
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual RR",
        "Actual SJF",
        "Actual Priority",
        "Actual MLFQ"
    ],
    columns=[
        "Pred RR",
        "Pred SJF",
        "Pred Priority",
        "Pred MLFQ"
    ]
)

print(cm_df)


# ============================================================
# CORRECT PREDICTIONS
# ============================================================

correct_predictions = (
    results_df["prediction_correct"]
    .sum()
)

print("\nCorrect scheduler selections:")

print(
    f"{correct_predictions}/"
    f"{len(results_df)}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    "lightsched_unseen_evaluation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("UNSEEN EVALUATION COMPLETE")
print("=" * 70)

print("\nSaved results to:")

print(output_file)