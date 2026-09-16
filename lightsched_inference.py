import pandas as pd
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
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/random_forest_tuned.joblib"

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
# FEATURE EXTRACTION
# ============================================================

def extract_features(workload):
    """
    Convert process-level workload data into the
    8 workload-level features used by LightSched.
    """

    num_processes = len(workload)

    avg_arrival_time = workload["arrival_time"].mean()
    arrival_time_variance = workload["arrival_time"].var()

    avg_burst_time = workload["burst_time"].mean()
    burst_time_variance = workload["burst_time"].var()

    avg_priority = workload["priority"].mean()

    avg_io_frequency = workload["io_frequency"].mean()

    # Arrival rate = number of processes / time span
    arrival_span = (
        workload["arrival_time"].max()
        - workload["arrival_time"].min()
    )

    if arrival_span == 0:
        arrival_rate = num_processes
    else:
        arrival_rate = num_processes / arrival_span

    features = pd.DataFrame([{
        "num_processes": num_processes,
        "avg_arrival_time": avg_arrival_time,
        "arrival_time_variance": arrival_time_variance,
        "avg_burst_time": avg_burst_time,
        "burst_time_variance": burst_time_variance,
        "avg_priority": avg_priority,
        "avg_io_frequency": avg_io_frequency,
        "arrival_rate": arrival_rate
    }])

    return features


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("\nLoading LightSched model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")
    print("Model: Tuned Random Forest")

    return model


# ============================================================
# PREDICT SCHEDULER
# ============================================================

def predict_scheduler(model, workload):

    features = extract_features(workload)

    prediction = model.predict(features[FEATURE_COLUMNS])[0]

    # Prediction probabilities
    probabilities = model.predict_proba(
        features[FEATURE_COLUMNS]
    )[0]

    classes = model.classes_

    probability_table = pd.DataFrame({
        "Scheduler": classes,
        "Probability": probabilities
    }).sort_values(
        "Probability",
        ascending=False
    )

    return prediction, features, probability_table


# ============================================================
# RUN SELECTED SCHEDULER
# ============================================================

def run_scheduler(scheduler_name, workload):

    # Convert dataframe to the format expected by
    # the V2 scheduler functions.

    processes = workload.to_dict("records")

    if scheduler_name == "RR":

        result = round_robin(processes)

    elif scheduler_name == "SJF":

        result = sjf(processes)

    elif scheduler_name == "Priority":

        result = priority_scheduling(processes)

    elif scheduler_name == "MLFQ":

        result = mlfq(processes)

    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}"
        )

    return result


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(scheduler_name, result):

    print("\n" + "=" * 70)
    print(f"{scheduler_name} RESULT")
    print("=" * 70)

    # Scheduler returns:
    # (process_results, summary_metrics)

    process_results, metrics = result

    print(f"Average Waiting Time   : {metrics['avg_waiting_time']:.4f}")
    print(f"Average Turnaround Time: {metrics['avg_turnaround_time']:.4f}")
    print(f"Average Response Time  : {metrics['avg_response_time']:.4f}")
    print(f"Throughput             : {metrics['throughput']:.4f}")
    print(f"CPU Utilization        : {metrics['cpu_utilization']:.4f}%")


# ============================================================
# MAIN LIGHTSCHED PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("LIGHTSCHED INFERENCE PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # 2. Load an unseen workload
    # --------------------------------------------------------
    #
    # For the first demonstration we use a workload from
    # the existing dataset.
    #
    # Later we will replace this with a completely new
    # workload / real workload trace.
    # --------------------------------------------------------

    raw_data = pd.read_csv(
        "lightsched_workload_dataset.csv"
    )

    workload_id = raw_data["workload_id"].max()

    workload = raw_data[
        raw_data["workload_id"] == workload_id
    ].copy()

    print("\n" + "=" * 70)
    print("NEW WORKLOAD")
    print("=" * 70)

    print(f"Workload ID       : {workload_id}")
    print(f"Number of processes: {len(workload)}")

    # --------------------------------------------------------
    # 3. Extract workload features
    # --------------------------------------------------------

    features = extract_features(workload)

    print("\nExtracted Features:")
    print("-" * 70)

    for column in FEATURE_COLUMNS:

        print(
            f"{column:<30}: "
            f"{features[column].iloc[0]:.4f}"
        )

    # --------------------------------------------------------
    # 4. Predict scheduler
    # --------------------------------------------------------

    predicted_scheduler, features, probabilities = (
        predict_scheduler(
            model,
            workload
        )
    )

    print("\n" + "=" * 70)
    print("LIGHTSCHED PREDICTION")
    print("=" * 70)

    print(
        f"\nPredicted Scheduler: "
        f"{predicted_scheduler}"
    )

    print("\nScheduler Probabilities:")
    print(probabilities.to_string(index=False))

    # --------------------------------------------------------
    # 5. Actually execute predicted scheduler
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXECUTING PREDICTED SCHEDULER")
    print("=" * 70)

    predicted_result = run_scheduler(
        predicted_scheduler,
        workload
    )

    display_result(
        predicted_scheduler,
        predicted_result
    )

    # --------------------------------------------------------
    # 6. Run all fixed schedulers on SAME workload
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARISON WITH FIXED SCHEDULERS")
    print("=" * 70)

    schedulers = [
        "RR",
        "SJF",
        "Priority",
        "MLFQ"
    ]

    results = {}

    for scheduler in schedulers:

        print(f"\nRunning {scheduler}...")

        results[scheduler] = run_scheduler(
            scheduler,
            workload
        )

    # --------------------------------------------------------
    # 7. Display comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    comparison = []

    for scheduler, result in results.items():

        process_results, metrics = result

        comparison.append({
            "Scheduler": scheduler,
            "Avg Waiting": metrics["avg_waiting_time"],
            "Avg Turnaround": metrics["avg_turnaround_time"],
            "Avg Response": metrics["avg_response_time"],
            "Throughput": metrics["throughput"],
            "CPU Utilization": metrics["cpu_utilization"]
        })

    comparison_df = pd.DataFrame(comparison)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n" + "=" * 70)
    print("LIGHTSCHED PREDICTED SCHEDULER")
    print("=" * 70)

    print(
        f"Predicted Scheduler: "
        f"{predicted_scheduler}"
    )

    predicted_row = comparison_df[
        comparison_df["Scheduler"] == predicted_scheduler
    ]

    print("\nPerformance of selected scheduler:")

    print(
        predicted_row.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n" + "=" * 70)
    print("LIGHTSCHED DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()