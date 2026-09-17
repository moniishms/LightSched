import pandas as pd
import joblib


# ============================================================
# LIGHTSCHED REAL-WORLD / PERFETTO DEMONSTRATION
# ============================================================

INPUT_FILE = "lightsched_real_world_3000.csv"
MODEL_FILE = "models/random_forest_tuned.joblib"
OUTPUT_FILE = "lightsched_real_world_demo_predictions.csv"


# ============================================================
# 1. Load real Android workload windows
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Make sure windows are in chronological order
if "window_start_s" in df.columns:
    df = df.sort_values("window_start_s").reset_index(drop=True)


# ============================================================
# 2. Load trained LightSched model
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# 3. Features used during LightSched training
# ============================================================

features = [
    "num_processes",
    "avg_arrival_time",
    "arrival_time_variance",
    "avg_burst_time",
    "burst_time_variance",
    "avg_priority",
    "avg_io_frequency",
    "arrival_rate"
]


# Check that required features exist
missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )


# ============================================================
# 4. Prepare feature matrix
# ============================================================

X = df[features].copy()


# ============================================================
# 5. Handle unavailable I/O-frequency feature
# ============================================================

# Perfetto sched_slice export does not provide the same
# I/O-frequency feature used by the synthetic training data.
# Therefore, use 0 as the adapter value for this demonstration.

X["avg_io_frequency"] = 0.0


# ============================================================
# 6. Handle any remaining missing values
# ============================================================

if X.isnull().any().any():

    missing_counts = X.isnull().sum()

    print("Missing feature values detected:")
    print(missing_counts[missing_counts > 0])

    # Use column medians as a simple demonstration fallback
    X = X.fillna(X.median(numeric_only=True))

    # If any column is still completely empty
    X = X.fillna(0.0)


# ============================================================
# 7. LightSched prediction
# ============================================================

predictions = model.predict(X)

df["predicted_scheduler"] = predictions


# ============================================================
# 8. Prediction confidence
# ============================================================

probabilities = model.predict_proba(X)

df["prediction_confidence"] = probabilities.max(axis=1)

df["prediction_confidence_percent"] = (
    df["prediction_confidence"] * 100
)


# ============================================================
# 9. Save demonstration results
# ============================================================

# Existing file is automatically overwritten.
df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 10. Trace information
# ============================================================

num_windows = len(df)

if "window_end_s" in df.columns:
    trace_duration = df["window_end_s"].max()
else:
    trace_duration = None

if "cpu" in df.columns:
    num_cpus = df["cpu"].nunique()
else:
    num_cpus = "N/A"

if "utid" in df.columns:
    num_threads = df["utid"].nunique()
elif "num_processes" in df.columns:
    num_threads = df["num_processes"].sum()
else:
    num_threads = "N/A"


# ============================================================
# 11. Display demonstration summary
# ============================================================

print()
print("=" * 65)
print("              LIGHTSCHED REAL-WORLD DEMONSTRATION")
print("=" * 65)

print()
print(f"Android workload windows : {num_windows}")

if "window_start_s" in df.columns and "window_end_s" in df.columns:

    window_duration = (
        df["window_end_s"] - df["window_start_s"]
    ).median()

    print(
        f"Window duration         : "
        f"{window_duration * 1000:.0f} ms"
    )

if trace_duration is not None:
    print(
        f"Trace duration          : "
        f"{trace_duration:.2f} seconds"
    )

print(f"CPUs observed           : {num_cpus}")
print(f"Unique Android threads  : {num_threads}")


# ============================================================
# 12. Prediction distribution
# ============================================================

print()
print("-" * 65)
print("SCHEDULER PREDICTION DISTRIBUTION")
print("-" * 65)

distribution = (
    df["predicted_scheduler"]
    .value_counts()
    .sort_index()
)

for scheduler, count in distribution.items():

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"{scheduler:<12} : "
        f"{count:4d} windows "
        f"({percentage:5.2f}%)"
    )


# ============================================================
# 13. Confidence summary
# ============================================================

print()
print("-" * 65)
print("PREDICTION CONFIDENCE")
print("-" * 65)

print(
    f"Mean confidence         : "
    f"{df['prediction_confidence_percent'].mean():.2f}%"
)

print(
    f"Minimum confidence      : "
    f"{df['prediction_confidence_percent'].min():.2f}%"
)

print(
    f"Maximum confidence      : "
    f"{df['prediction_confidence_percent'].max():.2f}%"
)


# ============================================================
# 14. Selected real-world windows
# ============================================================

print()
print("-" * 65)
print("SAMPLE REAL-WORLD PREDICTIONS")
print("-" * 65)


# Select windows distributed across the trace
sample_positions = [
    0,
    99,
    499,
    999,
    1499,
    1999,
    2499,
    len(df) - 1
]

# Remove duplicates / invalid positions
sample_positions = sorted(
    set(
        position
        for position in sample_positions
        if 0 <= position < len(df)
    )
)


for position in sample_positions:

    row = df.iloc[position]

    print()

    if "workload_id" in df.columns:
        print(
            f"Window {int(row['workload_id'])}"
        )
    else:
        print(
            f"Window {position + 1}"
        )

    if (
        "window_start_s" in df.columns
        and "window_end_s" in df.columns
    ):
        print(
            f"Time       : "
            f"{row['window_start_s']:.3f}s - "
            f"{row['window_end_s']:.3f}s"
        )

    print(
        f"Threads    : "
        f"{int(row['num_processes'])}"
    )

    print(
        f"Avg slice  : "
        f"{row['avg_burst_time']:.3f} ms"
    )

    print(
        f"Priority   : "
        f"{row['avg_priority']:.2f}"
    )

    if "scheduling_event_rate" in df.columns:
        print(
            f"Event rate : "
            f"{row['scheduling_event_rate']:.0f}/s"
        )

    print(
        f"Prediction : "
        f"{row['predicted_scheduler']}"
    )

    print(
        f"Confidence : "
        f"{row['prediction_confidence_percent']:.2f}%"
    )


# ============================================================
# 15. Final message
# ============================================================

print()
print("=" * 65)
print("LightSched generated scheduler recommendations")
print("from trace-derived Android workload windows.")
print("=" * 65)

print()
print("NOTE:")
print("These are trace-driven scheduler recommendations.")
print("They do not represent kernel-level scheduler replacement")
print("or counterfactual performance measurements on Android.")

print()
print(f"Results saved to: {OUTPUT_FILE}")