import pandas as pd
import joblib


# ============================================================
# LIGHTSCHED REAL-WORLD / PERFETTO DEMONSTRATION
# ============================================================

INPUT_FILE = "lightsched_real_world_3000.csv"
MODEL_FILE = "models/random_forest_tuned.joblib"
TRAIN_FEATURE_FILE = "lightsched_ml_features.csv"
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

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )


# ============================================================
# 4. Prepare feature matrix (fill missing values first)
# ============================================================

X = df[features].copy()

if X.isnull().any().any():

    missing_counts = X.isnull().sum()

    print("Missing feature values detected:")
    print(missing_counts[missing_counts > 0])

    # Column medians as a simple demonstration fallback
    X = X.fillna(X.median(numeric_only=True))

    # If any column is still completely empty
    X = X.fillna(0.0)


# ============================================================
# 5. Training-range check (how far are trace windows from the
#    synthetic training data the model has seen?)
# ============================================================

train_feats = pd.read_csv(TRAIN_FEATURE_FILE)

lo = train_feats[features].min()
hi = train_feats[features].max()

in_range = (X >= lo) & (X <= hi)

# I/O frequency is excluded: it is replaced by an adapter below
pct_inside = (
    in_range.drop(columns="avg_io_frequency").all(axis=1).mean() * 100
)

print()
print("-" * 65)
print("TRAINING-RANGE CHECK")
print("-" * 65)
print(f"Windows fully inside training range (excl. I/O): {pct_inside:.1f}%")
print("Per-feature in-range %:")
print((in_range.mean() * 100).round(1).to_string())


# ============================================================
# 6. Handle unavailable I/O-frequency feature
# ============================================================

# Perfetto sched_slice export does not provide the same
# I/O-frequency feature used by the synthetic training data.
# Therefore, use 0 as the adapter value for this demonstration.

X["avg_io_frequency"] = 0.0


# ============================================================
# 7. LightSched prediction
# ============================================================

predictions = model.predict(X)

df["predicted_scheduler"] = predictions


# ============================================================
# 8. Model score (random-forest vote share, NOT calibrated)
# ============================================================

probabilities = model.predict_proba(X)

df["prediction_confidence"] = probabilities.max(axis=1)

df["prediction_confidence_percent"] = (
    df["prediction_confidence"] * 100
)


# ============================================================
# 9. Stability and robustness checks
# ============================================================

# How often does the recommendation change between consecutive windows?
flip_rate = (
    (df["predicted_scheduler"] != df["predicted_scheduler"].shift())
    .iloc[1:]
    .mean()
)

# How much does the I/O adapter matter? Compare I/O = 0 against
# I/O = training median.
X_alt = X.copy()
X_alt["avg_io_frequency"] = train_feats["avg_io_frequency"].median()

io_agreement = (model.predict(X_alt) == predictions).mean()


# ============================================================
# 10. Save demonstration results
# ============================================================

# Existing file is automatically overwritten.
df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. Trace information
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
else:
    num_threads = "N/A (utid not in export)"


# ============================================================
# 12. Display demonstration summary
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
# 13. Prediction distribution
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
# 14. Stability and robustness summary
# ============================================================

print()
print("-" * 65)
print("STABILITY AND ROBUSTNESS")
print("-" * 65)

print(
    f"Prediction flip rate between consecutive windows : "
    f"{flip_rate * 100:.1f}%"
)

print(
    f"Agreement when I/O adapter = training median     : "
    f"{io_agreement * 100:.1f}%"
)


# ============================================================
# 15. Model score summary
# ============================================================

print()
print("-" * 65)
print("MODEL SCORE (RF vote share, uncalibrated)")
print("-" * 65)

print(
    f"Mean score              : "
    f"{df['prediction_confidence_percent'].mean():.2f}%"
)

print(
    f"Minimum score           : "
    f"{df['prediction_confidence_percent'].min():.2f}%"
)

print(
    f"Maximum score           : "
    f"{df['prediction_confidence_percent'].max():.2f}%"
)


# ============================================================
# 16. Selected real-world windows
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
        f"Score      : "
        f"{row['prediction_confidence_percent']:.2f}%"
    )


# ============================================================
# 17. Final message
# ============================================================

print()
print("=" * 65)
print("LightSched generated scheduler recommendations")
print("from trace-derived Android workload windows.")
print("=" * 65)

print()
print("NOTES:")
print("- These are trace-driven scheduler recommendations. They do not")
print("  represent kernel-level scheduler replacement or counterfactual")
print("  performance measurements on Android.")
print("- No ground-truth best scheduler exists for these windows.")
print("- I/O frequency is not available in the export and is set to 0.")
print("- Feature units and ranges differ from the synthetic training data;")
print("  see the training-range check above.")
print("- Observed slice lengths were produced by Android's own scheduler.")

print()
print(f"Results saved to: {OUTPUT_FILE}")