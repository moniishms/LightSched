import pandas as pd
import joblib

# --------------------------------------------------
# 1. Load real Android workload windows
# --------------------------------------------------

df = pd.read_csv("lightsched_real_world_3000.csv")

# --------------------------------------------------
# 2. Load trained LightSched model
# --------------------------------------------------

model = joblib.load("models/random_forest_tuned.joblib")

# --------------------------------------------------
# 3. Same features used during LightSched training
# --------------------------------------------------

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

X = df[features].copy()

# --------------------------------------------------
# 4. Handle unavailable I/O feature
# --------------------------------------------------

# The exported Perfetto sched_slice data does not
# provide the I/O-frequency feature used in training.
# For this demonstration, use 0 as the adapter value.

X["avg_io_frequency"] = 0.0

# --------------------------------------------------
# 5. LightSched prediction
# --------------------------------------------------

predictions = model.predict(X)

df["predicted_scheduler"] = predictions

# --------------------------------------------------
# 6. Prediction confidence
# --------------------------------------------------

probabilities = model.predict_proba(X)

df["prediction_confidence"] = probabilities.max(axis=1)

# Convert confidence to percentage
df["prediction_confidence_percent"] = (
    df["prediction_confidence"] * 100
)

# --------------------------------------------------
# 7. Save complete demonstration results
# --------------------------------------------------

output_file = "lightsched_real_world_demo_predictions.csv"

df.to_csv(output_file, index=False)

# --------------------------------------------------
# 8. Display demonstration
# --------------------------------------------------

print()
print("=" * 60)
print("        LIGHTSCHED REAL-WORLD DEMONSTRATION")
print("=" * 60)

print()
print(f"Real Android workload windows : {len(df)}")
print("Window duration               : 100 ms")
print("Total trace duration           : 26.12 seconds")
print("CPUs observed                  : 8")
print("Unique Android threads         : 889")

print()
print("-" * 60)
print("SCHEDULER PREDICTION DISTRIBUTION")
print("-" * 60)

distribution = df["predicted_scheduler"].value_counts()

for scheduler, count in distribution.items():
    percentage = (count / len(df)) * 100

    print(
        f"{scheduler:<12} : "
        f"{count:4d} windows "
        f"({percentage:5.2f}%)"
    )

# --------------------------------------------------
# 9. Show selected real workload windows
# --------------------------------------------------

print()
print("-" * 60)
print("SAMPLE REAL-WORLD PREDICTIONS")
print("-" * 60)

# Select windows spread across the trace
sample_positions = [
    0,
    99,
    499,
    999,
    1499,
    1999,
    2499,
    2999
]

for position in sample_positions:

    row = df.iloc[position]

    print()
    print(f"Window {int(row['workload_id'])}")
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

# --------------------------------------------------
# 10. Final message
# --------------------------------------------------

print()
print("=" * 60)
print("LightSched successfully generated scheduler")
print("recommendations from real Android workload windows.")
print("=" * 60)

print()
print(f"Results saved to: {output_file}")