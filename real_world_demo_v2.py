import numpy as np
import pandas as pd
import joblib

INPUT_FILE = "lightsched_real_world_3000.csv"
MODEL_FILE = "models/random_forest_tuned.joblib"
TRAIN_FILE = "lightsched_ml_features.csv"
OUTPUT_FILE = "lightsched_real_world_demo_predictions_v2.csv"

FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]
STEP = 12            # 100 ms window / 8.68 ms stride ~ 12 -> non-overlapping subsample

df = pd.read_csv(INPUT_FILE).sort_values("window_start_s").reset_index(drop=True)
train = pd.read_csv(TRAIN_FILE)[FEATURES]
model = joblib.load(MODEL_FILE)

# ---- Adapter: trace units -> training units (1 training time unit = 1 ms) ----
n = df["num_processes"].replace(0, np.nan)
slices_per_thread = df["scheduling_slices"] / n

X = pd.DataFrame({
    "num_processes": df["num_processes"],
    "avg_arrival_time": df["avg_arrival_time"] * 1000.0,            # s -> ms
    "arrival_time_variance": df["arrival_time_variance"] * 1e6,     # s^2 -> ms^2
    "avg_burst_time": df["avg_burst_time"] * slices_per_thread,     # per-thread CPU in window (ms)
    "burst_time_variance": df["burst_time_variance"],               # PROXY: slice-duration variance
    "avg_priority": (1 + 9 * df["avg_priority"] / 139.0).clip(1, 10),  # kernel prio -> 1..10
    "avg_io_frequency": (slices_per_thread - 1).clip(0, 10),        # off-CPU returns per thread
    "arrival_rate": df["arrival_rate"] / 1000.0,                    # per s -> per ms
})[FEATURES]

valid = X.notna().all(axis=1)
lo, hi = train.min(), train.max()
in_range = (X >= lo) & (X <= hi)
usable = valid & in_range.all(axis=1)

print("=" * 65)
print("REAL-WORLD DEMO v2 (unit-adapted features)")
print("=" * 65)
print(f"Windows total                      : {len(df)}")
print(f"Windows with complete features     : {int(valid.sum())}  (dropped {int((~valid).sum())} empty windows)")
print(f"Windows fully inside training range: {int(usable.sum())}  ({usable.sum() / len(df) * 100:.1f}% of all)")
print("\nPer-feature in-range % (among complete windows):")
print((in_range[valid].mean() * 100).round(1).to_string())

# ---- Predictions ----
pred = pd.Series(model.predict(X[valid]), index=X.index[valid])
score = pd.Series(model.predict_proba(X[valid]).max(axis=1), index=X.index[valid])
df["predicted_scheduler"] = pred
df["model_score"] = score
df["in_training_range"] = usable
df.to_csv(OUTPUT_FILE, index=False)

def dist(p):
    return (p.value_counts(normalize=True) * 100).round(1).to_dict()

print("\nPrediction mix, all complete windows :", dist(pred))
print("Prediction mix, in-range windows only:", dist(pred[usable[valid]]) if usable.any() else "none")

sub = pred.iloc[::STEP]
flips = (sub != sub.shift()).iloc[1:].mean()
print(f"\nFlip rate on non-overlapping windows (every {STEP}th, n={len(sub)}): {flips * 100:.1f}%")

X_alt = X[valid].copy()
X_alt["avg_io_frequency"] = 0.0
print(f"Agreement if I/O frequency were 0       : {(model.predict(X_alt) == pred).mean() * 100:.1f}%")
print(f"Mean model score (uncalibrated RF vote) : {score.mean() * 100:.1f}%")

print("\nASSUMPTIONS (state these in the paper):")
print("- 1 training time unit = 1 ms; priority mapped linearly from the kernel 0-139 scale.")
print("- Per-thread CPU = avg slice x slices per thread; I/O proxy = slices per thread - 1.")
print("- Burst variance is a proxy (slice variance), not per-thread total variance.")
print("- No ground truth; observed slices were produced by Android's own scheduler.")
print(f"\nSaved: {OUTPUT_FILE}")