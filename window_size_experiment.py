import numpy as np
import pandas as pd
import joblib

RAW_FILE = "perfetto_sched_slice.csv"   # columns assumed: ts, dur (ns), cpu, utid, priority
TRAIN_FEATURES = "lightsched_ml_features.csv"
MODEL_FILE = "models/random_forest_tuned.joblib"
WINDOW_SIZES_MS = [10, 25, 50, 100, 250, 500, 1000]

FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]

raw = pd.read_csv(RAW_FILE)
raw["t_ms"] = (raw["ts"] - raw["ts"].min()) / 1e6
raw["dur_ms"] = raw["dur"] / 1e6

train = pd.read_csv(TRAIN_FEATURES)
lo, hi = train[FEATURES].min(), train[FEATURES].max()
model = joblib.load(MODEL_FILE)


def build_windows(w_ms):
    d = raw.assign(win=(raw["t_ms"] // w_ms).astype(int))
    th = (d.groupby(["win", "utid"])
            .agg(first=("t_ms", "min"), dur=("dur_ms", "mean"), prio=("priority", "mean"))
            .reset_index())
    th["arr"] = th["first"] - th["win"] * w_ms          # arrival offset inside window
    g = th.groupby("win")
    f = pd.DataFrame({
        "num_processes": g.size(),
        "avg_arrival_time": g["arr"].mean(),
        "arrival_time_variance": g["arr"].var(ddof=0),
        "avg_burst_time": g["dur"].mean(),
        "burst_time_variance": g["dur"].var(ddof=0),
        "avg_priority": g["prio"].mean(),
        "avg_io_frequency": 0.0,
    })
    span = g["arr"].max() - g["arr"].min() + 1
    f["arrival_rate"] = f["num_processes"] / span
    return f.fillna(0.0)[FEATURES]


rows = []
for w in WINDOW_SIZES_MS:
    X = build_windows(w)
    inr = (X >= lo) & (X <= hi)
    pred = model.predict(X)
    proba = model.predict_proba(X).max(axis=1)
    row = {
        "window_ms": w,
        "n_windows": len(X),
        "median_threads": X["num_processes"].median(),
        "pct_threads_in_5_30": inr["num_processes"].mean() * 100,
        "pct_all_features_in_range": inr.drop(columns="avg_io_frequency").all(axis=1).mean() * 100,
        "flip_rate": (pred[1:] != pred[:-1]).mean() if len(pred) > 1 else np.nan,
        "mean_max_proba": proba.mean(),
    }
    for cls, share in pd.Series(pred).value_counts(normalize=True).items():
        row[f"pct_{cls}"] = share * 100
    rows.append(row)

out = pd.DataFrame(rows).round(2)
print(out.to_string(index=False))
out.to_csv("window_size_experiment.csv", index=False)