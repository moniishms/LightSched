import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import sys
warnings.filterwarnings("ignore")

FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]
SCHEDS = ["RR", "SJF", "Priority", "MLFQ"]
RF_PARAMS = dict(n_estimators=200, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", random_state=42, n_jobs=-1)

INITIAL, BATCH, N_BATCHES = 500, 50, 90
HISTORY_WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else None
TAG = "" if HISTORY_WINDOW is None else f"_w{HISTORY_WINDOW}"

PHASES = [(1, 10, "mixed"), (11, 20, "cpu_bound"), (21, 30, "interactive"),
          (31, 40, "priority_heavy"), (41, 50, "bursty"), (51, 60, "sjf_friendly"),
          (61, 70, "fairness"), (71, 80, "cpu_bound"), (81, 90, "interactive")]


def phase_of(b):
    return next(name for lo, hi, name in PHASES if lo <= b <= hi)


df = pd.merge(pd.read_csv("lightsched_shifted_ml_features.csv"),
              pd.read_csv("lightsched_shifted_labeled_dataset_v2.csv"),
              on="workload_id").sort_values("workload_id").reset_index(drop=True)
assert len(df) == INITIAL + BATCH * N_BATCHES, len(df)


def regret(rows, preds):
    scores = rows[[f"{s}_score" for s in SCHEDS]].to_numpy()
    oracle = scores.max(axis=1)
    got = np.array([scores[i, SCHEDS.index(p)] for i, p in enumerate(preds)])
    return np.where(oracle != 0, (oracle - got) / np.where(oracle == 0, 1, oracle) * 100, 0.0)


def fit(train):
    return RandomForestClassifier(**RF_PARAMS).fit(train[FEATURES], train["best_scheduler"])


init = df.iloc[:INITIAL]
static_model = fit(init)
adaptive_model = fit(init)
history = init.copy()

batch_rows, pred_rows = [], []

for b in range(1, N_BATCHES + 1):
    batch = df.iloc[INITIAL + (b - 1) * BATCH: INITIAL + b * BATCH]
    y = batch["best_scheduler"]

    for mode, model in [("adaptive", adaptive_model), ("static", static_model)]:
        pred = model.predict(batch[FEATURES])
        reg = regret(batch, pred)
        batch_rows.append({
            "batch_id": b, "phase": phase_of(b), "mode": mode, "batch_size": len(batch),
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "avg_regret_pct": reg.mean(),
        })
        pred_rows.append(pd.DataFrame({
            "batch_id": b, "phase": phase_of(b), "mode": mode,
            "workload_id": batch["workload_id"].values,
            "actual_best_scheduler": y.values, "predicted_scheduler": pred,
            "regret_pct": reg}))

    # learn from the batch only AFTER predicting it
    history = pd.concat([history, batch])
    train = history if HISTORY_WINDOW is None else history.tail(HISTORY_WINDOW)
    adaptive_model = fit(train)

    if b % 10 == 0:
        print(f"batch {b}/{N_BATCHES} done")

results = pd.DataFrame(batch_rows)
preds = pd.concat(pred_rows, ignore_index=True)
results.to_csv(f"adaptive_lightsched_results{TAG}.csv", index=False)
preds.to_csv(f"adaptive_lightsched_predictions{TAG}.csv", index=False)

print("\nMean of per-batch metrics:")
print(results.groupby("mode")[["accuracy", "balanced_accuracy", "macro_f1", "avg_regret_pct"]].mean().round(4))

print("\nPer-phase pooled accuracy and regret (more reliable than per-batch macro-F1):")
pooled = (preds.assign(correct=preds.actual_best_scheduler == preds.predicted_scheduler)
               .groupby(["phase", "mode"])[["correct", "regret_pct"]].mean().round(4)
               .unstack("mode"))
print(pooled)
print(f"\nWindow: {HISTORY_WINDOW}")
print(f"Saved: adaptive_lightsched_results{TAG}.csv, adaptive_lightsched_predictions{TAG}.csv")