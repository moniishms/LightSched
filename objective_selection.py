import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from schedulers_v2 import round_robin, sjf, priority_scheduling, mlfq

SCHED = {"RR": round_robin, "SJF": sjf, "Priority": priority_scheduling, "MLFQ": mlfq}
NAMES = list(SCHED)
OBJ = ["avg_turnaround", "avg_response", "max_waiting", "p95_turnaround"]
FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]
RF = dict(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)


def to_procs(g):
    return [dict(pid=int(r.pid), arrival_time=int(r.arrival_time), burst_time=int(r.burst_time),
                 priority=int(r.priority), io_frequency=int(r.io_frequency)) for r in g.itertuples()]


def metrics(raw_file, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col="workload_id")
    raw = pd.read_csv(raw_file)
    rows = []
    for k, (wid, g) in enumerate(raw.groupby("workload_id"), 1):
        procs = to_procs(g)
        row = {"workload_id": wid}
        for n, f in SCHED.items():
            res, s = f(procs)
            row[f"{n}|avg_turnaround"] = s["avg_turnaround_time"]
            row[f"{n}|avg_response"] = s["avg_response_time"]
            row[f"{n}|max_waiting"] = max(r["waiting_time"] for r in res)
            row[f"{n}|p95_turnaround"] = np.percentile([r["turnaround_time"] for r in res], 95)
        rows.append(row)
        if k % 500 == 0:
            print(f"  {raw_file}: {k} workloads")
    D = pd.DataFrame(rows).set_index("workload_id")
    D.to_csv(cache)
    return D


def unseen_features(raw_file):
    rows = {}
    for wid, g in pd.read_csv(raw_file).groupby("workload_id"):
        n = len(g)
        span = g.arrival_time.max() - g.arrival_time.min() + 1
        rows[wid] = {"num_processes": n,
                     "avg_arrival_time": g.arrival_time.mean(),
                     "arrival_time_variance": g.arrival_time.var(ddof=0),
                     "avg_burst_time": g.burst_time.mean(),
                     "burst_time_variance": g.burst_time.var(ddof=0),
                     "avg_priority": g.priority.mean(),
                     "avg_io_frequency": g.io_frequency.mean(),
                     "arrival_rate": n / span}
    return pd.DataFrame.from_dict(rows, orient="index")


def regret(M, choice):
    cols = list(M.columns)
    chosen = M.to_numpy()[np.arange(len(M)), [cols.index(c) for c in choice]]
    best = M.min(axis=1).to_numpy()
    return (chosen - best) / np.maximum(best, 1.0) * 100.0   # denominator floored at 1 time unit


print("Training-set metrics (cached after first run)...")
TR = metrics("lightsched_workload_dataset.csv", "objective_train_raw.csv")
print("Unseen-set metrics...")
UN = metrics("lightsched_unseen_workload_dataset.csv", "objective_check_raw.csv")

Xtr = pd.read_csv("lightsched_ml_features.csv").set_index("workload_id")[FEATURES].loc[TR.index]
Xun = unseen_features("lightsched_unseen_workload_dataset.csv").loc[UN.index][FEATURES]

out = []
for o in OBJ:
    Mtr = pd.DataFrame({n: TR[f"{n}|{o}"] for n in NAMES})
    Mun = pd.DataFrame({n: UN[f"{n}|{o}"] for n in NAMES})
    ytr = Mtr.idxmin(axis=1)

    # best fixed scheduler chosen on TRAIN only
    fixed = {n: regret(Mtr, [n] * len(Mtr)).mean() for n in NAMES}
    best_fixed = min(fixed, key=fixed.get)

    policies = {
        "Always-SJF": ["SJF"] * len(Mun),
        f"Best-fixed ({best_fixed})": [best_fixed] * len(Mun),
        "RF": RandomForestClassifier(**RF).fit(Xtr, ytr).predict(Xun),
        "RF balanced": RandomForestClassifier(**RF, class_weight="balanced").fit(Xtr, ytr).predict(Xun),
    }
    for name, pred in policies.items():
        r = regret(Mun, list(pred))
        out.append({"objective": o, "policy": name, "mean_regret_%": r.mean(),
                    "median_%": np.median(r), "p90_%": np.percentile(r, 90),
                    "within_5%": (r <= 5).mean()})

T = pd.DataFrame(out).round(3)
print("\nRAW-METRIC REGRET on the 500 unseen workloads (lower is better)")
print(T.to_string(index=False))
T.to_csv("objective_selection_results.csv", index=False)