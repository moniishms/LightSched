import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from schedulers_extra import CANDIDATES

NAMES = list(CANDIDATES)
OBJ = ["avg_turnaround", "avg_response", "max_waiting", "p95_turnaround"]
FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]
RF = dict(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
CACHE = "fresh_raw.csv"


def to_procs(g):
    return [dict(pid=int(r.pid), arrival_time=int(r.arrival_time), burst_time=int(r.burst_time),
                 priority=int(r.priority), io_frequency=int(r.io_frequency)) for r in g.itertuples()]


def holdout_metrics(raw):
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE, index_col="workload_id")
    rows = []
    for k, (wid, g) in enumerate(raw.groupby("workload_id"), 1):
        procs = to_procs(g)
        row = {"workload_id": wid}
        for n, f in CANDIDATES.items():
            res, s = f(procs)
            row[f"{n}|avg_turnaround"] = s["avg_turnaround_time"]
            row[f"{n}|avg_response"] = s["avg_response_time"]
            row[f"{n}|max_waiting"] = max(r["waiting_time"] for r in res)
            row[f"{n}|p95_turnaround"] = np.percentile([r["turnaround_time"] for r in res], 95)
        rows.append(row)
        if k % 100 == 0:
            print(f"  holdout: {k}/500", flush=True)
    D = pd.DataFrame(rows).set_index("workload_id")
    D.to_csv(CACHE)
    return D


def feats(raw, ids):
    rows = {}
    for wid, g in raw.groupby("workload_id"):
        n = len(g); span = g.arrival_time.max() - g.arrival_time.min() + 1
        rows[wid] = {"num_processes": n, "avg_arrival_time": g.arrival_time.mean(),
                     "arrival_time_variance": g.arrival_time.var(ddof=0),
                     "avg_burst_time": g.burst_time.mean(),
                     "burst_time_variance": g.burst_time.var(ddof=0),
                     "avg_priority": g.priority.mean(), "avg_io_frequency": g.io_frequency.mean(),
                     "arrival_rate": n / span}
    return pd.DataFrame.from_dict(rows, orient="index").loc[ids][FEATURES]


def combined(D):
    mats = []
    for o in OBJ:
        M = pd.DataFrame({n: D[f"{n}|{o}"] for n in NAMES})
        mats.append((M.sub(M.min(axis=1), axis=0).div(np.maximum(M.min(axis=1), 1.0), axis=0) * 100).to_numpy())
    return pd.DataFrame(np.maximum.reduce(mats), index=D.index, columns=NAMES)


raw = pd.read_csv("lightsched_fresh_holdout.csv")
print("Holdout:", raw.workload_id.nunique(), "workloads (fresh seed 20261010)")

HO = holdout_metrics(raw)
TR = pd.read_csv("extra_train_raw.csv", index_col="workload_id")
Xtr = pd.read_csv("lightsched_ml_features.csv").set_index("workload_id")[FEATURES].loc[TR.index]
Xho = feats(raw, HO.index)

Ctr, Cho = combined(TR), combined(HO)
bf = Ctr.mean().idxmin()

clf = RandomForestClassifier(**RF).fit(Xtr, Ctr.idxmin(axis=1))
reg = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                            random_state=42, n_jobs=-1).fit(Xtr, Ctr[NAMES])

pick = lambda choice: np.array([Cho.loc[w, c] for w, c in zip(Cho.index, choice)])
res = {
    "Oracle": Cho.min(axis=1).to_numpy(),
    f"Best-fixed ({bf}, chosen on train)": Cho[bf].to_numpy(),
    "RF classifier": pick(clf.predict(Xho)),
    "RF cost-sensitive (regressor)": pick(np.array(NAMES)[reg.predict(Xho).argmin(axis=1)]),
}
base = res[f"Best-fixed ({bf}, chosen on train)"]
headroom = base.mean() - res["Oracle"].mean()
rng = np.random.default_rng(0)

print("\nCOMBINED worst-case regret on the fresh holdout (lower is better)")
for k, v in res.items():
    line = f"{k:38s} mean {v.mean():6.2f}%   median {np.median(v):6.2f}%"
    if k.startswith("RF"):
        d = base - v
        ci = np.percentile([rng.choice(d, len(d)).mean() for _ in range(5000)], [2.5, 97.5])
        line += f"   gain vs best-fixed {d.mean():5.2f} pp  CI [{ci[0]:.2f}, {ci[1]:.2f}]  ({d.mean() / headroom * 100:.0f}% of headroom)"
    print(line)
print(f"\nHeadroom (best-fixed - oracle): {headroom:.2f} pp")