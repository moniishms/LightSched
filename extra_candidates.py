import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from schedulers_extra import CANDIDATES

NAMES = list(CANDIDATES)
OBJ = ["avg_turnaround", "avg_response", "max_waiting", "p95_turnaround"]
FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]
RF = dict(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
N_TRAIN = 5000


def to_procs(g):
    return [dict(pid=int(r.pid), arrival_time=int(r.arrival_time), burst_time=int(r.burst_time),
                 priority=int(r.priority), io_frequency=int(r.io_frequency)) for r in g.itertuples()]


def metrics(raw_file, cache, limit=None):
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col="workload_id")
    raw = pd.read_csv(raw_file)
    rows = []
    for k, (wid, g) in enumerate(raw.groupby("workload_id"), 1):
        if limit and k > limit:
            break
        procs = to_procs(g)
        row = {"workload_id": wid}
        for n, f in CANDIDATES.items():
            res, s = f(procs)
            row[f"{n}|avg_turnaround"] = s["avg_turnaround_time"]
            row[f"{n}|avg_response"] = s["avg_response_time"]
            row[f"{n}|max_waiting"] = max(r["waiting_time"] for r in res)
            row[f"{n}|p95_turnaround"] = np.percentile([r["turnaround_time"] for r in res], 95)
        rows.append(row)
        if k % 250 == 0:
            print(f"  {raw_file}: {k}", flush=True)
    D = pd.DataFrame(rows).set_index("workload_id")
    D.to_csv(cache)
    return D


def feats(raw_file, ids):
    rows = {}
    for wid, g in pd.read_csv(raw_file).groupby("workload_id"):
        n = len(g); span = g.arrival_time.max() - g.arrival_time.min() + 1
        rows[wid] = {"num_processes": n, "avg_arrival_time": g.arrival_time.mean(),
                     "arrival_time_variance": g.arrival_time.var(ddof=0),
                     "avg_burst_time": g.burst_time.mean(),
                     "burst_time_variance": g.burst_time.var(ddof=0),
                     "avg_priority": g.priority.mean(), "avg_io_frequency": g.io_frequency.mean(),
                     "arrival_rate": n / span}
    return pd.DataFrame.from_dict(rows, orient="index").loc[ids][FEATURES]


def reg_matrix(D, obj):
    M = pd.DataFrame({n: D[f"{n}|{obj}"] for n in NAMES})
    return M.sub(M.min(axis=1), axis=0).div(np.maximum(M.min(axis=1), 1.0), axis=0) * 100


def pick(R, choice):
    return np.array([R.loc[w, c] for w, c in zip(R.index, choice)])


print("Training metrics..."); TR = metrics("lightsched_workload_dataset.csv", "extra_train_raw.csv", N_TRAIN)
print("Unseen metrics...");   UN = metrics("lightsched_unseen_workload_dataset.csv", "extra_unseen_raw.csv")
Xtr = pd.read_csv("lightsched_ml_features.csv").set_index("workload_id")[FEATURES].loc[TR.index]
Xun = feats("lightsched_unseen_workload_dataset.csv", UN.index)

rows = []
for o in OBJ:
    Rtr, Run = reg_matrix(TR, o), reg_matrix(UN, o)
    bf = Rtr.mean().idxmin()
    rf = RandomForestClassifier(**RF).fit(Xtr, Rtr.idxmin(axis=1))
    rows.append({"objective": o, "best_fixed": bf,
                 "best_fixed_regret_%": Run[bf].mean(),
                 "RF_regret_%": pick(Run, rf.predict(Xun)).mean(),
                 "oracle_winners": Run.idxmin(axis=1).value_counts().head(3).to_dict()})
print("\nPER-OBJECTIVE (oracle regret = 0 by definition)")
print(pd.DataFrame(rows).round(2).to_string(index=False))

Ctr = np.maximum.reduce([reg_matrix(TR, o).to_numpy() for o in OBJ])
Cun = np.maximum.reduce([reg_matrix(UN, o).to_numpy() for o in OBJ])
Ctr = pd.DataFrame(Ctr, index=TR.index, columns=NAMES)
Cun = pd.DataFrame(Cun, index=UN.index, columns=NAMES)

bf = Ctr.mean().idxmin()
rf = RandomForestClassifier(**RF).fit(Xtr, Ctr.idxmin(axis=1))
rf_reg = pick(Cun, rf.predict(Xun))
orc = Cun.min(axis=1).to_numpy()

print(f"\nCOMBINED (worst-case regret across 4 objectives), best fixed on train = {bf}")
print(pd.DataFrame({"mean_%": [orc.mean(), rf_reg.mean(), Cun[bf].mean()]},
                   index=["oracle", "RF", f"best-fixed ({bf})"]).round(2).to_string())

diff = Cun[bf].to_numpy() - rf_reg            # positive = RF better
rng = np.random.default_rng(0)
bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
print(f"\nheadroom (best-fixed - oracle): {Cun[bf].mean() - orc.mean():.2f} pp")
print(f"RF gain over best-fixed: {diff.mean():.2f} pp, 95% CI {np.percentile(bs, [2.5, 97.5]).round(2)}")