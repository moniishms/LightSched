import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from schedulers_extra import CANDIDATES

NAMES = list(CANDIDATES)
OBJ = ["avg_turnaround", "avg_response", "max_waiting", "p95_turnaround"]
FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]
TR = pd.read_csv("extra_train_raw.csv", index_col="workload_id")
HO = pd.read_csv("fresh_raw.csv", index_col="workload_id")


def feats(raw, ids):
    rows = {}
    for wid, g in raw.groupby("workload_id"):
        n = len(g)
        span = g.arrival_time.max() - g.arrival_time.min() + 1
        rows[wid] = {"num_processes": n, "avg_arrival_time": g.arrival_time.mean(),
                     "arrival_time_variance": g.arrival_time.var(ddof=0),
                     "avg_burst_time": g.burst_time.mean(),
                     "burst_time_variance": g.burst_time.var(ddof=0),
                     "avg_priority": g.priority.mean(),
                     "avg_io_frequency": g.io_frequency.mean(),
                     "arrival_rate": n / span}
    return pd.DataFrame.from_dict(rows, orient="index").loc[ids][FEATURES]


def parts(D):
    reg, rank = [], []
    for o in OBJ:
        M = pd.DataFrame({n: D[f"{n}|{o}"] for n in NAMES})
        best = M.min(axis=1)
        reg.append((M.sub(best, axis=0).div(np.maximum(best, 1.0), axis=0) * 100).to_numpy())
        rank.append(M.rank(axis=1).to_numpy())
    return reg, rank


AGG = {"max regret (primary)": lambda r, k: np.maximum.reduce(r),
       "mean regret": lambda r, k: np.mean(r, axis=0),
       "rank sum": lambda r, k: np.sum(k, axis=0)}

Xtr = pd.read_csv("lightsched_ml_features.csv").set_index("workload_id")[FEATURES].loc[TR.index]
Xho = feats(pd.read_csv("lightsched_fresh_holdout.csv"), HO.index)
rtr, ktr = parts(TR)
rho, kho = parts(HO)
rng = np.random.default_rng(0)

print(f"{'aggregation':22s} {'oracle':>8s} {'best-fixed':>11s} {'RF-cost':>9s} {'gain':>7s}  95% CI              % headroom")
for name, f in AGG.items():
    Ctr = pd.DataFrame(f(rtr, ktr), index=TR.index, columns=NAMES)
    Cho = pd.DataFrame(f(rho, kho), index=HO.index, columns=NAMES)
    bf = Ctr.mean().idxmin()
    m = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                              random_state=42, n_jobs=-1).fit(Xtr, Ctr)
    choice = np.array(NAMES)[m.predict(Xho).argmin(axis=1)]
    got = np.array([Cho.loc[w, c] for w, c in zip(Cho.index, choice)])
    d = Cho[bf].to_numpy() - got
    ci = np.percentile([rng.choice(d, len(d)).mean() for _ in range(5000)], [2.5, 97.5])
    head = Cho[bf].mean() - Cho.min(axis=1).mean()
    print(f"{name:22s} {Cho.min(axis=1).mean():8.2f} {Cho[bf].mean():11.2f} {got.mean():9.2f} "
          f"{d.mean():7.2f}  [{ci[0]:6.2f},{ci[1]:6.2f}]  {d.mean() / head * 100:5.0f}%")