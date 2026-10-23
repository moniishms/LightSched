import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

NAMES = ["RR", "SJF", "Priority", "MLFQ"]
OBJ = ["avg_turnaround", "avg_response", "max_waiting", "p95_turnaround"]
FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]
RF = dict(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)

TR = pd.read_csv("objective_train_raw.csv", index_col="workload_id")
UN = pd.read_csv("objective_check_raw.csv", index_col="workload_id")


def combined_regret(D):
    """worst-case (max over objectives) regret % for each scheduler, per workload"""
    per_obj = []
    for o in OBJ:
        M = pd.DataFrame({n: D[f"{n}|{o}"] for n in NAMES})
        best = M.min(axis=1)
        per_obj.append(M.sub(best, axis=0).div(np.maximum(best, 1.0), axis=0) * 100)
    return pd.concat(per_obj).groupby(level=0).max()      # workloads x schedulers


def feats(raw_file):
    rows = {}
    for wid, g in pd.read_csv(raw_file).groupby("workload_id"):
        n = len(g); span = g.arrival_time.max() - g.arrival_time.min() + 1
        rows[wid] = {"num_processes": n, "avg_arrival_time": g.arrival_time.mean(),
                     "arrival_time_variance": g.arrival_time.var(ddof=0),
                     "avg_burst_time": g.burst_time.mean(),
                     "burst_time_variance": g.burst_time.var(ddof=0),
                     "avg_priority": g.priority.mean(), "avg_io_frequency": g.io_frequency.mean(),
                     "arrival_rate": n / span}
    return pd.DataFrame.from_dict(rows, orient="index")


Rtr, Run = combined_regret(TR), combined_regret(UN)
Xtr = pd.read_csv("lightsched_ml_features.csv").set_index("workload_id")[FEATURES].loc[Rtr.index]
Xun = feats("lightsched_unseen_workload_dataset.csv").loc[Run.index][FEATURES]

best_fixed = Rtr.mean().idxmin()
rf = RandomForestClassifier(**RF).fit(Xtr, Rtr.idxmin(axis=1))
pred = pd.Series(rf.predict(Xun), index=Run.index)

pick = lambda choice: np.array([Run.loc[w, c] for w, c in choice.items()])
res = {
    "Per-workload oracle": Run.min(axis=1).to_numpy(),
    "RF": pick(pred),
    f"Best-fixed ({best_fixed})": Run[best_fixed].to_numpy(),
    "Always-SJF": Run["SJF"].to_numpy(),
}
print("WORST-CASE REGRET ACROSS 4 OBJECTIVES, 500 unseen workloads (lower is better)")
print(pd.DataFrame({k: {"mean_%": v.mean(), "median_%": np.median(v), "p90_%": np.percentile(v, 90)}
                    for k, v in res.items()}).T.round(2).to_string())
print("\nHeadroom (best-fixed minus oracle):",
      round(res[f"Best-fixed ({best_fixed})"].mean() - res["Per-workload oracle"].mean(), 2), "pp")
print("RF captures:", round((res[f"Best-fixed ({best_fixed})"].mean() - res["RF"].mean()), 2), "pp of it")