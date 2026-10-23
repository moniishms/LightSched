import numpy as np
import pandas as pd
from schedulers_v2 import round_robin, sjf, priority_scheduling, mlfq

SCHED = {"RR": round_robin, "SJF": sjf, "Priority": priority_scheduling, "MLFQ": mlfq}
OBJ = ["avg_turnaround", "avg_response", "max_waiting", "p95_turnaround"]

raw = pd.read_csv("lightsched_unseen_workload_dataset.csv")
rows = []
for wid, g in raw.groupby("workload_id"):
    procs = [dict(pid=int(r.pid), arrival_time=int(r.arrival_time),
                  burst_time=int(r.burst_time), priority=int(r.priority),
                  io_frequency=int(r.io_frequency)) for r in g.itertuples()]
    row = {"workload_id": wid}
    for n, f in SCHED.items():
        res, s = f(procs)
        row[f"{n}|avg_turnaround"] = s["avg_turnaround_time"]
        row[f"{n}|avg_response"] = s["avg_response_time"]
        row[f"{n}|max_waiting"] = max(r["waiting_time"] for r in res)
        row[f"{n}|p95_turnaround"] = np.percentile([r["turnaround_time"] for r in res], 95)
    rows.append(row)
D = pd.DataFrame(rows).set_index("workload_id")

out = []
for o in OBJ:
    M = pd.DataFrame({n: D[f"{n}|{o}"] for n in SCHED})
    best = M.min(axis=1)
    reg = np.where(best > 0, (M["SJF"] - best) / best.where(best > 0, 1) * 100, 0.0)
    wins = M.idxmin(axis=1).value_counts(normalize=True).reindex(list(SCHED)).fillna(0)
    out.append({"objective": o,
                "SJF_mean_regret_%": reg.mean(),
                "SJF_within_5%_of_best": (reg <= 5).mean(),
                **{f"best_is_{n}": wins[n] for n in SCHED}})

print(pd.DataFrame(out).round(3).to_string(index=False))
D.to_csv("objective_check_raw.csv")