import numpy as np
import pandas as pd
from schedulers_v2 import round_robin, sjf, priority_scheduling, mlfq

SCHED = {"RR": round_robin, "SJF": sjf, "Priority": priority_scheduling, "MLFQ": mlfq}
raw = pd.read_csv("lightsched_shifted_stream.csv")
types = raw.drop_duplicates("workload_id")[["workload_id", "workload_type"]]

rows = []
for wtype, g in types.groupby("workload_type"):
    for wid in g["workload_id"].sample(min(60, len(g)), random_state=0):
        grp = raw[raw.workload_id == wid]
        procs = [dict(pid=int(r.pid), arrival_time=int(r.arrival_time),
                      burst_time=int(r.burst_time), priority=int(r.priority),
                      io_frequency=int(r.io_frequency)) for r in grp.itertuples()]
        m = {n: f(procs)[1] for n, f in SCHED.items()}
        tat = np.array([m[n]["avg_turnaround_time"] for n in SCHED])
        wt = np.array([m[n]["avg_waiting_time"] for n in SCHED])
        rows.append({
            "workload_type": wtype,
            "turnaround_spread_pct": (tat.max() - tat.min()) / tat.mean() * 100,
            "waiting_spread_vs_turnaround_pct": (wt.max() - wt.min()) / tat.mean() * 100,
        })

out = pd.DataFrame(rows).groupby("workload_type").median().round(2)
print("Median raw difference between best and worst scheduler, by workload type:")
print(out.to_string())