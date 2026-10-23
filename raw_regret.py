import numpy as np
import pandas as pd
from schedulers_v2 import round_robin, sjf, priority_scheduling, mlfq

SCHED = {"RR": round_robin, "SJF": sjf, "Priority": priority_scheduling, "MLFQ": mlfq}
TIE_TOL = 0.05   # within 5% of the best turnaround counts as "correct"

raw = pd.read_csv("lightsched_unseen_workload_dataset.csv")
rf = pd.read_csv("lightsched_unseen_evaluation.csv").set_index("workload_id")
rule = pd.read_csv("lightsched_rule_baseline_evaluation.csv").set_index("workload_id")

rows = []
for wid, g in raw.groupby("workload_id"):
    procs = [dict(pid=int(r.pid), arrival_time=int(r.arrival_time),
                  burst_time=int(r.burst_time), priority=int(r.priority),
                  io_frequency=int(r.io_frequency)) for r in g.itertuples()]
    rows.append({"workload_id": wid,
                 **{n: f(procs)[1]["avg_turnaround_time"] for n, f in SCHED.items()}})
T = pd.DataFrame(rows).set_index("workload_id")
best = T.min(axis=1)

policies = {
    "RF (tuned)": rf["predicted_scheduler"],
    "Rule baseline": rule["predicted_scheduler"],
    **{f"Always-{s}": pd.Series(s, index=T.index) for s in SCHED},
}

out = []
for name, pred in policies.items():
    pred = pred.reindex(T.index)
    chosen = pd.Series([T.loc[w, p] for w, p in pred.items()], index=T.index)
    reg = (chosen - best) / best * 100
    out.append({"policy": name,
                "mean_turnaround_regret_%": reg.mean(),
                "median_%": reg.median(),
                "p90_%": reg.quantile(0.9),
                "within_5pct_of_best": (reg <= TIE_TOL * 100).mean()})

print("RAW TURNAROUND REGRET on the 500 unseen workloads")
print(pd.DataFrame(out).round(3).to_string(index=False))

# composite-score regret for always-SJF (the number that was still missing)
S = rf[[f"{s}_score" for s in SCHED]]
oracle = S.max(axis=1)
print("\nComposite-score regret, always-SJF:",
      round(((oracle - S["SJF_score"]) / oracle.where(oracle != 0, 1) * 100).mean(), 2), "%")
print("Composite-score regret, RF        :",
      round((((oracle - pd.Series([S.loc[w, f"{p}_score"] for w, p in rf["predicted_scheduler"].items()], index=S.index))
              / oracle.where(oracle != 0, 1)) * 100).mean(), 2), "%")