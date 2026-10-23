import glob, re
import numpy as np
import pandas as pd

SCHEDS = ["RR", "SJF", "Priority", "MLFQ"]
INITIAL, BATCH, N = 500, 50, 90
PH = [(1, 10, "mixed"), (11, 20, "cpu_bound"), (21, 30, "interactive"),
      (31, 40, "priority_heavy"), (41, 50, "bursty"), (51, 60, "sjf_friendly"),
      (61, 70, "fairness"), (71, 80, "cpu_bound"), (81, 90, "interactive")]


def phase_of(b):
    return next(n for lo, hi, n in PH if lo <= b <= hi)


def summ(df, col):
    s = df.groupby("phase")[col].mean()
    s["ALL"] = df[col].mean()
    return s


acc, reg = {}, {}

# model-based runs
for f in sorted(glob.glob("adaptive_lightsched_predictions*.csv")):
    m = re.search(r"_w(\d+)", f)
    tag = f"adaptive_w{m.group(1)}" if m else "adaptive_all"
    p = pd.read_csv(f)
    p["correct"] = p.actual_best_scheduler == p.predicted_scheduler
    for mode in ["adaptive", "static"]:
        s = p[p["mode"] == mode]
        name = tag if mode == "adaptive" else "static"
        acc[name] = summ(s, "correct")
        reg[name] = summ(s, "regret_pct")

# recent-majority baseline (no model)
lab = (pd.read_csv("lightsched_shifted_labeled_dataset_v2.csv")
         .sort_values("workload_id").reset_index(drop=True))
S = lab[[f"{s}_score" for s in SCHEDS]].to_numpy()
oracle = S.max(axis=1)

for W in (200, 500):
    rows = []
    for b in range(1, N + 1):
        lo = INITIAL + (b - 1) * BATCH
        guess = lab["best_scheduler"].iloc[max(0, lo - W):lo].mode().iloc[0]
        for i in range(lo, lo + BATCH):
            got = S[i, SCHEDS.index(guess)]
            rows.append({
                "phase": phase_of(b),
                "correct": lab["best_scheduler"].iloc[i] == guess,
                "regret_pct": (oracle[i] - got) / oracle[i] * 100 if oracle[i] != 0 else 0.0,
            })
    r = pd.DataFrame(rows)
    acc[f"recentmaj_w{W}"] = summ(r, "correct")
    reg[f"recentmaj_w{W}"] = summ(r, "regret_pct")

order = [c for c in ["static", "recentmaj_w200", "recentmaj_w500", "adaptive_all",
                     "adaptive_w1000", "adaptive_w500", "adaptive_w200"] if c in acc]
A = pd.DataFrame(acc)[order].round(3)
R = pd.DataFrame(reg)[order].round(2)

print("\nACCURACY by phase")
print(A.to_string())
print("\nREGRET % by phase")
print(R.to_string())
A.to_csv("adaptive_summary_accuracy.csv")
R.to_csv("adaptive_summary_regret.csv")

# why is each phase easy or hard? best-scheduler mix per workload type
t = (pd.read_csv("lightsched_shifted_stream.csv")
       .drop_duplicates("workload_id")[["workload_id", "workload_type"]])
m = t.merge(lab[["workload_id", "best_scheduler"]])
print("\nBest-scheduler share by workload type")
print(pd.crosstab(m.workload_type, m.best_scheduler, normalize="index").round(2).to_string())