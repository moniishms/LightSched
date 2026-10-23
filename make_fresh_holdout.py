import random
import pandas as pd

rng = random.Random(20261010)
N_WORKLOADS = 1000
R = rng.randint


def gen(kind, n):
    if kind == "cpu_bound":
        return ([R(0, 20) for _ in range(n)], [R(60, 100) for _ in range(n)],
                [R(4, 10) for _ in range(n)], [R(0, 2) for _ in range(n)])
    if kind == "interactive":
        return ([R(0, 100) for _ in range(n)], [R(1, 10) for _ in range(n)],
                [R(1, 10) for _ in range(n)], [R(7, 10) for _ in range(n)])
    if kind == "priority_heavy":
        pr = [R(1, 2) if rng.random() < 0.5 else R(9, 10) for _ in range(n)]
        return ([R(0, 30) for _ in range(n)], [R(15, 50) for _ in range(n)], pr,
                [R(0, 4) for _ in range(n)])
    if kind == "sjf_friendly":
        bt = []
        for _ in range(n):
            r = rng.random()
            bt.append(R(1, 5) if r < 0.45 else R(20, 40) if r < 0.75 else R(70, 100))
        return ([R(0, 40) for _ in range(n)], bt, [R(4, 10) for _ in range(n)],
                [R(0, 4) for _ in range(n)])
    if kind == "bursty":
        centers = [R(0, 20), R(30, 50), R(60, 80)]
        arr = [max(0, min(100, rng.choice(centers) + R(-3, 3))) for _ in range(n)]
        return (arr, [R(5, 60) for _ in range(n)], [R(1, 10) for _ in range(n)],
                [R(2, 8) for _ in range(n)])
    return ([R(0, 10) for _ in range(n)], [R(15, 30) for _ in range(n)],        # fairness
            [R(4, 7) for _ in range(n)], [R(5, 10) for _ in range(n)])


KINDS = ["cpu_bound", "interactive", "priority_heavy", "sjf_friendly", "bursty", "fairness"]
rows = []
for wid in range(1, N_WORKLOADS + 1):
    kind, n = rng.choice(KINDS), R(5, 30)
    a, b, p, io = gen(kind, n)
    for pid in range(1, n + 1):
        rows.append(dict(workload_id=wid, pid=pid, arrival_time=a[pid - 1], burst_time=b[pid - 1],
                         priority=p[pid - 1], io_frequency=io[pid - 1], workload_type=kind))

pd.DataFrame(rows).to_csv("lightsched_fresh_holdout.csv", index=False)
print("Saved lightsched_fresh_holdout.csv:", len(rows), "rows,", N_WORKLOADS, "workloads")