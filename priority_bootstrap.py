import pandas as pd
import numpy as np

d = pd.read_csv("priority_variance_feature_test_results.csv")
a = d[d.variant == "original_8_features"].set_index("workload_id").sort_index()
b = d[d.variant == "plus_priority_variance"].set_index("workload_id").sort_index()


def ci(x, y, n=5000):
    diff = (y.astype(float) - x.astype(float)).to_numpy()
    rng = np.random.default_rng(0)
    m = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return round(diff.mean(), 4), np.percentile(m, [2.5, 97.5]).round(4)


p = a.actual_best_scheduler == "Priority"
print("overall regret diff (pp):", ci(a.regret_pct, b.regret_pct))
print("priority accuracy diff  :", ci(a.correct[p], b.correct[p]))
print("priority regret diff    :", ci(a.regret_pct[p], b.regret_pct[p]))
