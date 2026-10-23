import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

FEATURES = ["num_processes", "avg_arrival_time", "arrival_time_variance",
            "avg_burst_time", "burst_time_variance", "avg_priority",
            "avg_io_frequency", "arrival_rate"]
RF_PARAMS = dict(n_estimators=200, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=-1)

df = pd.merge(pd.read_csv("lightsched_ml_features.csv"),
              pd.read_csv("lightsched_labeled_dataset_v2.csv")[["workload_id", "best_scheduler"]],
              on="workload_id")
X, y = df[FEATURES], df["best_scheduler"]
print("Majority-class accuracy:", round(y.value_counts(normalize=True).max(), 4))


def fit_eval(Xtr, ytr, Xte, yte, seed):
    m = RandomForestClassifier(**RF_PARAMS, random_state=seed).fit(Xtr, ytr)
    p = m.predict(Xte)
    return {"accuracy": accuracy_score(yte, p),
            "balanced_acc": balanced_accuracy_score(yte, p),
            "macro_f1": f1_score(yte, p, average="macro", zero_division=0)}


# ---- Exp 1: test-size sweep (20 random stratified splits each) ----
rows = []
for ts in [0.1, 0.2, 0.3, 0.4]:
    for seed in range(20):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=seed, stratify=y)
        rows.append({"test_size": ts, "n_test": len(Xte), **fit_eval(Xtr, ytr, Xte, yte, seed)})
exp1 = pd.DataFrame(rows).groupby("test_size").agg(["mean", "std"]).round(4)
print("\nEXP 1: test-size sweep (mean, std over 20 splits)\n", exp1.to_string())
exp1.to_csv("split_exp1_test_size.csv")

# ---- Exp 2: learning curve on a fixed 80/20 split ----
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rows = []
for frac in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for seed in range(10):
        if frac < 1.0:
            Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=frac, random_state=seed, stratify=ytr)
        else:
            Xs, ys = Xtr, ytr
        rows.append({"train_frac": frac, "n_train": len(Xs), **fit_eval(Xs, ys, Xte, yte, seed)})
exp2 = pd.DataFrame(rows).groupby(["train_frac", "n_train"]).agg(["mean", "std"]).round(4)
print("\nEXP 2: learning curve (does more training data still help?)\n", exp2.to_string())
exp2.to_csv("split_exp2_learning_curve.csv")

# ---- Exp 3: sampling error of accuracy vs. number of test workloads ----
m = RandomForestClassifier(**RF_PARAMS, random_state=42).fit(Xtr, ytr)
correct = (m.predict(Xte) == yte.values).astype(float)
rng = np.random.default_rng(0)
rows = []
for n in [50, 100, 250, 500, 1000, 2000]:
    draws = [rng.choice(correct, size=n, replace=True).mean() for _ in range(2000)]
    rows.append({"n_test": n,
                 "empirical_halfwidth_95": 1.96 * np.std(draws),
                 "worst_case_halfwidth": 1.96 * np.sqrt(0.25 / n)})
exp3 = pd.DataFrame(rows).round(4)
print("\nEXP 3: 95% CI half-width of accuracy vs test-set size\n", exp3.to_string(index=False))
exp3.to_csv("split_exp3_ci_vs_n.csv", index=False)