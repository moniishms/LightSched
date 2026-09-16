"""
analyze_regret_by_class.py

Breaks the RF's aggregate 8.28% average regret (from
lightsched_performance_evaluation.csv) down by the TRUE best
scheduler for each workload. The aggregate number hides an important
question: is regret roughly flat across classes, or is it low on
SJF-true workloads (which the model predicts well) and much higher on
RR/Priority/MLFQ-true workloads (which it predicts poorly)? Given
Priority recall was already known to be weak (13/52 on the unseen
set) and the rule-based baseline run just confirmed Priority is hard
to call from these features at all, this checks how expensive that
weakness actually is in performance terms, not just accuracy terms.

Reuses lightsched_performance_evaluation.csv as-is -- no simulator
re-run needed. That file already has, per workload: predicted_scheduler,
oracle_scheduler (= true best), lightsched_score, oracle_score,
score_gap, regret_percentage, prediction_correct.
"""

import pandas as pd

INPUT_FILE = "lightsched_performance_evaluation.csv"
OUTPUT_FILE = "lightsched_regret_by_class.csv"

CLASS_ORDER = ["RR", "SJF", "Priority", "MLFQ"]


def main():

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("REGRET BREAKDOWN BY TRUE (ORACLE) BEST SCHEDULER")
    print("=" * 70)

    print(f"\nLoaded {len(df)} evaluated workloads from {INPUT_FILE}")

    # --------------------------------------------------------
    # Per-class summary
    # --------------------------------------------------------

    summary = (
        df.groupby("oracle_scheduler")
        .agg(
            n_workloads=("workload_id", "count"),
            accuracy=("prediction_correct", "mean"),
            avg_regret_pct=("regret_percentage", "mean"),
            median_regret_pct=("regret_percentage", "median"),
            max_regret_pct=("regret_percentage", "max"),
            avg_score_gap=("score_gap", "mean"),
        )
        .reindex(CLASS_ORDER)
        .round(4)
    )

    print("\nPer-class summary:")
    print(summary.to_string())

    # --------------------------------------------------------
    # Overall regret, for reference / sanity check against the
    # 8.28% figure from the original end-to-end evaluation run
    # --------------------------------------------------------

    overall_regret = df["regret_percentage"].mean()

    print(f"\nOverall average regret (sanity check): {overall_regret:.2f}%")

    # --------------------------------------------------------
    # How much does each class CONTRIBUTE to total regret?
    # (share of workloads x that class's average regret, relative
    # to the overall average -- shows where the regret "budget" is
    # actually being spent)
    # --------------------------------------------------------

    contribution = summary.copy()
    contribution["share_of_workloads"] = (
        contribution["n_workloads"] / len(df)
    )
    contribution["contribution_to_total_regret"] = (
        contribution["share_of_workloads"] * contribution["avg_regret_pct"]
    )
    contribution["pct_of_total_regret_budget"] = (
        contribution["contribution_to_total_regret"]
        / contribution["contribution_to_total_regret"].sum()
        * 100.0
    )

    print("\nRegret contribution by class (where the regret budget goes):")
    print(
        contribution[
            [
                "n_workloads",
                "share_of_workloads",
                "avg_regret_pct",
                "pct_of_total_regret_budget",
            ]
        ].round(3).to_string()
    )

    # --------------------------------------------------------
    # Prediction distribution WITHIN each true class -- what does
    # the model actually pick when RR/SJF/Priority/MLFQ is truly
    # best? (this is the confusion matrix restated as row-normalized
    # percentages, which is easier to read for this purpose)
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WHAT DOES THE MODEL PREDICT, GIVEN THE TRUE BEST SCHEDULER?")
    print("=" * 70)

    for true_class in CLASS_ORDER:

        subset = df[df["oracle_scheduler"] == true_class]

        if len(subset) == 0:
            continue

        pred_dist = (
            subset["predicted_scheduler"]
            .value_counts(normalize=True)
            .reindex(CLASS_ORDER)
            .fillna(0.0)
            .mul(100)
            .round(1)
        )

        print(f"\nTrue best = {true_class}  (n={len(subset)}):")
        print(pred_dist.to_string())

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary.to_csv(OUTPUT_FILE)

    print(f"\nSaved per-class summary to: {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("REGRET-BY-CLASS ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()