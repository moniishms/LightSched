"""
results_summary.py

Consolidates every experiment run so far into one master comparison
table:

  1. RF (official, tuned)      -- from lightsched_performance_evaluation.csv
                                   (trained on 80% split via tune_models.py's
                                   GridSearchCV)
  2. RF (refit, 8 features)    -- from priority_variance_feature_test_results.csv,
                                   variant == 'original_8_features'
                                   (same hyperparams, but trained on the FULL
                                   5000-row population, not an 80% split --
                                   see caveat printed below)
  3. RF (refit, 9 features)    -- same file, variant == 'plus_priority_variance'
  4. Rule-based baseline       -- from lightsched_rule_baseline_evaluation.csv
  5. Adaptive (mean of 90 batches) -- from adaptive_lightsched_results.csv,
                                   mode == 'adaptive'
  6. Static (mean of 90 batches)   -- same file, mode == 'static'

All metrics are RECOMPUTED from the per-workload prediction columns in
each file (not copy-pasted from earlier console output), so this script
is the single source of truth going forward -- rerun it any time a
result file changes and the table updates itself.

Every file load is wrapped so a missing file produces a clear warning
and an empty row rather than crashing -- you don't need every artifact
present to get a partial table.
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

OUTPUT_FILE = "lightsched_results_summary.csv"


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [skipped] {path} not found")
        return None


def metrics_from_predictions(y_true, y_pred, regret_series=None):

    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "n_evaluated": len(y_true),
    }

    if regret_series is not None:
        row["avg_regret_pct"] = regret_series.mean()
    else:
        row["avg_regret_pct"] = float("nan")

    return row


def priority_slice_metrics(y_true, y_pred, regret_series):

    mask = y_true == "Priority"

    if mask.sum() == 0:
        return {
            "priority_accuracy": float("nan"),
            "priority_avg_regret_pct": float("nan"),
            "n_priority": 0,
        }

    return {
        "priority_accuracy": accuracy_score(y_true[mask], y_pred[mask]),
        "priority_avg_regret_pct": (
            regret_series[mask].mean() if regret_series is not None else float("nan")
        ),
        "n_priority": int(mask.sum()),
    }


def main():

    print("=" * 70)
    print("LOADING RESULT FILES")
    print("=" * 70)

    overall_rows = {}
    priority_rows = {}

    # --------------------------------------------------------
    # 1. RF official (tuned, 80% split)
    # --------------------------------------------------------
    df = safe_read_csv("lightsched_performance_evaluation.csv")
    if df is not None:
        y_true = df["oracle_scheduler"]
        y_pred = df["predicted_scheduler"]
        regret = df["regret_percentage"]
        overall_rows["RF_official_tuned"] = metrics_from_predictions(
            y_true, y_pred, regret
        )
        priority_rows["RF_official_tuned"] = priority_slice_metrics(
            y_true, y_pred, regret
        )

    # --------------------------------------------------------
    # 2 & 3. RF refit (8 features) and RF refit (+priority_variance)
    # --------------------------------------------------------
    df = safe_read_csv("priority_variance_feature_test_results.csv")
    if df is not None:
        for variant, label in [
            ("original_8_features", "RF_refit_8_features"),
            ("plus_priority_variance", "RF_refit_9_features"),
        ]:
            subset = df[df["variant"] == variant]
            if len(subset) == 0:
                continue
            y_true = subset["actual_best_scheduler"]
            y_pred = subset["predicted_scheduler"]
            regret = subset["regret_pct"]
            overall_rows[label] = metrics_from_predictions(y_true, y_pred, regret)
            priority_rows[label] = priority_slice_metrics(y_true, y_pred, regret)

    # --------------------------------------------------------
    # 4. Rule-based baseline
    # --------------------------------------------------------
    df = safe_read_csv("lightsched_rule_baseline_evaluation.csv")
    if df is not None:
        y_true = df["actual_best_scheduler"]
        y_pred = df["predicted_scheduler"]
        regret = df["regret_pct"]
        overall_rows["Rule_based_baseline"] = metrics_from_predictions(
            y_true, y_pred, regret
        )
        priority_rows["Rule_based_baseline"] = priority_slice_metrics(
            y_true, y_pred, regret
        )

    # --------------------------------------------------------
    # 5 & 6. Adaptive / static (mean across all 90 batches)
    #    Note: these are per-batch aggregate metrics, not per-workload
    #    predictions, so there's no priority-class slice available at
    #    this granularity -- the adaptive script only logged prediction
    #    COUNTS per class per batch, not per-workload correctness.
    # --------------------------------------------------------
    df = safe_read_csv("adaptive_lightsched_results.csv")
    if df is not None:
        for mode, label in [("adaptive", "Adaptive_mean_90_batches"),
                             ("static", "Static_mean_90_batches")]:
            subset = df[df["mode"] == mode]
            if len(subset) == 0:
                continue
            overall_rows[label] = {
                "accuracy": subset["accuracy"].mean(),
                "balanced_accuracy": subset["balanced_accuracy"].mean(),
                "macro_f1": subset["macro_f1"].mean(),
                "avg_regret_pct": subset["avg_regret_pct"].mean(),
                "n_evaluated": int(subset["batch_size"].sum()),
            }

    # --------------------------------------------------------
    # Build and print tables
    # --------------------------------------------------------

    if not overall_rows:
        print("\nNo result files found. Run the earlier experiment "
              "scripts first, then rerun this script.")
        return

    overall_df = pd.DataFrame(overall_rows).T
    overall_df = overall_df[
        ["accuracy", "balanced_accuracy", "macro_f1", "avg_regret_pct", "n_evaluated"]
    ]

    print("\n" + "=" * 70)
    print("MASTER COMPARISON TABLE (overall)")
    print("=" * 70)
    print(overall_df.round(4).to_string())

    if priority_rows:
        priority_df = pd.DataFrame(priority_rows).T
        priority_df = priority_df[
            ["priority_accuracy", "priority_avg_regret_pct", "n_priority"]
        ]

        print("\n" + "=" * 70)
        print("PRIORITY-CLASS SLICE (where available)")
        print("=" * 70)
        print(priority_df.round(4).to_string())

    overall_df.to_csv(OUTPUT_FILE)
    print(f"\nSaved master table to: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Caveats -- printed every run so they're never forgotten
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CAVEATS (read before comparing rows directly)")
    print("=" * 70)

    print(
        "\n1. RF_official_tuned was trained on an 80% split via "
        "GridSearchCV (tune_models.py). RF_refit_8_features uses the "
        "same hyperparameters but was trained on the FULL 5000-row "
        "population (no held-out split). A few points of difference "
        "between these two rows is expected from that alone -- it is "
        "not evidence the feature set changed anything."
    )

    print(
        "\n2. Adaptive/Static rows are MEANS of 90 batch-level metrics, "
        "each batch scored on only 50 workloads -- these are noisier "
        "per-metric than the other rows, which are each scored on the "
        "full 500-workload unseen set in one pass. Don't over-read "
        "small differences against the single-pass rows."
    )

    print(
        "\n3. No priority-class slice is available for the Adaptive/"
        "Static rows: the adaptive experiment logged prediction COUNTS "
        "per class per batch, not per-workload correctness, so a "
        "priority-only accuracy/regret can't be recovered from "
        "adaptive_lightsched_results.csv without rerunning that "
        "experiment with per-workload logging added."
    )

    print("\n" + "=" * 70)
    print("SUMMARY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()