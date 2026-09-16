"""
adaptive_lightsched.py

Adaptive scheduler-selection experiment for LightSched.

============================================================
METHODOLOGY NOTES (read before changing anything)
============================================================

1. TRUE LABELS COME FROM THE EXISTING V2 LABEL FILE, NOT FROM
   RE-RUNNING THE SIMULATOR. This is not a shortcut that weakens the
   experiment: generate_labels_v2.py already ran all four schedulers
   through the validated schedulers_v2.py simulator for every one of
   the 5000 workloads, independent of any ML model. Revealing those
   labels progressively as the "stream" advances is equivalent to
   calling the simulator on demand -- it is not self-training and it
   is not circular (unlike reusing RF predictions as labels would be).

2. HYPERPARAMETER LEAKAGE CAVEAT. The RF hyperparameters below
   (n_estimators=200, max_depth=12, min_samples_leaf=5,
   class_weight='balanced') were originally selected via CV grid
   search over the FULL 5000-row dataset (tune_models.py). That means
   every workload in this adaptive stream was indirectly "seen" during
   model *selection*, even though no individual batch's *training* set
   ever includes future workloads. This is a second-order leakage in
   hyperparameter choice, not in labels or predictions. We keep the
   fixed hyperparameters here and disclose this rather than re-tuning
   on only 500 rows, which would be noisy for a 4-class imbalanced
   problem and isn't really what this experiment is testing.

3. STREAM ORDERING. workload_generation.py assigns
   `workload_type = random.choice(workload_types)` independently for
   EVERY workload_id in a plain 1..5000 loop -- workload types are not
   grouped by ID range. Because of this, IDs 1-500 are not expected to
   be skewed toward particular workload types, so natural ID order
   (1, 2, 3, ...) is used as the stream order below. This is verified
   empirically at runtime (see verify_stream_order_unbiased) rather
   than just assumed. If you ever regenerate the raw dataset with a
   different (grouped) generation scheme, flip USE_SHUFFLED_STREAM to
   True to get a fixed-seed shuffle instead -- the diagnostic will
   warn you if it detects skew.

4. STATIC VS ADAPTIVE COMPARISON. Both use the identical stream and
   identical batch boundaries so the comparison is apples-to-apples.
   The static model is fit once on the initial 500 workloads and never
   retrained. The adaptive model retrains from scratch (same fixed
   hyperparameters, not warm-started) after each batch's labels are
   revealed.

5. NO LABEL LEAKAGE. At every step, a batch's RF predictions are
   computed and scored BEFORE that batch's true labels are added to
   the adaptive training pool. Order within the loop matters: predict
   first, evaluate second, append third.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = "lightsched_ml_features.csv"
LABEL_FILE = "lightsched_labeled_dataset_v2.csv"

RESULTS_FILE = "adaptive_lightsched_results.csv"
MODEL_FILE = "models/adaptive_lightsched_final.joblib"

INITIAL_TRAIN_SIZE = 500
BATCH_SIZE = 50

RANDOM_STATE = 42

# Fixed RF hyperparameters (see methodology note 2 above).
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

USE_SHUFFLED_STREAM = False
SHUFFLE_SEED = 123  # distinct from the data-generation seed (42/9999)

FEATURE_COLUMNS = [
    "num_processes",
    "avg_arrival_time",
    "arrival_time_variance",
    "avg_burst_time",
    "burst_time_variance",
    "avg_priority",
    "avg_io_frequency",
    "arrival_rate",
]

TARGET_COLUMN = "best_scheduler"

SCHEDULERS = ["RR", "SJF", "Priority", "MLFQ"]

SCORE_COLUMNS = {s: f"{s}_score" for s in SCHEDULERS}


# ============================================================
# LOAD + MERGE
# ============================================================

def load_data():

    features_df = pd.read_csv(FEATURE_FILE)

    labels_df = pd.read_csv(LABEL_FILE)

    keep_cols = (
        ["workload_id", TARGET_COLUMN]
        + list(SCORE_COLUMNS.values())
    )

    df = pd.merge(
        features_df,
        labels_df[keep_cols],
        on="workload_id",
        how="inner",
    )

    df = df.sort_values("workload_id").reset_index(drop=True)

    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)

    print(f"Merged dataset shape : {df.shape}")

    print(
        "Target distribution  :\n"
        f"{df[TARGET_COLUMN].value_counts().to_string()}"
    )

    return df


# ============================================================
# STREAM ORDER + DIAGNOSTIC
# ============================================================

def verify_stream_order_unbiased(df, initial_ids):
    """
    Diagnostic only (does not change behaviour unless
    USE_SHUFFLED_STREAM is toggled). We don't have workload_type in
    the merged feature/label frame (it's deliberately excluded as an
    ML feature to avoid leakage), so this checks the one thing we can
    check directly: whether the *label* distribution in the initial
    block looks wildly different from the population as a whole. A
    large skew here would be a proxy signal for a possible ordering
    problem even without loading the raw per-process dataset.
    """

    initial_dist = (
        df[df["workload_id"].isin(initial_ids)][TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
    )

    overall_dist = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
    )

    diff = (initial_dist - overall_dist).abs()

    max_diff = diff.max() if len(diff) else 0.0

    print("\n" + "=" * 70)
    print("STREAM ORDER DIAGNOSTIC")
    print("=" * 70)

    print("Label distribution, first", len(initial_ids), "IDs:")
    print(initial_dist.round(4).to_string())

    print("\nLabel distribution, full population:")
    print(overall_dist.round(4).to_string())

    print(f"\nMax per-class distribution difference: {max_diff:.4f}")

    if max_diff > 0.15:
        print(
            "WARNING: initial block's label distribution differs from "
            "the population by more than 15 percentage points on at "
            "least one class. Consider setting USE_SHUFFLED_STREAM = "
            "True."
        )
    else:
        print(
            "OK: initial block's label distribution is reasonably "
            "close to the population. Natural ID order is being used "
            "as documented in the module docstring."
        )


def build_stream(df):

    ids = df["workload_id"].to_numpy().copy()

    if USE_SHUFFLED_STREAM:
        rng = np.random.default_rng(SHUFFLE_SEED)
        rng.shuffle(ids)
        print(
            f"\nStream order: SHUFFLED (fixed seed={SHUFFLE_SEED})"
        )
    else:
        ids = np.sort(ids)
        print("\nStream order: NATURAL (ascending workload_id)")

    return ids


# ============================================================
# TRAIN / PREDICT HELPERS
# ============================================================

def fit_model(df, train_ids):

    train_df = df[df["workload_id"].isin(train_ids)]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    model = RandomForestClassifier(**RF_PARAMS)

    model.fit(X_train, y_train)

    return model


def evaluate_batch(model, df, batch_ids):

    batch_df = df[df["workload_id"].isin(batch_ids)].copy()

    X_batch = batch_df[FEATURE_COLUMNS]
    y_true = batch_df[TARGET_COLUMN]

    y_pred = model.predict(X_batch)

    batch_df["predicted_scheduler"] = y_pred

    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    correct = int((y_true.values == y_pred).sum())

    pred_counts = {
        f"pred_{s}": int((y_pred == s).sum()) for s in SCHEDULERS
    }

    # --------------------------------------------------------
    # Regret, computed straight from the stored composite scores
    # in lightsched_labeled_dataset_v2.csv -- no simulator call
    # needed (see methodology note 1).
    # --------------------------------------------------------

    lightsched_score = batch_df.apply(
        lambda row: row[SCORE_COLUMNS[row["predicted_scheduler"]]],
        axis=1,
    )

    oracle_score = batch_df.apply(
        lambda row: row[SCORE_COLUMNS[row[TARGET_COLUMN]]],
        axis=1,
    )

    score_gap = oracle_score - lightsched_score

    # Avoid divide-by-zero; oracle_score should essentially never be
    # exactly 0 given the composite-score construction, but guard
    # anyway.
    regret_pct = np.where(
        oracle_score != 0,
        (score_gap / oracle_score) * 100.0,
        0.0,
    )

    result = {
        "batch_size": len(batch_df),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "correct_predictions": correct,
        "avg_lightsched_score": float(lightsched_score.mean()),
        "avg_oracle_score": float(oracle_score.mean()),
        "avg_score_gap": float(score_gap.mean()),
        "avg_regret_pct": float(np.mean(regret_pct)),
    }

    result.update(pred_counts)

    return result


# ============================================================
# MAIN ADAPTIVE + STATIC LOOP
# ============================================================

def run_experiment(df):

    stream = build_stream(df)

    initial_ids = stream[:INITIAL_TRAIN_SIZE]
    remaining_ids = stream[INITIAL_TRAIN_SIZE:]

    verify_stream_order_unbiased(df, initial_ids)

    n_batches = len(remaining_ids) // BATCH_SIZE

    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)

    print(f"Initial training size : {INITIAL_TRAIN_SIZE}")
    print(f"Batch size            : {BATCH_SIZE}")
    print(f"Number of batches     : {n_batches}")
    print(
        f"Total streamed workloads : "
        f"{n_batches * BATCH_SIZE} "
        f"(of {len(remaining_ids)} available after initial block)"
    )

    # --------------------------------------------------------
    # Static baseline: fit once on the initial block, freeze.
    # --------------------------------------------------------

    print("\nFitting static baseline model on initial block...")

    static_model = fit_model(df, initial_ids)

    # --------------------------------------------------------
    # Adaptive: starts identical to static, then retrains after
    # every batch.
    # --------------------------------------------------------

    print("Fitting initial adaptive model on initial block...")

    adaptive_model = fit_model(df, initial_ids)

    adaptive_train_ids = list(initial_ids)

    all_results = []

    for batch_id in range(1, n_batches + 1):

        start = (batch_id - 1) * BATCH_SIZE
        end = start + BATCH_SIZE

        batch_ids = remaining_ids[start:end]

        training_size_before = len(adaptive_train_ids)

        # ---- STATIC: predict with the frozen model -------------
        static_result = evaluate_batch(
            static_model, df, batch_ids
        )
        static_result.update(
            {
                "mode": "static",
                "batch_id": batch_id,
                "training_size_before": INITIAL_TRAIN_SIZE,
            }
        )
        all_results.append(static_result)

        # ---- ADAPTIVE: predict BEFORE revealing labels ----------
        adaptive_result = evaluate_batch(
            adaptive_model, df, batch_ids
        )
        adaptive_result.update(
            {
                "mode": "adaptive",
                "batch_id": batch_id,
                "training_size_before": training_size_before,
            }
        )
        all_results.append(adaptive_result)

        # ---- Now reveal this batch's true labels and retrain ----
        adaptive_train_ids.extend(batch_ids.tolist())

        adaptive_model = fit_model(df, adaptive_train_ids)

        if batch_id % 10 == 0 or batch_id == n_batches:
            print(
                f"Batch {batch_id:3d}/{n_batches}  "
                f"train_size={training_size_before:4d}  "
                f"adaptive_acc={adaptive_result['accuracy']:.3f}  "
                f"static_acc={static_result['accuracy']:.3f}  "
                f"adaptive_macro_f1={adaptive_result['macro_f1']:.3f}"
            )

    results_df = pd.DataFrame(all_results)

    col_order = [
        "mode",
        "batch_id",
        "training_size_before",
        "batch_size",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "correct_predictions",
        "avg_lightsched_score",
        "avg_oracle_score",
        "avg_score_gap",
        "avg_regret_pct",
    ] + [f"pred_{s}" for s in SCHEDULERS]

    results_df = results_df[col_order]

    return results_df, adaptive_model


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results_df):

    print("\n" + "=" * 70)
    print("SUMMARY: ADAPTIVE VS STATIC (mean across all batches)")
    print("=" * 70)

    summary = (
        results_df
        .groupby("mode")[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "avg_regret_pct",
            ]
        ]
        .mean()
        .round(4)
    )

    print(summary.to_string())

    print("\n" + "=" * 70)
    print("FIRST 5 BATCHES (adaptive)")
    print("=" * 70)

    print(
        results_df[results_df["mode"] == "adaptive"]
        .head(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("LAST 5 BATCHES (adaptive)")
    print("=" * 70)

    print(
        results_df[results_df["mode"] == "adaptive"]
        .tail(5)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    results_df, final_adaptive_model = run_experiment(df)

    results_df.to_csv(RESULTS_FILE, index=False)

    print(f"\nResults saved to: {RESULTS_FILE}")

    joblib.dump(final_adaptive_model, MODEL_FILE)

    print(f"Final adaptive model saved to: {MODEL_FILE}")

    print_summary(results_df)

    print("\n" + "=" * 70)
    print("ADAPTIVE EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()