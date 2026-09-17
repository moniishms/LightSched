"""
Adaptive scheduler-selection experiment for LightSched.

SLIDING-WINDOW VERSION
======================

The adaptive model always trains on the most recent 500 workloads.

At every step:

    1. Train on current 500 workloads
    2. Predict the next 50 workloads
    3. Evaluate predictions using their true labels
    4. Remove the oldest 50 workloads
    5. Add the newly revealed 50 workloads
    6. Retrain using the new 500-workload window

The static baseline is trained once on the initial 500 workloads
and is never retrained.

TRUE LABELS:
    True labels come from the existing V2 label file. These labels were
    generated independently by evaluating all four schedulers using
    schedulers_v2.py.

IMPORTANT:
    The model prediction is NEVER used as the label.

SLIDING WINDOW:
    Initial:
        [1 ... 500]

    After batch 1:
        [51 ... 550]

    After batch 2:
        [101 ... 600]

    After batch 3:
        [151 ... 650]

    ...

Therefore, the adaptive training set always contains exactly 500
workloads.
"""

# ============================================================
# IMPORTS
# ============================================================

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
FEATURE_FILE = "lightsched_shifted_ml_features.csv"

LABEL_FILE = "lightsched_shifted_labeled_dataset_v2.csv"

RESULTS_FILE = "adaptive_lightsched_results.csv"

MODEL_FILE = "models/adaptive_lightsched_final.joblib"

# Sliding-window configuration
INITIAL_TRAIN_SIZE = 500
WINDOW_SIZE = 500
BATCH_SIZE = 50

RANDOM_STATE = 42

# Fixed RF hyperparameters
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

# Stream ordering
USE_SHUFFLED_STREAM = False
SHUFFLE_SEED = 123

# ============================================================
# FEATURES
# ============================================================

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

SCORE_COLUMNS = {
    s: f"{s}_score"
    for s in SCHEDULERS
}


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
    Diagnostic only.

    Checks whether the initial 500 workloads have a very different
    label distribution from the full population.
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

    print(
        f"\nMax per-class distribution difference: "
        f"{max_diff:.4f}"
    )

    if max_diff > 0.15:

        print(
            "WARNING: initial block's label distribution differs "
            "from the population by more than 15 percentage points."
        )

        print(
            "Consider setting USE_SHUFFLED_STREAM = True."
        )

    else:

        print(
            "OK: initial block's label distribution is reasonably "
            "close to the population."
        )


# ============================================================
# BUILD STREAM
# ============================================================

def build_stream(df):

    ids = df["workload_id"].to_numpy().copy()

    if USE_SHUFFLED_STREAM:

        rng = np.random.default_rng(SHUFFLE_SEED)

        rng.shuffle(ids)

        print(
            f"\nStream order: SHUFFLED "
            f"(fixed seed={SHUFFLE_SEED})"
        )

    else:

        ids = np.sort(ids)

        print(
            "\nStream order: NATURAL "
            "(ascending workload_id)"
        )

    return ids


# ============================================================
# TRAIN MODEL
# ============================================================

def fit_model(df, train_ids):

    train_df = df[
        df["workload_id"].isin(train_ids)
    ]

    X_train = train_df[FEATURE_COLUMNS]

    y_train = train_df[TARGET_COLUMN]

    model = RandomForestClassifier(
        **RF_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATE BATCH
# ============================================================

def evaluate_batch(model, df, batch_ids):

    batch_df = df[
        df["workload_id"].isin(batch_ids)
    ].copy()

    X_batch = batch_df[FEATURE_COLUMNS]

    y_true = batch_df[TARGET_COLUMN]

    # --------------------------------------------------------
    # IMPORTANT:
    # Predict BEFORE adding this batch to training data.
    # --------------------------------------------------------

    y_pred = model.predict(X_batch)

    batch_df["predicted_scheduler"] = y_pred

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    correct = int(
        (y_true.values == y_pred).sum()
    )

    pred_counts = {
        f"pred_{s}": int(
            (y_pred == s).sum()
        )
        for s in SCHEDULERS
    }

    # --------------------------------------------------------
    # Regret
    # --------------------------------------------------------

    lightsched_score = batch_df.apply(
        lambda row:
        row[
            SCORE_COLUMNS[
                row["predicted_scheduler"]
            ]
        ],
        axis=1,
    )

    oracle_score = batch_df.apply(
        lambda row:
        row[
            SCORE_COLUMNS[
                row[TARGET_COLUMN]
            ]
        ],
        axis=1,
    )

    score_gap = (
        oracle_score
        - lightsched_score
    )

    regret_pct = np.where(
        oracle_score != 0,
        (score_gap / oracle_score) * 100.0,
        0.0,
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    result = {

        "batch_size":
            len(batch_df),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "correct_predictions":
            correct,

        "avg_lightsched_score":
            float(
                lightsched_score.mean()
            ),

        "avg_oracle_score":
            float(
                oracle_score.mean()
            ),

        "avg_score_gap":
            float(
                score_gap.mean()
            ),

        "avg_regret_pct":
            float(
                np.mean(regret_pct)
            ),
    }

    result.update(pred_counts)

    return result


# ============================================================
# MAIN ADAPTIVE + STATIC LOOP
# ============================================================

def run_experiment(df):

    stream = build_stream(df)

    # --------------------------------------------------------
    # Initial 500 workloads
    # --------------------------------------------------------

    initial_ids = stream[
        :INITIAL_TRAIN_SIZE
    ]

    remaining_ids = stream[
        INITIAL_TRAIN_SIZE:
    ]

    verify_stream_order_unbiased(
        df,
        initial_ids
    )

    n_batches = (
        len(remaining_ids)
        // BATCH_SIZE
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)

    print(
        f"Initial training size : "
        f"{INITIAL_TRAIN_SIZE}"
    )

    print(
        f"Sliding window size   : "
        f"{WINDOW_SIZE}"
    )

    print(
        f"Batch size            : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Number of batches     : "
        f"{n_batches}"
    )

    print(
        f"Total streamed workloads : "
        f"{n_batches * BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # STATIC BASELINE
    #
    # Train ONCE on initial 500.
    # Never retrain.
    # --------------------------------------------------------

    print(
        "\nFitting static baseline model "
        "on initial block..."
    )

    static_model = fit_model(
        df,
        initial_ids
    )

    # --------------------------------------------------------
    # ADAPTIVE MODEL
    #
    # Starts with same initial 500.
    # --------------------------------------------------------

    print(
        "Fitting initial adaptive model "
        "on initial block..."
    )

    adaptive_model = fit_model(
        df,
        initial_ids
    )

    # Current sliding training window
    adaptive_train_ids = list(
        initial_ids
    )

    all_results = []

    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch_id in range(
        1,
        n_batches + 1
    ):

        start = (
            (batch_id - 1)
            * BATCH_SIZE
        )

        end = (
            start
            + BATCH_SIZE
        )

        batch_ids = remaining_ids[
            start:end
        ]

        training_size_before = len(
            adaptive_train_ids
        )

        # ----------------------------------------------------
        # STATIC MODEL
        # ----------------------------------------------------

        static_result = evaluate_batch(
            static_model,
            df,
            batch_ids
        )

        static_result.update(
            {
                "mode": "static",

                "batch_id":
                    batch_id,

                "training_size_before":
                    INITIAL_TRAIN_SIZE,
            }
        )

        all_results.append(
            static_result
        )

        # ----------------------------------------------------
        # ADAPTIVE MODEL
        #
        # IMPORTANT:
        # Prediction happens BEFORE this batch is added.
        # ----------------------------------------------------

        adaptive_result = evaluate_batch(
            adaptive_model,
            df,
            batch_ids
        )

        adaptive_result.update(
            {
                "mode": "adaptive",

                "batch_id":
                    batch_id,

                "training_size_before":
                    training_size_before,
            }
        )

        all_results.append(
            adaptive_result
        )

        # ====================================================
        # SLIDING WINDOW UPDATE
        # ====================================================
        #
        # OLD:
        #     adaptive_train_ids.extend(batch_ids)
        #
        # That creates an EXPANDING window.
        #
        # NEW:
        #     Remove oldest 50
        #     Add newest 50
        #
        # Result:
        #
        # Batch 0:
        #     1 ... 500
        #
        # Batch 1:
        #     51 ... 550
        #
        # Batch 2:
        #     101 ... 600
        #
        # etc.
        # ====================================================

        adaptive_train_ids = (
            adaptive_train_ids[BATCH_SIZE:]
            + batch_ids.tolist()
        )

        # Safety check
        assert len(
            adaptive_train_ids
        ) == WINDOW_SIZE, (
            "Sliding training window "
            "size is incorrect."
        )

        # ----------------------------------------------------
        # Retrain adaptive model using ONLY
        # the new sliding window.
        # ----------------------------------------------------

        adaptive_model = fit_model(
            df,
            adaptive_train_ids
        )

        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        if (
            batch_id % 10 == 0
            or batch_id == n_batches
        ):

            print(
                f"Batch {batch_id:3d}/"
                f"{n_batches}  "
                f"train_size="
                f"{training_size_before:4d}  "
                f"adaptive_acc="
                f"{adaptive_result['accuracy']:.3f}  "
                f"static_acc="
                f"{static_result['accuracy']:.3f}  "
                f"adaptive_macro_f1="
                f"{adaptive_result['macro_f1']:.3f}"
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

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

    ] + [
        f"pred_{s}"
        for s in SCHEDULERS
    ]

    results_df = results_df[
        col_order
    ]

    return (
        results_df,
        adaptive_model
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results_df):

    print("\n" + "=" * 70)
    print(
        "SUMMARY: SLIDING ADAPTIVE VS STATIC"
    )
    print(
        "(mean across all streamed batches)"
    )
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

    print(
        summary.to_string()
    )

    # --------------------------------------------------------
    # First 5 adaptive batches
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "FIRST 5 BATCHES (SLIDING ADAPTIVE)"
    )
    print("=" * 70)

    print(
        results_df[
            results_df["mode"] == "adaptive"
        ]
        .head(5)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Last 5 adaptive batches
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "LAST 5 BATCHES (SLIDING ADAPTIVE)"
    )
    print("=" * 70)

    print(
        results_df[
            results_df["mode"] == "adaptive"
        ]
        .tail(5)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Verify training window stayed fixed
    # --------------------------------------------------------

    adaptive_sizes = results_df[
        results_df["mode"] == "adaptive"
    ]["training_size_before"]

    print("\n" + "=" * 70)
    print("SLIDING WINDOW CHECK")
    print("=" * 70)

    print(
        f"Minimum training size : "
        f"{adaptive_sizes.min()}"
    )

    print(
        f"Maximum training size : "
        f"{adaptive_sizes.max()}"
    )

    if (
        adaptive_sizes.min()
        == WINDOW_SIZE
        and
        adaptive_sizes.max()
        == WINDOW_SIZE
    ):

        print(
            "PASS: adaptive training window "
            f"remained fixed at {WINDOW_SIZE}."
        )

    else:

        print(
            "WARNING: training window size "
            "changed unexpectedly."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    (
        results_df,
        final_adaptive_model
    ) = run_experiment(df)

    # Save results
    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\nResults saved to: "
        f"{RESULTS_FILE}"
    )

    # Save final adaptive model
    joblib.dump(
        final_adaptive_model,
        MODEL_FILE
    )

    print(
        f"Final adaptive model saved to: "
        f"{MODEL_FILE}"
    )

    # Print summary
    print_summary(
        results_df
    )

    print("\n" + "=" * 70)
    print(
        "SLIDING ADAPTIVE EXPERIMENT COMPLETE"
    )
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()