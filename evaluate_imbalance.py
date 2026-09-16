import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = "lightsched_ml_features.csv"
LABEL_FILE = "lightsched_labeled_dataset_v2.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "num_processes",
    "avg_arrival_time",
    "arrival_time_variance",
    "avg_burst_time",
    "burst_time_variance",
    "avg_priority",
    "avg_io_frequency",
    "arrival_rate"
]

TARGET_COLUMN = "best_scheduler"

CLASS_LABELS = [
    "RR",
    "SJF",
    "Priority",
    "MLFQ"
]


# ============================================================
# LOAD DATA
# ============================================================

features_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)

df = pd.merge(
    features_df,
    labels_df[["workload_id", "best_scheduler"]],
    on="workload_id",
    how="inner"
)

print("=" * 70)
print("IMBALANCE-AWARE MODEL EVALUATION")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(y.value_counts())

print("\nPercentages:")
print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# MODELS
# ============================================================

models = {

    # Original Decision Tree
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    ),

    # NEW: imbalance-aware Decision Tree
    "Balanced Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    # Original Random Forest
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),

    # Logistic Regression
    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            )
        )
    ]),

    # KNN
    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=7
            )
        )
    ]),

    # Naive Bayes
    "Naive Bayes": GaussianNB()
}


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------------

    print(f"\nAccuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall   : {recall_weighted:.4f}")
    print(f"Weighted F1       : {f1_weighted:.4f}")
    print(f"Macro F1          : {f1_macro:.4f}")

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=CLASS_LABELS,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_LABELS,
        columns=CLASS_LABELS
    )

    print(cm_df)

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "f1_macro",
    ascending=False
)


print("\n" + "=" * 70)
print("FINAL IMBALANCE-AWARE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "ml_model_results_imbalance.csv",
    index=False
)

print("\nResults saved to:")
print("ml_model_results_imbalance.csv")