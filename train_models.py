import os
import joblib
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

MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

features_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)

print("=" * 70)
print("DATA LOADING")
print("=" * 70)

print("Features shape:", features_df.shape)
print("Labels shape:", labels_df.shape)


# ============================================================
# MERGE
# ============================================================

df = pd.merge(
    features_df,
    labels_df[["workload_id", "best_scheduler"]],
    on="workload_id",
    how="inner"
)

print("\nMerged dataset shape:", df.shape)


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
    "arrival_rate"
]

TARGET_COLUMN = "best_scheduler"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ============================================================
# DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("FEATURE / TARGET CHECK")
print("=" * 70)

print("\nFeatures:")
print(FEATURE_COLUMNS)

print("\nMissing feature values:")
print(X.isnull().sum().sum())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
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
print("Testing samples:", len(X_test))


# ============================================================
# MAJORITY BASELINE
# ============================================================

print("\n" + "=" * 70)
print("MAJORITY CLASS BASELINE")
print("=" * 70)

majority_class = y_train.value_counts().idxmax()

baseline_predictions = [
    majority_class
] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_predictions
)

baseline_precision = precision_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)

baseline_f1_weighted = f1_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)

baseline_f1_macro = f1_score(
    y_test,
    baseline_predictions,
    average="macro",
    zero_division=0
)

print("Majority class:", majority_class)
print(f"Accuracy          : {baseline_accuracy:.4f}")
print(f"Balanced Accuracy : {baseline_balanced_accuracy:.4f}")
print(f"Weighted Precision: {baseline_precision:.4f}")
print(f"Weighted Recall   : {baseline_recall:.4f}")
print(f"Weighted F1       : {baseline_f1_weighted:.4f}")
print(f"Macro F1          : {baseline_f1_macro:.4f}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),

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

    "Naive Bayes": GaussianNB()
}


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# TRAIN + EVALUATE
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
    # PRINT OVERALL METRICS
    # --------------------------------------------------------

    print(f"Accuracy          : {accuracy:.4f}")
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
            labels=[
                "RR",
                "SJF",
                "Priority",
                "MLFQ"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("Confusion Matrix:")

    class_labels = [
        "RR",
        "SJF",
        "Priority",
        "MLFQ"
    ]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )

    print(cm_df)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    safe_name = model_name.lower().replace(
        " ",
        "_"
    )

    model_path = os.path.join(
        MODEL_DIR,
        safe_name + ".joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved:", model_path)

    # --------------------------------------------------------
    # SAVE RESULT
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
# ADD BASELINE
# ============================================================

results.append({
    "model": "Majority Baseline",
    "accuracy": baseline_accuracy,
    "balanced_accuracy": baseline_balanced_accuracy,
    "precision_weighted": baseline_precision,
    "recall_weighted": baseline_recall,
    "f1_weighted": baseline_f1_weighted,
    "f1_macro": baseline_f1_macro
})


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "f1_macro",
    ascending=False
)

results_df.to_csv(
    "ml_model_results_v2.csv",
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print("ml_model_results_v2.csv")

print("\nTrained models saved in:")
print("models/")