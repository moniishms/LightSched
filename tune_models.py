import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

FEATURE_FILE = "lightsched_ml_features.csv"
LABEL_FILE = "lightsched_labeled_dataset_v2.csv"

features_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)

# ============================================================
# 2. MERGE FEATURES + LABELS
# ============================================================

df = pd.merge(
    features_df,
    labels_df[["workload_id", "best_scheduler"]],
    on="workload_id",
    how="inner"
)

print("Merged dataset shape:", df.shape)

# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

feature_columns = [
    "num_processes",
    "avg_arrival_time",
    "arrival_time_variance",
    "avg_burst_time",
    "burst_time_variance",
    "avg_priority",
    "avg_io_frequency",
    "arrival_rate"
]

X = df[feature_columns]
y = df["best_scheduler"]

print("\nClass distribution:")
print(y.value_counts())

# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# IMPORTANT:
# f1_macro is a multiclass scorer and works with
# labels such as RR, SJF, Priority and MLFQ.
scoring = "f1_macro"

# ============================================================
# 6. MODEL DEFINITIONS + PARAMETER GRIDS
# ============================================================

models = {

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),

        {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 4, 5, 6, 8, None],
            "min_samples_leaf": [5, 10, 15, 20],
            "class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),

        {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 8, 12],
            "min_samples_leaf": [1, 5, 10],
            "class_weight": [None, "balanced"]
        }
    ),

    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]),

        {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver": ["lbfgs"]
        }
    ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier())
        ]),

        {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        }
    )
}

# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

results = []

best_models = {}

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

for name, (model, param_grid) in models.items():

    print(f"\nTuning {name}...")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    grid.fit(X_train, y_train)

    # Save best model
    best_models[name] = grid.best_estimator_

    # --------------------------------------------------------
    # Best parameters
    # --------------------------------------------------------

    print("\nBest parameters:")
    print(grid.best_params_)

    print("Best CV Macro F1:")
    print(f"{grid.best_score_:.4f}")

    # --------------------------------------------------------
    # Test-set evaluation
    # --------------------------------------------------------

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
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

    results.append({
        "model": name,
        "best_cv_macro_f1": grid.best_score_,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro
    })

    print("\nTest Results:")
    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Balanced Accuracy   : {balanced_accuracy:.4f}")
    print(f"Weighted Precision  : {precision:.4f}")
    print(f"Weighted Recall     : {recall:.4f}")
    print(f"Weighted F1         : {f1_weighted:.4f}")
    print(f"Macro F1            : {f1_macro:.4f}")

# ============================================================
# 8. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_macro",
    ascending=False
)

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "ml_model_results_tuned.csv",
    index=False
)

print("\nSaved results to:")
print("ml_model_results_tuned.csv")

# ============================================================
# 10. FINAL MODEL
# ============================================================

best_model_name = results_df.iloc[0]["model"]

print("\n" + "=" * 70)
print("FINAL MODEL BASED ON TEST MACRO F1")
print("=" * 70)

print("Selected model:", best_model_name)

print("\nBest model parameters:")

selected_model = best_models[best_model_name]

# Find the GridSearch result again so we can display
# its exact best parameters.
selected_base_model, selected_param_grid = models[best_model_name]

selected_grid = GridSearchCV(
    estimator=selected_base_model,
    param_grid=selected_param_grid,
    scoring=scoring,
    cv=cv,
    n_jobs=-1,
    refit=True
)

selected_grid.fit(X_train, y_train)

print(selected_grid.best_params_)

# ============================================================
# 11. SAVE FINAL MODEL
# ============================================================

import joblib
import os

os.makedirs("models", exist_ok=True)

model_filename = (
    best_model_name.lower()
    .replace(" ", "_")
    .replace("-", "_")
    + "_tuned.joblib"
)

model_path = os.path.join(
    "models",
    model_filename
)

joblib.dump(
    selected_grid.best_estimator_,
    model_path
)

print("\nFinal tuned model saved to:")
print(model_path)

print("\nTuning completed successfully.")