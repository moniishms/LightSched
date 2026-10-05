import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. ADAPTIVE VS STATIC
# ============================================================

def adaptive_vs_static():
    df = pd.read_csv("adaptive_lightsched_results.csv")

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1"
    ]

    labels = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro-F1",
        "Weighted-F1"
    ]

    summary = df.groupby("mode")[metrics].mean() * 100

    # Make sure both modes exist
    summary = summary.loc[["adaptive", "static"]]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width / 2 for i in x],
        summary.loc["adaptive"],
        width,
        label="Adaptive"
    )

    plt.bar(
        [i + width / 2 for i in x],
        summary.loc["static"],
        width,
        label="Static"
    )

    plt.xticks(x, labels)
    plt.ylabel("Percentage (%)")
    plt.title("Adaptive LightSched vs Static Baseline")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/adaptive_vs_static.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# ============================================================
# 2. REGRET COMPARISON
# ============================================================

def regret_comparison():
    df = pd.read_csv("adaptive_lightsched_results.csv")

    summary = (
        df.groupby("mode")["avg_regret_pct"]
        .mean()
        .loc[["adaptive", "static"]]
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        ["Adaptive", "Static"],
        summary.values
    )

    plt.ylabel("Average Regret (%)")
    plt.title("Average Regret: Adaptive vs Static")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/regret_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# ============================================================
# 3. ACCURACY OVER STREAMING BATCHES
# ============================================================

def accuracy_over_batches():
    df = pd.read_csv("adaptive_lightsched_results.csv")

    adaptive = df[df["mode"] == "adaptive"]
    static = df[df["mode"] == "static"]

    plt.figure(figsize=(10, 6))

    plt.plot(
        adaptive["batch_id"],
        adaptive["accuracy"] * 100,
        label="Adaptive"
    )

    plt.plot(
        static["batch_id"],
        static["accuracy"] * 100,
        label="Static"
    )

    plt.xlabel("Batch ID")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Across Streaming Workloads")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/accuracy_over_batches.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# ============================================================
# 4. UNSEEN WORKLOAD EVALUATION
# ============================================================

def unseen_evaluation():
    df = pd.read_csv("lightsched_unseen_evaluation.csv")

    accuracy = df["prediction_correct"].mean() * 100

    print("\nUnseen Workload Evaluation")
    print("--------------------------------")
    print(f"Total workloads : {len(df)}")
    print(f"Correct         : {df['prediction_correct'].sum()}")
    print(f"Accuracy        : {accuracy:.2f}%")

    # Scheduler prediction distribution
    distribution = df["predicted_scheduler"].value_counts()

    plt.figure(figsize=(8, 5))

    plt.bar(
        distribution.index,
        distribution.values
    )

    plt.xlabel("Predicted Scheduler")
    plt.ylabel("Number of Workloads")
    plt.title("Scheduler Predictions on Unseen Workloads")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/unseen_scheduler_predictions.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# ============================================================
# 5. LIGHTSCHED VS RULE-BASED BASELINE
# ============================================================

def rule_baseline_comparison():
    df = pd.read_csv("lightsched_rule_baseline_evaluation.csv")

    accuracy = df["prediction_correct"].mean() * 100
    regret = df["regret_pct"].mean()

    print("\nRule-Based Baseline")
    print("--------------------------------")
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Regret   : {regret:.2f}%")

    metrics = ["Accuracy", "Average Regret"]

    values = [
        accuracy,
        regret
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(metrics, values)

    plt.ylabel("Percentage (%)")
    plt.title("Rule-Based Baseline Evaluation")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/rule_baseline.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# ============================================================
# 6. SCHEDULER DISTRIBUTION IN TRAINING DATA
# ============================================================

def training_label_distribution():
    filename = "lightsched-label_dataset_v2.csv"

    if not os.path.exists(filename):
        print("\nSkipping training label distribution.")
        print(f"File not found: {filename}")
        return

    df = pd.read_csv(filename)

    distribution = df["best_scheduler"].value_counts()

    plt.figure(figsize=(8, 5))

    plt.bar(
        distribution.index,
        distribution.values
    )

    plt.xlabel("Best Scheduler")
    plt.ylabel("Number of Workloads")
    plt.title("Distribution of Best Scheduling Algorithms")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/scheduler_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Generating LightSched graphs...")

    adaptive_vs_static()
    regret_comparison()
    accuracy_over_batches()
    unseen_evaluation()
    rule_baseline_comparison()
    training_label_distribution()

    print("\nAll graphs generated successfully.")
    print(f"Saved in: {OUTPUT_DIR}/")