import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Load results
# ============================================================

df = pd.read_csv("adaptive_lightsched_results.csv")

# Separate static and adaptive results
static = df[df["mode"] == "static"].sort_values("batch_id")
adaptive = df[df["mode"] == "adaptive"].sort_values("batch_id")


# ============================================================
# Create plot
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    static["batch_id"],
    static["avg_regret_pct"],
    label="Static",
    linewidth=2
)

plt.plot(
    adaptive["batch_id"],
    adaptive["avg_regret_pct"],
    label="Adaptive",
    linewidth=2
)


# ============================================================
# Workload phase boundaries
# ============================================================

boundaries = [10, 20, 30, 40, 50, 60, 70, 80]

for x in boundaries:
    plt.axvline(
        x=x,
        linestyle="--",
        alpha=0.5
    )


# ============================================================
# Workload phase labels
# ============================================================

phases = [
    (5, "Mixed"),
    (15, "CPU-bound"),
    (25, "Interactive"),
    (35, "Priority-heavy"),
    (45, "Bursty"),
    (55, "SJF-friendly"),
    (65, "Fairness"),
    (75, "CPU-bound"),
    (85, "Interactive")
]

# Get top of y-axis for positioning labels
y_max = max(
    static["avg_regret_pct"].max(),
    adaptive["avg_regret_pct"].max()
)

for x, label in phases:
    plt.text(
        x,
        y_max * 0.97,
        label,
        ha="center",
        va="top",
        fontsize=9
    )


# ============================================================
# Labels and title
# ============================================================

plt.xlabel("Batch", fontsize=11)
plt.ylabel("Average Regret (%)", fontsize=11)

plt.title(
    "Static vs Adaptive LightSched Under Workload Distribution Shifts",
    fontsize=14
)


# ============================================================
# Formatting
# ============================================================

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xlim(1, 90)

plt.tight_layout()


# ============================================================
# Save figure
# ============================================================

# This overwrites the file if it already exists
plt.savefig(
    "adaptive_regret_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()