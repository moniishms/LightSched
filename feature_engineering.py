import pandas as pd
import numpy as np


# ============================================================
# LIGHTSCHED - FEATURE ENGINEERING
# ============================================================

INPUT_FILE = "lightsched_shifted_stream.csv"
OUTPUT_FILE = "lightsched_shifted_ml_features.csv"


# ============================================================
# 1. LOAD RAW WORKLOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Raw dataset shape:")
print(df.shape)


# ============================================================
# 2. GROUP PROCESSES BY WORKLOAD
# ============================================================

def create_workload_features(group):

    # Number of processes
    num_processes = len(group)

    # -----------------------------
    # Arrival time features
    # -----------------------------

    avg_arrival_time = group["arrival_time"].mean()

    arrival_time_variance = group["arrival_time"].var(
        ddof=0
    )

    min_arrival_time = group["arrival_time"].min()
    max_arrival_time = group["arrival_time"].max()

    # Arrival rate
    arrival_span = (
        max_arrival_time -
        min_arrival_time +
        1
    )

    arrival_rate = (
        num_processes /
        arrival_span
    )

    # -----------------------------
    # Burst time features
    # -----------------------------

    avg_burst_time = group["burst_time"].mean()

    burst_time_variance = group["burst_time"].var(
        ddof=0
    )

    # -----------------------------
    # Priority
    # -----------------------------

    avg_priority = group["priority"].mean()

    # -----------------------------
    # I/O frequency
    # -----------------------------

    avg_io_frequency = group["io_frequency"].mean()

    # Return one row for this workload
    return pd.Series({

        "num_processes": num_processes,

        "avg_arrival_time": avg_arrival_time,

        "arrival_time_variance": arrival_time_variance,

        "avg_burst_time": avg_burst_time,

        "burst_time_variance": burst_time_variance,

        "avg_priority": avg_priority,

        "avg_io_frequency": avg_io_frequency,

        "arrival_rate": arrival_rate
    })


# ============================================================
# 3. CREATE ONE ROW PER WORKLOAD
# ============================================================

features = (
    df
    .groupby("workload_id")
    .apply(
        create_workload_features,
        include_groups=False
    )
    .reset_index()
)


# ============================================================
# 4. CHECK FOR MISSING VALUES
# ============================================================

print("\nMissing values:")
print(features.isnull().sum())


# ============================================================
# 5. CHECK FOR DUPLICATE WORKLOADS
# ============================================================

duplicate_rows = features.duplicated().sum()

print("\nDuplicate feature rows:")
print(duplicate_rows)


# ============================================================
# 6. BASIC STATISTICS
# ============================================================

print("\nFeature dataset shape:")
print(features.shape)

print("\nFeature statistics:")
print(features.describe())


# ============================================================
# 7. SAVE ML FEATURE DATASET
# ============================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFeature dataset saved successfully as:")
print(OUTPUT_FILE)