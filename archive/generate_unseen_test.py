import random
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

random.seed(9999)
np.random.seed(9999)

NUM_WORKLOADS = 500

START_WORKLOAD_ID = 5001

MIN_PROCESSES = 5
MAX_PROCESSES = 30

MAX_ARRIVAL_TIME = 100


# ============================================================
# 1. CPU-BOUND WORKLOAD
# ============================================================

def generate_cpu_bound(n):

    arrival_times = [
        random.randint(0, 20)
        for _ in range(n)
    ]

    burst_times = [
        random.randint(60, 100)
        for _ in range(n)
    ]

    priorities = [
        random.randint(4, 10)
        for _ in range(n)
    ]

    io_frequency = [
        random.randint(0, 2)
        for _ in range(n)
    ]

    return arrival_times, burst_times, priorities, io_frequency


# ============================================================
# 2. INTERACTIVE WORKLOAD
# ============================================================

def generate_interactive(n):

    arrival_times = [
        random.randint(0, 100)
        for _ in range(n)
    ]

    burst_times = [
        random.randint(1, 10)
        for _ in range(n)
    ]

    priorities = [
        random.randint(1, 10)
        for _ in range(n)
    ]

    io_frequency = [
        random.randint(7, 10)
        for _ in range(n)
    ]

    return arrival_times, burst_times, priorities, io_frequency


# ============================================================
# 3. PRIORITY-INTENSIVE WORKLOAD
# ============================================================

def generate_priority_heavy(n):

    arrival_times = [
        random.randint(0, 30)
        for _ in range(n)
    ]

    burst_times = [
        random.randint(15, 50)
        for _ in range(n)
    ]

    priorities = []

    for _ in range(n):

        if random.random() < 0.5:
            priorities.append(random.randint(1, 2))
        else:
            priorities.append(random.randint(9, 10))

    io_frequency = [
        random.randint(0, 4)
        for _ in range(n)
    ]

    return arrival_times, burst_times, priorities, io_frequency


# ============================================================
# 4. SJF-FRIENDLY WORKLOAD
# ============================================================

def generate_sjf_friendly(n):

    arrival_times = [
        random.randint(0, 40)
        for _ in range(n)
    ]

    burst_times = []

    for _ in range(n):

        r = random.random()

        if r < 0.45:
            burst_times.append(random.randint(1, 5))

        elif r < 0.75:
            burst_times.append(random.randint(20, 40))

        else:
            burst_times.append(random.randint(70, 100))

    priorities = [
        random.randint(4, 10)
        for _ in range(n)
    ]

    io_frequency = [
        random.randint(0, 4)
        for _ in range(n)
    ]

    return arrival_times, burst_times, priorities, io_frequency


# ============================================================
# 5. BURSTY / CONTENTION WORKLOAD
# ============================================================

def generate_bursty(n):

    cluster_centers = [
        random.randint(0, 20),
        random.randint(30, 50),
        random.randint(60, 80)
    ]

    arrival_times = []

    for _ in range(n):

        center = random.choice(cluster_centers)

        arrival = center + random.randint(-3, 3)

        arrival = max(
            0,
            min(MAX_ARRIVAL_TIME, arrival)
        )

        arrival_times.append(arrival)

    burst_times = [
        random.randint(5, 60)
        for _ in range(n)
    ]

    priorities = [
        random.randint(1, 10)
        for _ in range(n)
    ]

    io_frequency = [
        random.randint(2, 8)
        for _ in range(n)
    ]

    return arrival_times, burst_times, priorities, io_frequency


# ============================================================
# 6. ROUND-ROBIN / FAIRNESS WORKLOAD
# ============================================================

def generate_fairness(n):

    arrival_times = [
        random.randint(0, 10)
        for _ in range(n)
    ]

    burst_times = [
        random.randint(15, 30)
        for _ in range(n)
    ]

    priorities = [
        random.randint(4, 7)
        for _ in range(n)
    ]

    io_frequency = [
        random.randint(5, 10)
        for _ in range(n)
    ]

    return arrival_times, burst_times, priorities, io_frequency


# ============================================================
# WORKLOAD TYPES
# ============================================================

generators = {
    "cpu_bound": generate_cpu_bound,
    "interactive": generate_interactive,
    "priority_heavy": generate_priority_heavy,
    "sjf_friendly": generate_sjf_friendly,
    "bursty": generate_bursty,
    "fairness": generate_fairness
}


# ============================================================
# GENERATE UNSEEN DATASET
# ============================================================

all_records = []

workload_types = list(generators.keys())


for workload_id in range(
    START_WORKLOAD_ID,
    START_WORKLOAD_ID + NUM_WORKLOADS
):

    workload_type = random.choice(workload_types)

    generator = generators[workload_type]

    n = random.randint(
        MIN_PROCESSES,
        MAX_PROCESSES
    )

    (
        arrival_times,
        burst_times,
        priorities,
        io_frequency
    ) = generator(n)

    for pid in range(1, n + 1):

        all_records.append({

            "workload_id": workload_id,

            "pid": pid,

            "arrival_time": arrival_times[pid - 1],

            "burst_time": burst_times[pid - 1],

            "priority": priorities[pid - 1],

            "io_frequency": io_frequency[pid - 1],

            "workload_type": workload_type
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_records)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "lightsched_unseen_workload_dataset.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("=" * 60)
print("LightSched Unseen Workload Generation")
print("=" * 60)

print("\nNumber of workloads:")
print(df["workload_id"].nunique())

print("\nWorkload ID range:")
print(
    df["workload_id"].min(),
    "to",
    df["workload_id"].max()
)

print("\nNumber of process records:")
print(len(df))

print("\nDataset shape:")
print(df.shape)

print("\nWorkload type distribution:")
print(
    df.groupby("workload_type")["workload_id"]
    .nunique()
)

print("\nProcess count statistics:")
print(
    df.groupby("workload_id")
    .size()
    .describe()
)

print("\nDuplicate records:")
print(
    df.duplicated().sum()
)

print("\nSaved to:")
print(output_file)

print("\nUnseen workload generation complete.")