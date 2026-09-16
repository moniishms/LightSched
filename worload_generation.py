import random
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

random.seed(42)
np.random.seed(42)

NUM_WORKLOADS = 5000

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

    # Moderate bursts so priority ordering has a strong effect
    burst_times = [
        random.randint(15, 50)
        for _ in range(n)
    ]

    priorities = []

    for _ in range(n):

        # Strong priority separation
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

    # Strong variation in burst lengths
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

    # Most processes arrive in a few clusters
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

    # Many processes become ready around the same time
    arrival_times = [
        random.randint(0, 10)
        for _ in range(n)
    ]

    # Similar burst lengths reduce SJF's advantage
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
# GENERATE DATASET
# ============================================================

all_records = []

workload_types = list(generators.keys())


for workload_id in range(1, NUM_WORKLOADS + 1):

    # Select workload category
    workload_type = random.choice(workload_types)

    generator = generators[workload_type]

    # Number of processes
    n = random.randint(
        MIN_PROCESSES,
        MAX_PROCESSES
    )

    # Generate workload
    (
        arrival_times,
        burst_times,
        priorities,
        io_frequency
    ) = generator(n)


    # Store processes
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

output_file = "lightsched_workload_dataset.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("=" * 60)
print("LightSched Workload Generation")
print("=" * 60)

print("\nNumber of workloads:")
print(df["workload_id"].nunique())

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

print("\nArrival time statistics:")
print(
    df["arrival_time"].describe()
)

print("\nBurst time statistics:")
print(
    df["burst_time"].describe()
)

print("\nPriority statistics:")
print(
    df["priority"].describe()
)

print("\nI/O frequency statistics:")
print(
    df["io_frequency"].describe()
)

print("\nDuplicate records:")
print(
    df.duplicated().sum()
)

print("\nSaved to:")
print(output_file)

print("\nGeneration complete.")