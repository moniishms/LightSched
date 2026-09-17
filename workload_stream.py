import random
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

random.seed(9999)
np.random.seed(9999)

INITIAL_WORKLOADS = 500
BATCH_SIZE = 50
NUM_BATCHES = 90

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
# 6. FAIRNESS WORKLOAD
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
# WORKLOAD GENERATORS
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
# DISTRIBUTION-SHIFT PHASES
# ============================================================

PHASES = {
    1: "mixed",
    2: "mixed",
    3: "mixed",
    4: "mixed",
    5: "mixed",
    6: "mixed",
    7: "mixed",
    8: "mixed",
    9: "mixed",
    10: "mixed",

    11: "cpu_bound",
    12: "cpu_bound",
    13: "cpu_bound",
    14: "cpu_bound",
    15: "cpu_bound",
    16: "cpu_bound",
    17: "cpu_bound",
    18: "cpu_bound",
    19: "cpu_bound",
    20: "cpu_bound",

    21: "interactive",
    22: "interactive",
    23: "interactive",
    24: "interactive",
    25: "interactive",
    26: "interactive",
    27: "interactive",
    28: "interactive",
    29: "interactive",
    30: "interactive",

    31: "priority_heavy",
    32: "priority_heavy",
    33: "priority_heavy",
    34: "priority_heavy",
    35: "priority_heavy",
    36: "priority_heavy",
    37: "priority_heavy",
    38: "priority_heavy",
    39: "priority_heavy",
    40: "priority_heavy",

    41: "bursty",
    42: "bursty",
    43: "bursty",
    44: "bursty",
    45: "bursty",
    46: "bursty",
    47: "bursty",
    48: "bursty",
    49: "bursty",
    50: "bursty",

    51: "sjf_friendly",
    52: "sjf_friendly",
    53: "sjf_friendly",
    54: "sjf_friendly",
    55: "sjf_friendly",
    56: "sjf_friendly",
    57: "sjf_friendly",
    58: "sjf_friendly",
    59: "sjf_friendly",
    60: "sjf_friendly",

    61: "fairness",
    62: "fairness",
    63: "fairness",
    64: "fairness",
    65: "fairness",
    66: "fairness",
    67: "fairness",
    68: "fairness",
    69: "fairness",
    70: "fairness",

    71: "cpu_bound",
    72: "cpu_bound",
    73: "cpu_bound",
    74: "cpu_bound",
    75: "cpu_bound",
    76: "cpu_bound",
    77: "cpu_bound",
    78: "cpu_bound",
    79: "cpu_bound",
    80: "cpu_bound",

    81: "interactive",
    82: "interactive",
    83: "interactive",
    84: "interactive",
    85: "interactive",
    86: "interactive",
    87: "interactive",
    88: "interactive",
    89: "interactive",
    90: "interactive"
}


# ============================================================
# MIXED WORKLOAD TYPE
# ============================================================

def choose_mixed_type():

    return random.choice([
        "cpu_bound",
        "interactive",
        "priority_heavy",
        "sjf_friendly",
        "bursty",
        "fairness"
    ])


# ============================================================
# GENERATE STREAM
# ============================================================

all_records = []

workload_id = 1


# ------------------------------------------------------------
# INITIAL TRAINING SET
# ------------------------------------------------------------

for _ in range(INITIAL_WORKLOADS):

    workload_type = choose_mixed_type()

    n = random.randint(
        MIN_PROCESSES,
        MAX_PROCESSES
    )

    generator = generators[workload_type]

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

    workload_id += 1


# ------------------------------------------------------------
# STREAMING BATCHES
# ------------------------------------------------------------

for batch_id in range(1, NUM_BATCHES + 1):

    phase_type = PHASES[batch_id]

    for _ in range(BATCH_SIZE):

        if phase_type == "mixed":
            workload_type = choose_mixed_type()
        else:
            workload_type = phase_type

        n = random.randint(
            MIN_PROCESSES,
            MAX_PROCESSES
        )

        generator = generators[workload_type]

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

        workload_id += 1


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(all_records)


# ============================================================
# SAVE
# ============================================================

output_file = "lightsched_shifted_stream.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("=" * 60)
print("LightSched Distribution-Shifted Workload Stream")
print("=" * 60)

print("\nTotal workloads:")
print(df["workload_id"].nunique())

print("\nTotal process records:")
print(len(df))

print("\nDataset shape:")
print(df.shape)

print("\nOverall workload type distribution:")
print(
    df.groupby("workload_type")["workload_id"]
    .nunique()
)

print("\nBatch distribution:")

for batch_id in range(1, NUM_BATCHES + 1):

    start_id = (
        INITIAL_WORKLOADS
        + (batch_id - 1) * BATCH_SIZE
        + 1
    )

    end_id = (
        INITIAL_WORKLOADS
        + batch_id * BATCH_SIZE
    )

    batch_df = df[
        (df["workload_id"] >= start_id)
        & (df["workload_id"] <= end_id)
    ]

    print(
        f"Batch {batch_id:02d}: "
        f"{PHASES[batch_id]:15s} | "
        f"{batch_df['workload_id'].nunique()} workloads"
    )

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nSaved to:")
print(output_file)

print("\nGeneration complete.")