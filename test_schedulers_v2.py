from schedulers_v2 import (
    round_robin,
    sjf,
    priority_scheduling,
    mlfq
)


test_workload = [
    {
        "pid": 1,
        "arrival_time": 0,
        "burst_time": 5,
        "priority": 1,
        "io_frequency": 1
    },
    {
        "pid": 2,
        "arrival_time": 20,
        "burst_time": 5,
        "priority": 2,
        "io_frequency": 0
    }
]


schedulers = {
    "RR": round_robin,
    "SJF": sjf,
    "Priority": priority_scheduling,
    "MLFQ": mlfq
}


for name, scheduler in schedulers.items():

    results, summary = scheduler(test_workload)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    for r in results:
        print(r)

    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")