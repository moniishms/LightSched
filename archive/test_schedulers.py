from archive.schedulers import (
    round_robin,
    sjf,
    priority_scheduling,
    mlfq
)


test_workload = [
    {
        "pid": 1,
        "arrival_time": 0,
        "burst_time": 8,
        "priority": 2
    },
    {
        "pid": 2,
        "arrival_time": 1,
        "burst_time": 4,
        "priority": 1
    },
    {
        "pid": 3,
        "arrival_time": 2,
        "burst_time": 2,
        "priority": 3
    },
    {
        "pid": 4,
        "arrival_time": 3,
        "burst_time": 6,
        "priority": 2
    }
]


schedulers = {
    "Round Robin": round_robin,
    "SJF": sjf,
    "Priority": priority_scheduling,
    "MLFQ": mlfq
}


for name, scheduler in schedulers.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    results, summary = scheduler(test_workload)

    print("\nProcess results:")

    for result in results:
        print(result)

    print("\nOverall metrics:")

    for metric, value in summary.items():
        print(
            f"{metric}: {value:.4f}"
        )