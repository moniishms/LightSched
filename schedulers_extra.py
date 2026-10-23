import random
from schedulers_v2 import (prepare_processes, calculate_metrics, add_to_ready,
                           check_new_arrivals, check_io_completion,
                           finish_cpu_burst, all_completed,
                           round_robin, sjf, priority_scheduling, mlfq)


def srtf(processes):
    """Preemptive shortest-remaining-time-first (on the current CPU burst)."""
    processes = prepare_processes(processes)
    time, cpu_busy = 0, 0
    ready, blocked = [], []

    while not all_completed(processes):
        check_new_arrivals(processes, time, ready)
        check_io_completion(processes, time, ready)

        future = ([p["arrival_time"] for p in processes
                   if p["state"] == "NEW" and p["arrival_time"] > time] +
                  [p["blocked_until"] for p in processes
                   if p["state"] == "BLOCKED" and p["blocked_until"] > time])

        if not ready:
            time = min(future)
            continue

        ready.sort(key=lambda p: (p["remaining_cpu"], p["arrival_time"], p["pid"]))
        p = ready.pop(0)

        if p["ready_since"] is not None:
            p["waiting_time"] += time - p["ready_since"]
            p["ready_since"] = None
        p["state"] = "RUNNING"
        if p["first_start"] is None:
            p["first_start"] = time

        run = p["remaining_cpu"]
        if future:
            run = min(run, min(future) - time)   # run until next arrival / I/O completion

        p["remaining_cpu"] -= run
        time += run
        cpu_busy += run

        if p["remaining_cpu"] == 0:
            finish_cpu_burst(p, time, blocked)
        else:
            add_to_ready(p, time, ready)         # preempted or continuing

    return calculate_metrics(processes, time, cpu_busy)


CANDIDATES = {
    "RR-q2": lambda p: round_robin(p, quantum=2),
    "RR-q4": round_robin,
    "RR-q8": lambda p: round_robin(p, quantum=8),
    "RR-q16": lambda p: round_robin(p, quantum=16),
    "SJF": sjf,
    "SRTF": srtf,
    "Priority": priority_scheduling,
    "MLFQ-4/8": mlfq,
    "MLFQ-2/4": lambda p: mlfq(p, 2, 4),
    "MLFQ-8/16": lambda p: mlfq(p, 8, 16),
}


if __name__ == "__main__":
    def expected_io(pid, burst, io):
        k = min(io, max(0, burst - 1))
        return sum(2 + ((pid + j) % 7) for j in range(1, k + 1))

    rng = random.Random(1)
    for _ in range(150):
        n = rng.randint(2, 12)
        procs = [dict(pid=i + 1, arrival_time=rng.randint(0, 40),
                      burst_time=rng.randint(1, 40), priority=rng.randint(1, 10),
                      io_frequency=rng.randint(0, 6)) for i in range(n)]
        for name, f in CANDIDATES.items():
            res, _ = f(procs)
            by = {r["pid"]: r for r in res}
            for p in procs:
                r = by[p["pid"]]
                io = expected_io(p["pid"], p["burst_time"], p["io_frequency"])
                assert r["waiting_time"] == r["turnaround_time"] - p["burst_time"] - io, (name, p)
                assert r["response_time"] <= r["waiting_time"], (name, p)
    print("All candidate schedulers pass the invariants.")