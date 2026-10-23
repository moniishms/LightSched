import random
import pytest
from schedulers_v2 import round_robin, sjf, priority_scheduling, mlfq

SCHEDULERS = {"RR": round_robin, "SJF": sjf,
              "Priority": priority_scheduling, "MLFQ": mlfq}


def P(pid, arr, burst, prio=5, io=0):
    return dict(pid=pid, arrival_time=arr, burst_time=burst,
                priority=prio, io_frequency=io)


def test_sjf_hand_computed():
    # P1 runs 0-10, P2 (arrived at 1) runs 10-13
    _, s = sjf([P(1, 0, 10), P(2, 1, 3)])
    assert s["avg_waiting_time"] == pytest.approx(4.5)      # (0 + 9) / 2
    assert s["avg_turnaround_time"] == pytest.approx(11.0)  # (10 + 12) / 2
    assert s["avg_response_time"] == pytest.approx(4.5)
    assert s["throughput"] == pytest.approx(2 / 13)
    assert s["cpu_utilization"] == pytest.approx(100.0)


def test_rr_hand_computed():
    # quantum 4: P1 0-4, P2 4-7, P1 7-11, P1 11-13
    _, s = round_robin([P(1, 0, 10), P(2, 1, 3)])
    assert s["avg_waiting_time"] == pytest.approx(3.0)
    assert s["avg_turnaround_time"] == pytest.approx(9.5)
    assert s["avg_response_time"] == pytest.approx(1.5)


def expected_io(pid, burst, io):
    num_io = min(io, max(0, burst - 1))
    return sum(2 + ((pid + k) % 7) for k in range(1, num_io + 1))


def test_waiting_invariant_random():
    """waiting == turnaround - cpu - io, and response <= waiting."""
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(2, 12)
        procs = [P(i + 1, rng.randint(0, 40), rng.randint(1, 40),
                   rng.randint(1, 10), rng.randint(0, 6)) for i in range(n)]
        for name, fn in SCHEDULERS.items():
            results, _ = fn(procs)
            by_pid = {r["pid"]: r for r in results}
            for p in procs:
                r = by_pid[p["pid"]]
                io = expected_io(p["pid"], p["burst_time"], p["io_frequency"])
                assert r["waiting_time"] == r["turnaround_time"] - p["burst_time"] - io, (name, p)
                assert r["response_time"] <= r["waiting_time"], (name, p)


if __name__ == "__main__":
    test_sjf_hand_computed()
    test_rr_hand_computed()
    test_waiting_invariant_random()
    print("All scheduler tests passed.")