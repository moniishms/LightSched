from collections import deque


# ============================================================
# PROCESS PREPARATION
# ============================================================

def prepare_processes(processes):
    """
    Convert each process into CPU bursts separated by I/O.

    burst_time = total CPU time required
    io_frequency = number of I/O operations

    Example:
        burst_time = 10
        io_frequency = 2

        CPU bursts might become:
        [4, 3, 3]

        with 2 I/O operations between them.
    """

    prepared = []

    for p in processes:

        burst = int(p["burst_time"])
        io_frequency = int(p.get("io_frequency", 0))

        # Cannot have more I/O operations than CPU divisions
        num_io = min(io_frequency, max(0, burst - 1))

        # Divide total CPU burst into num_io + 1 CPU bursts
        num_cpu_bursts = num_io + 1

        base = burst // num_cpu_bursts
        remainder = burst % num_cpu_bursts

        cpu_bursts = []

        for i in range(num_cpu_bursts):
            value = base

            if i < remainder:
                value += 1

            cpu_bursts.append(value)

        prepared.append({
            "pid": int(p["pid"]),
            "arrival_time": int(p["arrival_time"]),
            "priority": int(p["priority"]),
            "io_frequency": io_frequency,

            "cpu_bursts": cpu_bursts,
            "current_burst": 0,
            "remaining_cpu": cpu_bursts[0],

            "state": "NEW",

            "completion_time": None,
            "first_start": None,

            # Actual time spent in READY
            "waiting_time": 0,

            # When process entered READY
            "ready_since": None,

            # When I/O finishes
            "blocked_until": None,
        })

    return prepared


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(processes, total_time, cpu_busy):

    results = []

    for p in processes:

        arrival = p["arrival_time"]
        completion = p["completion_time"]
        first_start = p["first_start"]

        turnaround = completion - arrival

        waiting = p["waiting_time"]

        response = first_start - arrival

        results.append({
            "pid": p["pid"],
            "completion_time": completion,
            "turnaround_time": turnaround,
            "waiting_time": waiting,
            "response_time": response
        })

    n = len(processes)

    avg_waiting = sum(
        x["waiting_time"] for x in results
    ) / n

    avg_turnaround = sum(
        x["turnaround_time"] for x in results
    ) / n

    avg_response = sum(
        x["response_time"] for x in results
    ) / n

    throughput = n / total_time if total_time > 0 else 0

    cpu_utilization = (
        cpu_busy / total_time * 100
        if total_time > 0
        else 0
    )

    summary = {
        "avg_waiting_time": avg_waiting,
        "avg_turnaround_time": avg_turnaround,
        "avg_response_time": avg_response,
        "throughput": throughput,
        "cpu_utilization": cpu_utilization
    }

    return results, summary


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def add_to_ready(process, current_time, ready_queue):

    process["state"] = "READY"

    if process["ready_since"] is None:
        process["ready_since"] = current_time

    ready_queue.append(process)


def update_ready_waiting(ready_queue, current_time):

    for p in ready_queue:

        if p["ready_since"] is not None:

            p["waiting_time"] += current_time - p["ready_since"]

            p["ready_since"] = current_time


def finish_cpu_burst(process, current_time, blocked):

    """
    Called when the current CPU burst finishes.
    """

    process["current_burst"] += 1

    # All CPU bursts completed
    if process["current_burst"] >= len(process["cpu_bursts"]):

        process["state"] = "COMPLETED"
        process["completion_time"] = current_time

        return

    # Otherwise process goes to I/O
    process["state"] = "BLOCKED"

    # Synthetic I/O duration.
    # Kept deterministic for reproducibility.
    io_time = 2 + (
        (process["pid"] + process["current_burst"]) % 7
    )

    process["blocked_until"] = current_time + io_time

    blocked.append(process)

    # Prepare next CPU burst
    process["remaining_cpu"] = (
        process["cpu_bursts"][process["current_burst"]]
    )

    process["ready_since"] = None


def check_new_arrivals(processes, current_time, ready_queue):

    for p in processes:

        if (
            p["state"] == "NEW"
            and p["arrival_time"] <= current_time
        ):

            add_to_ready(
                p,
                current_time,
                ready_queue
            )


def check_io_completion(processes, current_time, ready_queue):

    for p in processes:

        if (
            p["state"] == "BLOCKED"
            and p["blocked_until"] <= current_time
        ):

            p["blocked_until"] = None

            add_to_ready(
                p,
                current_time,
                ready_queue
            )


def all_completed(processes):

    return all(
        p["state"] == "COMPLETED"
        for p in processes
    )


# ============================================================
# ROUND ROBIN
# ============================================================

def round_robin(processes, quantum=4):

    processes = prepare_processes(processes)

    ready = deque()
    blocked = []

    time = 0
    cpu_busy = 0

    while not all_completed(processes):

        check_new_arrivals(
            processes,
            time,
            ready
        )

        check_io_completion(
            processes,
            time,
            ready
        )

        # CPU idle
        if not ready:

            future_times = []

            for p in processes:

                if p["state"] == "NEW":
                    future_times.append(p["arrival_time"])

                elif p["state"] == "BLOCKED":
                    future_times.append(p["blocked_until"])

            if future_times:

                time = min(future_times)
                continue

        # Select process
        p = ready.popleft()

        # Add READY waiting time
        if p["ready_since"] is not None:

            p["waiting_time"] += (
                time - p["ready_since"]
            )

            p["ready_since"] = None

        p["state"] = "RUNNING"

        if p["first_start"] is None:
            p["first_start"] = time

        run_time = min(
            quantum,
            p["remaining_cpu"]
        )

        p["remaining_cpu"] -= run_time

        time += run_time
        cpu_busy += run_time

        # New arrivals / I/O completions during execution
        check_new_arrivals(
            processes,
            time,
            ready
        )

        check_io_completion(
            processes,
            time,
            ready
        )

        # CPU burst finished
        if p["remaining_cpu"] == 0:

            finish_cpu_burst(
                p,
                time,
                blocked
            )

        else:

            # Quantum expired
            add_to_ready(
                p,
                time,
                ready
            )

    return calculate_metrics(
        processes,
        time,
        cpu_busy
    )


# ============================================================
# SJF
# ============================================================

def sjf(processes):

    processes = prepare_processes(processes)

    time = 0
    cpu_busy = 0

    ready = []
    blocked = []

    while not all_completed(processes):

        check_new_arrivals(
            processes,
            time,
            ready
        )

        check_io_completion(
            processes,
            time,
            ready
        )

        if not ready:

            future_times = []

            for p in processes:

                if p["state"] == "NEW":
                    future_times.append(p["arrival_time"])

                elif p["state"] == "BLOCKED":
                    future_times.append(p["blocked_until"])

            if future_times:

                time = min(future_times)
                continue

        # Shortest CURRENT CPU burst
        ready.sort(
            key=lambda p: (
                p["remaining_cpu"],
                p["arrival_time"],
                p["pid"]
            )
        )

        p = ready.pop(0)

        if p["ready_since"] is not None:

            p["waiting_time"] += (
                time - p["ready_since"]
            )

            p["ready_since"] = None

        p["state"] = "RUNNING"

        if p["first_start"] is None:
            p["first_start"] = time

        run_time = p["remaining_cpu"]

        time += run_time
        cpu_busy += run_time

        p["remaining_cpu"] = 0

        check_new_arrivals(
            processes,
            time,
            ready
        )

        check_io_completion(
            processes,
            time,
            ready
        )

        finish_cpu_burst(
            p,
            time,
            blocked
        )

    return calculate_metrics(
        processes,
        time,
        cpu_busy
    )


# ============================================================
# PRIORITY
# ============================================================

def priority_scheduling(processes):

    processes = prepare_processes(processes)

    time = 0
    cpu_busy = 0

    ready = []
    blocked = []

    while not all_completed(processes):

        check_new_arrivals(
            processes,
            time,
            ready
        )

        check_io_completion(
            processes,
            time,
            ready
        )

        if not ready:

            future_times = []

            for p in processes:

                if p["state"] == "NEW":
                    future_times.append(p["arrival_time"])

                elif p["state"] == "BLOCKED":
                    future_times.append(p["blocked_until"])

            if future_times:

                time = min(future_times)
                continue

        # Lower priority number = higher priority
        ready.sort(
            key=lambda p: (
                p["priority"],
                p["arrival_time"],
                p["pid"]
            )
        )

        p = ready.pop(0)

        if p["ready_since"] is not None:

            p["waiting_time"] += (
                time - p["ready_since"]
            )

            p["ready_since"] = None

        p["state"] = "RUNNING"

        if p["first_start"] is None:
            p["first_start"] = time

        run_time = p["remaining_cpu"]

        time += run_time
        cpu_busy += run_time

        p["remaining_cpu"] = 0

        check_new_arrivals(
            processes,
            time,
            ready
        )

        check_io_completion(
            processes,
            time,
            ready
        )

        finish_cpu_burst(
            p,
            time,
            blocked
        )

    return calculate_metrics(
        processes,
        time,
        cpu_busy
    )


# ============================================================
# MLFQ
# ============================================================

def mlfq(processes, quantum_q0=4, quantum_q1=8):

    processes = prepare_processes(processes)

    q0 = deque()
    q1 = deque()
    q2 = deque()

    time = 0
    cpu_busy = 0

    while not all_completed(processes):

        # Add new processes to highest queue
        for p in processes:

            if (
                p["state"] == "NEW"
                and p["arrival_time"] <= time
            ):

                add_to_ready(
                    p,
                    time,
                    q0
                )

        # I/O completed → return to highest queue
        for p in processes:

            if (
                p["state"] == "BLOCKED"
                and p["blocked_until"] <= time
            ):

                p["blocked_until"] = None

                add_to_ready(
                    p,
                    time,
                    q0
                )

        # Select highest non-empty queue
        if q0:
            queue = q0
            quantum = quantum_q0
            level = 0

        elif q1:
            queue = q1
            quantum = quantum_q1
            level = 1

        elif q2:
            queue = q2
            quantum = None
            level = 2

        else:

            future_times = []

            for p in processes:

                if p["state"] == "NEW":
                    future_times.append(p["arrival_time"])

                elif p["state"] == "BLOCKED":
                    future_times.append(p["blocked_until"])

            if future_times:

                time = min(future_times)

            continue

        p = queue.popleft()

        if p["ready_since"] is not None:

            p["waiting_time"] += (
                time - p["ready_since"]
            )

            p["ready_since"] = None

        p["state"] = "RUNNING"

        if p["first_start"] is None:
            p["first_start"] = time

        if quantum is None:

            run_time = p["remaining_cpu"]

        else:

            run_time = min(
                quantum,
                p["remaining_cpu"]
            )

        p["remaining_cpu"] -= run_time

        time += run_time
        cpu_busy += run_time

        # Handle arrivals / I/O
        for new_p in processes:

            if (
                new_p["state"] == "NEW"
                and new_p["arrival_time"] <= time
            ):

                add_to_ready(
                    new_p,
                    time,
                    q0
                )

        for blocked_p in processes:

            if (
                blocked_p["state"] == "BLOCKED"
                and blocked_p["blocked_until"] <= time
            ):

                blocked_p["blocked_until"] = None

                add_to_ready(
                    blocked_p,
                    time,
                    q0
                )

        # CPU burst completed
        if p["remaining_cpu"] == 0:

            finish_cpu_burst(
                p,
                time,
                []
            )

        else:

            # Quantum expired
            if level == 0:

                p["state"] = "READY"
                p["ready_since"] = time
                q1.append(p)

            elif level == 1:

                p["state"] = "READY"
                p["ready_since"] = time
                q2.append(p)

            else:

                p["state"] = "READY"
                p["ready_since"] = time
                q2.append(p)

    return calculate_metrics(
        processes,
        time,
        cpu_busy
    )