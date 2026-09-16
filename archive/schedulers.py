from collections import deque


# ============================================================
# Helper function
# ============================================================

def calculate_metrics(processes, completion, first_start, total_time, cpu_busy):
    """
    Calculate per-process and overall scheduling metrics.
    """

    n = len(processes)

    results = []

    for p in processes:

        pid = p["pid"]
        arrival = p["arrival_time"]
        burst = p["burst_time"]

        ct = completion[pid]
        ft = first_start[pid]

        turnaround = ct - arrival
        waiting = turnaround - burst
        response = ft - arrival

        results.append({
            "pid": pid,
            "arrival_time": arrival,
            "burst_time": burst,
            "completion_time": ct,
            "turnaround_time": turnaround,
            "waiting_time": waiting,
            "response_time": response
        })

    avg_waiting = sum(
        r["waiting_time"] for r in results
    ) / n

    avg_turnaround = sum(
        r["turnaround_time"] for r in results
    ) / n

    avg_response = sum(
        r["response_time"] for r in results
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
# 1. ROUND ROBIN
# ============================================================

def round_robin(processes, quantum=4):

    processes = sorted(
        processes,
        key=lambda x: (x["arrival_time"], x["pid"])
    )

    n = len(processes)

    remaining = {
        p["pid"]: p["burst_time"]
        for p in processes
    }

    completion = {}
    first_start = {}

    ready_queue = deque()

    current_time = 0
    index = 0
    cpu_busy = 0

    while len(completion) < n:

        # Add arrived processes
        while (
            index < n
            and processes[index]["arrival_time"] <= current_time
        ):
            ready_queue.append(processes[index]["pid"])
            index += 1

        # CPU idle
        if not ready_queue:

            if index < n:
                current_time = processes[index]["arrival_time"]

            continue

        pid = ready_queue.popleft()

        # First CPU execution
        if pid not in first_start:
            first_start[pid] = current_time

        execution_time = min(
            quantum,
            remaining[pid]
        )

        remaining[pid] -= execution_time
        current_time += execution_time
        cpu_busy += execution_time

        # Add newly arrived processes
        while (
            index < n
            and processes[index]["arrival_time"] <= current_time
        ):
            ready_queue.append(processes[index]["pid"])
            index += 1

        # Process finished
        if remaining[pid] == 0:

            completion[pid] = current_time

        else:

            ready_queue.append(pid)

    return calculate_metrics(
        processes,
        completion,
        first_start,
        current_time,
        cpu_busy
    )


# ============================================================
# 2. SJF - NON-PREEMPTIVE
# ============================================================

def sjf(processes):

    processes = sorted(
        processes,
        key=lambda x: (x["arrival_time"], x["pid"])
    )

    n = len(processes)

    completed = set()

    completion = {}
    first_start = {}

    current_time = 0
    cpu_busy = 0

    while len(completed) < n:

        # Processes currently available
        available = [
            p for p in processes
            if (
                p["pid"] not in completed
                and p["arrival_time"] <= current_time
            )
        ]

        # No process available
        if not available:

            next_process = min(
                (
                    p for p in processes
                    if p["pid"] not in completed
                ),
                key=lambda x: (
                    x["arrival_time"],
                    x["pid"]
                )
            )

            current_time = next_process["arrival_time"]

            continue

        # Shortest burst first
        selected = min(
            available,
            key=lambda x: (
                x["burst_time"],
                x["arrival_time"],
                x["pid"]
            )
        )

        pid = selected["pid"]

        first_start[pid] = current_time

        current_time += selected["burst_time"]

        cpu_busy += selected["burst_time"]

        completion[pid] = current_time

        completed.add(pid)

    return calculate_metrics(
        processes,
        completion,
        first_start,
        current_time,
        cpu_busy
    )


# ============================================================
# 3. PRIORITY - NON-PREEMPTIVE
# ============================================================

def priority_scheduling(processes):

    processes = sorted(
        processes,
        key=lambda x: (x["arrival_time"], x["pid"])
    )

    n = len(processes)

    completed = set()

    completion = {}
    first_start = {}

    current_time = 0
    cpu_busy = 0

    while len(completed) < n:

        available = [
            p for p in processes
            if (
                p["pid"] not in completed
                and p["arrival_time"] <= current_time
            )
        ]

        # CPU idle
        if not available:

            next_process = min(
                (
                    p for p in processes
                    if p["pid"] not in completed
                ),
                key=lambda x: (
                    x["arrival_time"],
                    x["pid"]
                )
            )

            current_time = next_process["arrival_time"]

            continue

        # Lower number = higher priority
        selected = min(
            available,
            key=lambda x: (
                x["priority"],
                x["arrival_time"],
                x["pid"]
            )
        )

        pid = selected["pid"]

        first_start[pid] = current_time

        current_time += selected["burst_time"]

        cpu_busy += selected["burst_time"]

        completion[pid] = current_time

        completed.add(pid)

    return calculate_metrics(
        processes,
        completion,
        first_start,
        current_time,
        cpu_busy
    )


# ============================================================
# 4. MULTI-LEVEL FEEDBACK QUEUE
# ============================================================

def mlfq(
    processes,
    quantum_q0=4,
    quantum_q1=8
):

    processes = sorted(
        processes,
        key=lambda x: (x["arrival_time"], x["pid"])
    )

    n = len(processes)

    remaining = {
        p["pid"]: p["burst_time"]
        for p in processes
    }

    completion = {}
    first_start = {}

    # Three queues
    q0 = deque()
    q1 = deque()
    q2 = deque()

    # Track current queue level
    level = {
        p["pid"]: 0
        for p in processes
    }

    current_time = 0
    index = 0
    cpu_busy = 0

    while len(completion) < n:

        # Add newly arrived processes to Q0
        while (
            index < n
            and processes[index]["arrival_time"] <= current_time
        ):
            q0.append(processes[index]["pid"])
            index += 1

        # Choose highest-priority non-empty queue
        if q0:
            pid = q0.popleft()
            quantum = quantum_q0

        elif q1:
            pid = q1.popleft()
            quantum = quantum_q1

        elif q2:
            pid = q2.popleft()

            # Lowest queue behaves as FCFS
            quantum = remaining[pid]

        else:

            if index < n:
                current_time = processes[index]["arrival_time"]

            continue

        # First CPU execution
        if pid not in first_start:
            first_start[pid] = current_time

        execution_time = min(
            quantum,
            remaining[pid]
        )

        remaining[pid] -= execution_time

        current_time += execution_time

        cpu_busy += execution_time

        # Add processes that arrived during execution
        while (
            index < n
            and processes[index]["arrival_time"] <= current_time
        ):
            q0.append(processes[index]["pid"])
            index += 1

        # Process finished
        if remaining[pid] == 0:

            completion[pid] = current_time

        else:

            # Demote process to next queue
            if level[pid] == 0:

                level[pid] = 1
                q1.append(pid)

            elif level[pid] == 1:

                level[pid] = 2
                q2.append(pid)

            else:

                q2.append(pid)

    return calculate_metrics(
        processes,
        completion,
        first_start,
        current_time,
        cpu_busy
    )