# Adaptive ML Inference Server: OS Concurrency Study - Phase 1

**Student**: Abdullah Al Galib  
**ID**: 2232535642 &emsp;&emsp; **Sec**: 2              
**Course**: CSE-323 Operating Systems  
**Faculty**: Dr. Safat Siddiqui (SSI)  

---

## Executive Summary

**Problem**: Modern task execution servers face fundamental OS concurrency questions - should we use threads, processes, or async I/O? How do different scheduling policies affect performance under mixed workloads? Current solutions use heuristics without systematic OS-level analysis.

**Phase 1 Approach**: Build a task execution server with three concurrency implementations (threads, processes, async) and two scheduling policies (FIFO, Priority). Execute diverse workloads (CPU-bound, I/O-bound, memory-bound, mixed, simple ML). Measure how OS primitives (context switching, GIL, scheduling, IPC) affect latency, throughput, and fairness.

**Why OS-Focused**: This is fundamentally an OS analysis project studying concurrency primitives and scheduling algorithms. Workloads are diverse and realistic (including simple ML tasks) to demonstrate real-world applicability, but the core contribution is understanding OS-level behavior.

---

## Core Research Questions

### Primary Question
**How do OS concurrency primitives (threads, processes, async I/O) and scheduling policies (FIFO, Priority) affect task execution server performance under varying workload patterns?**

### Sub-Questions
1. When does Python's GIL make multi-threading slower than multi-processing?
2. How does context switching overhead differ between threads and processes for different workload types?
3. What is the measured impact of FIFO vs Priority scheduling on waiting time and fairness?
4. Under what conditions does priority scheduling cause starvation?
5. How does async I/O compare to threads for I/O-bound vs CPU-bound workloads?

---

## Technical Approach - Phase 1 (4 Weeks)

### Workload Categories

Phase 1 uses a **diverse task suite** to understand how different workload characteristics interact with concurrency primitives and scheduling policies:

| Task Type | Example | Computational Pattern | Expected Concurrency Behavior |
|-----------|---------|----------------------|------------------------------|
| **CPU-Bound** | Fibonacci(n), Matrix multiply | Pure computation | Threads limited by GIL, processes scale |
| **I/O-Bound** | File operations, sleep simulation | Waiting on external resources | Threads efficient, async excellent |
| **Memory-Bound** | Large array allocation/operations | Data movement heavy | Memory overhead visible |
| **Mixed** | CPU then I/O, I/O then CPU | Combined patterns | Reveals scheduling trade-offs |
| **ML (Simple)** | Scikit-learn prediction | CPU-bound with library overhead | Realistic production workload |

**Rationale**: By testing concurrency models and scheduling policies against diverse workloads, we systematically identify which OS primitives and policies suit which computational patterns.

### Implementation A: Multi-threaded Server
- **Framework**: Flask + Python threading
- **Architecture**: Thread pool (configurable size: 4, 8, 16 threads)
- **Synchronization**: Queue for request management, Lock for shared resources
- **Scheduling**: FIFO and Priority implementations
- **Workloads**: All 5 task types
- **Key Learning**: GIL impact on CPU vs I/O tasks, thread context switching cost, lock contention, scheduling policy effects

```python
# Conceptual structure
class ThreadedTaskServer:
    def __init__(self, num_threads, scheduler_type):
        self.scheduler = FIFOScheduler() if scheduler_type == 'fifo' else PriorityScheduler()
        self.dispatcher = Dispatcher(scheduler=self.scheduler)
        self.thread_pool = ThreadPoolExecutor(max_workers=num_threads)
        self.workloads = {
            'cpu_fib': fibonacci,
            'io_file': file_operations,
            'memory_array': array_operations,
            'mixed_cpu_io': cpu_then_io,
            'ml_predict': predict_batch
        }
    
    def handle_request(self, request):
        job = Job(workload_type=request.type, params=request.params, priority=request.priority)
        self.scheduler.enqueue(job)
        self.dispatcher.dispatch_to_executor(self.thread_pool)
```

### Implementation B: Multi-process Server
- **Framework**: Python multiprocessing + message passing
- **Architecture**: Process pool (2, 4, 8 processes)
- **IPC**: Queue for request/response
- **Scheduling**: Same FIFO and Priority implementations
- **Workloads**: Same 5 task types for direct comparison
- **Key Learning**: Process isolation benefits, IPC overhead, memory duplication costs, CPU-bound task scaling, scheduling behavior with process overhead

### Implementation C: Async I/O Server
- **Framework**: FastAPI + asyncio
- **Architecture**: Event loop with async/await
- **Scheduling**: FIFO (Priority deferred due to async complexity)
- **Workloads**: Focus on I/O-bound and mixed (CPU offloaded to thread pool)
- **Key Learning**: When non-blocking I/O helps, event loop blocking by CPU tasks, async vs threading trade-offs

### Scheduling Policy Analysis Within the Server

**FIFO (First-In-First-Out)**:
- **Policy**: Process jobs in arrival order
- **Implementation**: Standard queue
- **Expected Behavior**: Fair but inefficient (head-of-line blocking)
- **Metrics**: Average waiting time, fairness (coefficient of variation)

**Priority-Based Scheduling**:
- **Policy**: Process high-priority jobs first
- **Priority Assignment**: Task-based (CPU=high, I/O=low, ML=medium) or user-specified
- **Implementation**: Priority queue with timestamps for starvation detection
- **Expected Behavior**: Better responsiveness for high-priority tasks, potential starvation of low-priority tasks
- **Metrics**: Per-priority waiting time, starvation events (tasks waiting >30s), responsiveness (p95/p99 latency for high-priority)

**Comparison Experiments**:
- Mixed workload: 40% CPU-bound (priority=1), 40% I/O-bound (priority=3), 20% ML (priority=2)
- Run with FIFO @ 50 users, 120s
- Run with Priority @ 50 users, 120s
- Measure: waiting time distribution, completion order, fairness metrics

---

## System Architecture

```
Request (JSON POST) → Scheduler (FIFO/Priority) → Dispatcher → Executor (Thread/Process/Async) → Workload → Response
                           ↓
                    Metrics Tracker (delta-based)
                           ↓
                    Experiment Logger (JSONL)
```

**Key Components**:
- **Job**: Abstraction with metadata (workload_type, params, priority, timestamps)
- **Scheduler**: Enqueue/dequeue policy (FIFO or Priority)
- **Dispatcher**: Pulls jobs from scheduler, submits to executor
- **Executor**: Concurrency mechanism (ThreadPool, ProcessPool, AsyncIO)
- **Metrics Tracker**: Delta-based OS metrics (CPU time, context switches, memory)

---

## Metrics & Measurement

| Metric | How to Measure | Why It Matters |
|--------|----------------|----------------|
| **Latency (p50, p95, p99)** | Request arrival to response time | User experience, tail latency |
| **Throughput (req/sec)** | Completed requests per second | System capacity |
| **Waiting Time** | Dequeue time - Enqueue time | Scheduling efficiency |
| **Execution Time** | Completion time - Start time | Workload performance |
| **CPU Time (user, system)** | Delta via `psutil.cpu_times()` | Compute efficiency |
| **Context Switches (voluntary, involuntary)** | Delta via `psutil.num_ctx_switches()` | OS overhead |
| **Memory Usage (RSS)** | Delta via `psutil.memory_info()` | Resource consumption |
| **Starvation Events** | Count of jobs waiting >30s | Scheduling fairness |

**Platform**: Windows (development) + WSL2 Ubuntu (execution and measurement)  
**OS Profiling**: psutil for metrics; perf noted as future work (Linux-specific)

---

## Phase 1 Deliverables (4 Weeks)

### Week 1: Workloads + Baseline Server
- **Deliverables**: 5 workload modules (parameterized), baseline single-threaded server, metrics infrastructure
- **Output**: Baseline performance data (12 experiments)

### Week 2: Multi-threaded Server + Scheduling Infrastructure
- **Deliverables**: Threaded server with FIFO and Priority schedulers, dispatcher implementation
- **Output**: Thread performance data (9 experiments), GIL impact analysis

### Week 3: Multi-process Server
- **Deliverables**: Process-based server with same scheduling policies
- **Output**: Process performance data (6 experiments), IPC overhead analysis

### Week 4: Async Server + Comparative Analysis
- **Deliverables**: Async server (FIFO only), scheduling policy comparison (FIFO vs Priority on threaded server), final comparative analysis report
- **Output**: 
  - Async performance data (4 experiments)
  - Scheduling analysis: waiting time, starvation, fairness comparison
  - Final report with all concurrency model and scheduling policy comparisons

### Final Analysis Report Structure
1. **Executive Summary**
2. **Methodology**
   - Workloads (5 types with parameters)
   - Concurrency models (baseline/thread/process/async)
   - **Scheduling Policies (FIFO vs Priority)** ← Dedicated section
   - Metrics (latency, throughput, waiting time, CPU, context switches)
   - Platform: Windows/WSL, psutil for OS profiling
3. **Results**
   - Concurrency model comparison (tables + charts)
   - **Scheduling Policy Analysis Within the Server** ← Key section
   - GIL impact on CPU workloads
   - Process overhead analysis
   - Async sweet spot (high-concurrency I/O)
4. **Conclusions**
   - When to use threads/processes/async
   - FIFO vs Priority trade-offs (fairness vs responsiveness)
   - Memory, CPU, complexity trade-offs
5. **Future Work**
   - Advanced scheduling (SJF, MLFQ)
   - Linux perf for cache misses, CPU migrations
   - Complex ML workloads (deferred to Phase 2)

---

## Experiment Matrix (Total: ~28 runs)

| Week | Focus | Experiments |
|------|-------|-------------|
| 1 | Baseline | 12 (5 workloads × varying params × 2 user loads) |
| 2 | Threads | 9 (3 pool sizes × 3 workloads) |
| 3 | Processes | 6 (3 pool sizes × 2 workloads) |
| 4 | Async + Scheduling | 6 (2 async + 2 scheduling comparison + 2 aggregate) |

---

## Skills Demonstrated

### Operating Systems Concepts
**Core**:
- Process vs thread architecture
- Synchronization primitives (locks, queues)
- Context switching measurement
- **Scheduling algorithms (FIFO, Priority)**
- **Waiting time, starvation, fairness analysis**

**Intermediate**:
- GIL implications for Python concurrency
- IPC mechanisms (queues, pipes)
- CPU time and context switch delta tracking
- Async I/O event loop behavior

**Advanced**:
- Policy vs mechanism separation (scheduler vs executor)
- OS-level profiling with psutil
- Dispatcher design pattern for scheduling

### Software Engineering
- Reproducible experiment design
- Structured logging (JSON, JSONL)
- Performance profiling and analysis
- Clean architecture (separation of concerns)

---

## Why This Project Aligns With Course Objectives

**OS Course Focus**: Understanding operating systems through practical implementation and measurement.

**This Project Delivers**:
1. **Concurrency primitives**: Hands-on experience with threads, processes, async I/O
2. **Scheduling algorithms**: Implementation and analysis of FIFO and Priority scheduling
3. **Performance measurement**: OS-level metrics (context switches, CPU time, memory)
4. **Trade-off analysis**: Empirical data on when to use which primitive/policy
5. **Real workloads**: Diverse task suite demonstrates OS concepts with practical applications

**Academic Rigor**:
- Hypothesis-driven experiments
- Controlled variables (same workloads across all implementations)
- Quantitative metrics with statistical analysis (p50/p95/p99)
- Reproducible methodology (config files, JSONL logs, experiment.yaml)

---

## Timeline (4 Weeks)

| Week | Milestone | Hours |
|------|-----------|-------|
| 1 | Workloads + Baseline | 15-20 |
| 2 | Threading + Scheduling | 15-20 |
| 3 | Multiprocessing | 15-20 |
| 4 | Async + Analysis | 15-20 |

**Total Effort**: 60-80 hours over 4 weeks

---

## Conclusion

This project is a focused, achievable OS concurrency and scheduling study. By the end of Phase 1, we will have:

- **Built**: Three server implementations with two scheduling policies
- **Measured**: OS-level performance across diverse workloads
- **Analyzed**: Concurrency primitive and scheduling policy trade-offs
- **Documented**: Reproducible methodology and clear results

The scope is realistic for 4 weeks, the learning outcomes map directly to OS course objectives, and the deliverables are concrete and defensible.

**Next Steps**: Upon approval, begin Week 1 implementation (workloads + baseline server).

---

**Appendix: Directory Structure**

```
Process_Task-server/
├── config/
│   ├── logging_config.py
│   └── experiment.yaml
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── experiments.md
├── src/
│   ├── core/
│   │   └── job.py
│   ├── scheduler/
│   │   ├── fifo.py
│   │   └── priority.py
│   ├── dispatch/
│   │   └── dispatcher.py
│   ├── executors/
│   │   ├── thread_pool.py
│   │   ├── process_pool.py
│   │   └── async_executor.py
│   ├── workloads/
│   │   ├── cpu_bound.py
│   │   ├── io_bound.py
│   │   ├── memory_bound.py
│   │   ├── mixed.py
│   │   └── ml_simple.py
│   ├── servers/
│   │   ├── flask_app.py
│   │   └── fastapi_app.py
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       └── experiment_logger.py
├── experiments/
│   ├── locustfile.py
│   └── analysis/
│       ├── summarize.py
│       └── plot.py
├── results/
│   ├── week1_baseline/
│   ├── week2_threaded/
│   ├── week3_multiprocess/
│   ├── week4_async/
│   └── week4_scheduling/
└── logs/
```
