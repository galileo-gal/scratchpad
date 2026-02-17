# Adaptive ML Inference Server: OS Concurrency Study

**Student**: Abdullah Al Galib  
**ID**: 2232535642 &emsp;&emsp; **Sec**: 2              
**Course**: CSE-323 Operating Systems  
**Faculty**: Dr. Safat Siddiqui (SSI)  

---

## Executive Summary

**Problem**: Modern ML inference servers face a fundamental OS concurrency question - should we use threads, processes, or async I/O? Current solutions use heuristics or stick to one model. Nobody systematically measures how OS-level decisions affect ML workload performance.

**Approach**: Build an inference server with three concurrency implementations (threads, processes, async). Serve real ML models with different computational profiles. Measure how OS primitives (context switching, GIL, scheduling, IPC) affect latency and throughput. Train an ML meta-scheduler to automatically pick the best configuration based on workload patterns.

**Why OS + ML**: The OS part is building and analyzing concurrent systems with different primitives. The ML part is (1) the workload being served and (2) the intelligent scheduler that learns optimal configurations. This directly addresses how operating systems should handle AI workloads - a critical question as ML deployment scales.

---

## Core Research Questions

### Primary Question
**How do OS concurrency primitives (threads, processes, async I/O) affect ML inference server performance under varying load patterns?**

### Sub-Questions
1. When does Python's GIL make multi-threading slower than multi-processing?
2. How does context switching overhead differ between threads and processes for ML workloads?
3. Can we predict which concurrency model performs best given request characteristics?
4. How do different scheduling algorithms (FIFO, Priority, SJF) interact with concurrency choices?
5. Can an ML meta-scheduler outperform static configuration?

---

## Technical Approach

### Phase 1: Concurrency Fundamentals with Diverse Workloads (Weeks 1-4)

**Objective**: Build three versions of a general task execution server, each using a different OS concurrency primitive. Study concurrency behavior across multiple workload types (CPU-bound, I/O-bound, memory-bound, and simple ML tasks) before scaling to complex ML models in Phase 2.

**Pedagogical Approach**: Start with simpler, diverse computational tasks to isolate and understand core OS concurrency concepts (threading, multiprocessing, async I/O, scheduling). Once fundamentals are mastered, apply this knowledge to complex ML workloads in subsequent phases.

#### Workload Categories (Phase 1)

Phase 1 uses a **diverse task suite** to understand how different workload characteristics interact with concurrency primitives:

| Task Type | Example | Computational Pattern | Expected Concurrency Behavior |
|-----------|---------|----------------------|------------------------------|
| **CPU-Bound** | Matrix multiplication (NumPy) | Pure computation | Threads limited by GIL, processes scale |
| **I/O-Bound** | File operations, HTTP requests | Waiting on external resources | Threads efficient, async excellent |
| **Memory-Bound** | Large array sorting | Data movement heavy | Cache effects visible |
| **Mixed** | File read + computation | CPU + I/O combined | Reveals scheduling trade-offs |
| **ML (Simple)** | Scikit-learn prediction | CPU-bound with library overhead | Baseline for Phase 2 comparison |

**Rationale**: By testing concurrency models against diverse workloads, we can systematically identify which OS primitives suit which computational patterns, before applying these insights to complex ML models (Phase 2).

#### Implementation A: Multi-threaded Server
- **Framework**: Flask + Python threading
- **Architecture**: Thread pool (configurable size: 2, 4, 8 threads)
- **Synchronization**: Queue for request management, Lock for shared resources
- **Workloads**: All 5 task types above
- **Key Learning**: GIL impact on CPU vs I/O tasks, thread context switching cost, lock contention
```python
# Conceptual structure
class ThreadedTaskServer:
    def __init__(self, num_threads):
        self.request_queue = Queue()
        self.thread_pool = [Thread(target=self.worker) for _ in range(num_threads)]
        self.lock = Lock()
        self.workloads = {
            'cpu_bound': matrix_multiply_task,
            'io_bound': file_operation_task,
            'memory_bound': large_sort_task,
            'mixed': file_compute_task,
            'ml_simple': sklearn_prediction_task
        }
    
    def worker(self):
        while True:
            request = self.request_queue.get()
            task_func = self.workloads[request.task_type]
            with self.lock:  # Measure contention here
                result = task_func(request.data)
            self.send_response(result)
```

#### Implementation B: Multi-process Server
- **Framework**: Python multiprocessing + message passing
- **Architecture**: Process pool (2, 4 processes)
- **IPC**: Pipes for request/response
- **Workloads**: Same 5 task types for direct comparison
- **Key Learning**: Process isolation benefits, IPC overhead, memory duplication costs, CPU-bound task scaling
```python
# Conceptual structure
class ProcessTaskServer:
    def __init__(self, num_processes):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.process_pool = [Process(target=self.worker) for _ in range(num_processes)]
        # Each process loads workloads independently
    
    def worker(self):
        workloads = load_task_suite()  # Per-process loading
        while True:
            request = self.task_queue.get()
            result = workloads[request.task_type](request.data)
            self.result_queue.put(result)
```

#### Implementation C: Async I/O Server
- **Framework**: FastAPI + asyncio
- **Architecture**: Event loop with async/await
- **Workloads**: Same 5 task types
- **Key Learning**: When non-blocking I/O helps (I/O-bound tasks), when CPU-bound tasks block the event loop, async vs threading trade-offs
```python
# Conceptual structure
from fastapi import FastAPI
import asyncio

app = FastAPI()
workloads = load_task_suite()

@app.post("/execute/{task_type}")
async def execute_task(task_type: str, data: dict):
    task_func = workloads[task_type]
    # Use thread pool for CPU-bound tasks to avoid blocking event loop
    result = await asyncio.to_thread(task_func, data)
    return {"result": result}
```

#### Basic Scheduling (Introduced in Phase 1)

Implement **FIFO** and **Priority-based** scheduling with the diverse workload mix:

**FIFO (Baseline)**:
- Simple queue, process in arrival order
- Expected result: fair but inefficient (head-of-line blocking when slow tasks arrive first)

**Priority-based**:
- Assign priority based on task type: `priority = {'ml_simple': 1, 'io_bound': 2, 'memory_bound': 3, 'cpu_bound': 4}`
- Prevents fast tasks waiting behind slow ones
- Demonstrates basic scheduling impact

**Note**: Advanced scheduling (SJF with ML prediction, MLFQ) remains in Phase 3.

#### Phase 1 Deliverables

1. **Task Suite**: 5 characterized workloads with measured baseline performance
2. **Three Servers**: Threaded, process-based, and async implementations
3. **Performance Data**: Latency, throughput, CPU usage, context switches across all (3 concurrency models × 5 task types)
4. **Basic Scheduling**: FIFO vs Priority comparison
5. **Analysis Report**: 
   - Which concurrency model suits which workload type
   - GIL effects quantified
   - IPC overhead measured
   - Foundation for Phase 2 ML-specific analysis

**Transition to Phase 2**: With concurrency fundamentals established using diverse tasks, Phase 2 loads complex ML models (MobileNet, DistilBERT, LSTM) into the same infrastructure to study ML-specific concurrency challenges.
---

### Phase 2: ML Workload Characterization (Weeks 3–6)

**Objective**: Load real ML models with different computational profiles. Understand how model characteristics interact with concurrency choices.

#### Models Selected

| Model | Task | Inference Time | Memory | CPU Pattern | Why Chosen |
|-------|------|---------------|---------|-------------|------------|
| MobileNetV2 (CIFAR-10) | Image classification | ~5ms | 14MB | Burst compute | Fast, tensor ops |
| DistilBERT (IMDB) | Sentiment analysis | ~20ms | 250MB | Sequential compute | Medium, attention layers |
| LSTM Time-Series | Forecasting | ~100ms | 50MB | Recurrent compute | Slow, memory-bound |

#### Dataset Sources
1. **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html (170MB, 60k images)
2. **IMDB Reviews**: https://ai.stanford.edu/~amaas/data/sentiment/ (80MB, 50k reviews)
3. **ETTh1 (Electricity Transformer Temperature)**: https://github.com/zhouhaoyi/ETDataset (10MB, hourly readings)

#### Characterization Metrics
For each model, measure:
- **Inference time distribution** (mean, p50, p95, p99)
- **CPU utilization pattern** (burst vs sustained)
- **Memory footprint** (working set size)
- **Cache behavior** (L1/L2/L3 miss rates via `perf stat`)
- **Thread scaling** (speedup from 1→2→4→8 threads)

**Deliverable**: Model performance profiles, identifying which models are thread-friendly vs process-friendly

---

### Phase 3: Scheduling Algorithms (Weeks 5-10)

**Objective**: Implement multiple request scheduling strategies. Measure how scheduling interacts with concurrency model.

#### Scheduling Strategies

**1. FIFO (Baseline)**
- Simple queue, process in arrival order
- Expected result: fair but inefficient (head-of-line blocking)

**2. Priority-Based**
- Priority = 1 / expected_inference_time (shorter jobs = higher priority)
- Prevents fast requests waiting behind slow ones
- Requires: inference time prediction

**3. Shortest Job First (SJF)**
- Requires accurate runtime prediction
- ML predictor: `request_features → predicted_inference_time`
- Features: model_id, input_size, current_system_load

**4. Multi-Level Feedback Queue (MLFQ)** (Stretch)
- Start all requests in high-priority queue
- Demote to lower priority if exceeds time quantum
- Adaptive without prediction

#### ML Runtime Predictor

**Model**: Gradient Boosting Regressor (scikit-learn)

**Features**:
- model_id (categorical)
- input_size (bytes)
- current_queue_length
- current_cpu_utilization
- time_of_day (captures load patterns)

**Training data**: Collect from Phase 2 experiments

**Target**: actual inference time (ms)

**Validation**: Mean Absolute Percentage Error < 15%

```python
# Training pipeline
from sklearn.ensemble import GradientBoostingRegressor

features = ['model_id', 'input_size', 'queue_len', 'cpu_util', 'hour']
X_train, y_train = collect_historical_data()

predictor = GradientBoostingRegressor(n_estimators=100, max_depth=5)
predictor.fit(X_train, y_train)

# Usage in scheduler
def schedule_next_request(queue):
    predictions = [predictor.predict(req.features) for req in queue]
    return queue[np.argmin(predictions)]  # SJF
```

**Deliverable**: Four scheduling algorithms implemented, performance compared across all concurrency models

---

### Phase 4: Meta-Optimization (Weeks 9-14)

**Objective**: Build an intelligent system that automatically selects the best (concurrency_model, scheduling_algorithm) pair based on real-time conditions.

#### Meta-Scheduler Architecture

**Input State**:
- Current request rate (req/sec)
- Request type distribution (% fast, % medium, % slow models)
- System load (CPU %, memory %)
- Recent latency metrics (p95, p99)

**Output Decision**:
- Concurrency model: {threads, processes, async}
- Scheduling algorithm: {FIFO, Priority, SJF}
- Configuration: {thread_count, process_count, ...}

**Learning Approach**: Multi-Armed Bandit (ε-greedy)

**Why MAB**: Exploration-exploitation trade-off. We want to try different configs but also exploit known-good ones.

```python
# Simplified meta-scheduler
class MetaScheduler:
    def __init__(self):
        self.configs = [
            ('threads', 'FIFO', 8),
            ('threads', 'SJF', 8),
            ('processes', 'FIFO', 4),
            # ... 12 total configs
        ]
        self.rewards = {config: [] for config in self.configs}
        self.epsilon = 0.1  # Exploration rate
    
    def select_config(self, system_state):
        if random.random() < self.epsilon:
            return random.choice(self.configs)  # Explore
        else:
            # Exploit: pick config with best recent reward
            avg_rewards = {c: np.mean(self.rewards[c][-100:]) for c in self.configs}
            return max(avg_rewards, key=avg_rewards.get)
    
    def update_reward(self, config, latency_p99, throughput):
        # Reward = high throughput, low latency
        reward = throughput / (1 + latency_p99)
        self.rewards[config].append(reward)
```

**Alternative (More Advanced)**: Contextual bandit or RL with system state as input

**Validation**: Compare meta-scheduler against:
- Best static configuration
- Random selection
- Round-robin between configs

**Success Metric**: Meta-scheduler achieves within 5% of oracle (best possible config per workload)

**Deliverable**: Adaptive system that switches configurations in real-time, outperforms static choices

---

## OS Concepts Covered (Proof This Is OS)

| OS Concept | How It's Studied | Measurement |
|------------|------------------|-------------|
| **Process vs Thread** | Build both, measure overhead | Context switch time (via `perf`) |
| **Context Switching** | Vary concurrency level, measure CPU time vs wall time | `getrusage()`, kernel time tracking |
| **Synchronization Primitives** | Locks in threaded server, measure contention | Lock wait time, deadlock scenarios |
| **IPC Mechanisms** | Pipes, queues in multiprocessing | IPC overhead vs shared memory |
| **Python GIL** | Compare threads with/without C extension | CPU parallelization efficiency |
| **CPU Scheduling Effects** | SJF vs FIFO, measure starvation | Waiting time distribution |
| **Memory Models** | Shared (threads) vs isolated (processes) | Memory usage per concurrency model |
| **Cache Effects** | Thread affinity experiments | `perf stat` cache miss rates |

### Deep Dive Topics (For Report/Presentation)

1. **Why does GIL exist?** (Memory management, reference counting)
2. **When does multiprocessing outweigh IPC cost?** (Crossover analysis)
3. **Copy-on-write in fork()** (Linux optimization for processes)
4. **CPU affinity and NUMA** (Pinning threads to cores)
5. **Scheduler activations** (Kernel-user space cooperation)

---

## Implementation Timeline

### Week-by-Week Breakdown

| Week | Phase | Tasks | Deliverable |
|------|-------|-------|-------------|
| 1 | Setup | Environment setup, model loading, basic Flask server | Single-threaded working server |
| 2 | Phase 1 | Implement threaded server with metrics | Threaded server + instrumentation |
| 3 | Phase 1 | Implement process-based server | Process server comparison |
| 4 | Phase 1 | Implement async server, C extension (optional) | All three servers |
| 5 | Phase 2 | Load 3 models, characterization experiments | Model profiles |
| 6 | Phase 2 | Cache analysis, thread scaling study | Detailed perf report |
| 7 | Phase 3 | FIFO and Priority scheduling | Two schedulers working |
| 8 | Phase 3 | Train runtime predictor, implement SJF | ML predictor + SJF |
| 9 | Phase 3 | Comprehensive scheduling comparison | Scheduler performance report |
| 10 | Phase 4 | Design meta-scheduler, collect training data | Meta-scheduler v1 |
| 11 | Phase 4 | Train and validate meta-scheduler | Adaptive system |
| 12 | Phase 4 | Stress testing, edge case analysis | Robust system |
| 13 | Analysis | Data analysis, visualization | Graphs, tables |
| 14 | Documentation | Write report, prepare presentation | Final deliverables |


## Tools and Technologies

### Development Stack

| Component | Tool/Library | Why |
|-----------|--------------|-----|
| **Web Framework** | Flask (threads/processes), FastAPI (async) | Well-documented, easy instrumentation |
| **ML Framework** | PyTorch / TensorFlow | Model loading and inference |
| **Models** | Pre-trained from HuggingFace | torchvision, transformers libraries |
| **Concurrency** | threading, multiprocessing, asyncio | Python stdlib |
| **C Extension** | Python.h, Cython (optional) | GIL bypass for advanced tier |
| **Monitoring** | psutil, py-spy, perf | CPU, memory, profiling |
| **Load Testing** | Locust, wrk | Generate realistic traffic |
| **ML Training** | scikit-learn, pandas | Runtime predictor, meta-scheduler |
| **Visualization** | matplotlib, seaborn | Performance graphs |
| **Profiling** | cProfile, line_profiler, perf | Bottleneck identification |

### System Requirements
- **OS**: Linux (Ubuntu 22.04 on WSL2 or native)
- **Python**: 3.10+
- **RAM**: 16GB (sufficient for all models + concurrent processes)
- **CPU**: 4+ cores (i5 11th gen tested, 4 cores / 8 threads)
- **Storage**: 5GB free (models + datasets + code)

### Development Environment
- **Primary**: WSL2 Ubuntu on Windows 11
- **IDE**: VS Code with WSL extension (or any text editor)
- **Terminal**: Windows Terminal with WSL2 profile

### Hardware Specifications (Confirmed Available)
- **CPU**: Intel Core i5-11th Generation (4 physical cores, 8 logical threads via Hyper-Threading)
- **RAM**: 16GB DDR3 (2 × 8GB sticks)
- **Storage**: 256GB SSD with 50GB+ available space
- **Machine**: Dell Inspiron Laptop
- **OS**: Windows 11  with WSL2 (Ubuntu 22.04 LTS)

*Performance Note*: All throughput and latency targets are calibrated for this hardware profile. The 4-core CPU limits process pool to 4 workers maximum, which is optimal for measuring process vs thread trade-offs.

## Success Metrics

### Quantitative Goals

1. **Concurrency Comparison**:
   - Measure p99 latency difference between threads/processes/async under identical load
   - Target: Identify crossover point (when processes outperform threads)

2. **Scheduling Impact**:
   - Show SJF reduces average waiting time by ≥20% vs FIFO for mixed workload
   - Predictor accuracy: MAPE < 15%

3. **Meta-Scheduler Performance**:
   - Achieve ≥90% of oracle performance (best possible config per workload)
   - Adaptation time: Switch config in <5 seconds after workload shift

4. **Throughput**:
   - Handle ≥500 req/sec sustained load
   - System remains stable under 2x overload

### Qualitative Goals

1. **Understanding**: Articulate why threads fail under CPU-bound load (GIL)
2. **Insight**: Identify which model characteristics predict thread-friendliness
3. **Documentation**: Reproducible experiments with clear methodology

---

## Learning Outcomes

### OS Mastery (What I'll Learn)

**Foundational**:
- Process lifecycle (fork, exec, wait)
- Thread lifecycle (create, join, detach)
- Context switching mechanics
- Synchronization primitives (mutex, semaphore, condition variable)

**Intermediate**:
- GIL implications for Python concurrency
- IPC mechanisms (pipes, queues, shared memory)
- Copy-on-write memory optimization
- CPU affinity and scheduling effects

**Advanced**:
- Writing Python C extensions
- Kernel-level profiling with perf
- NUMA awareness
- Lock-free data structures (if time permits)

### ML Infrastructure Skills

1. **Model Serving**: How to deploy ML models in production
2. **Performance Engineering**: Profiling and optimizing inference pipelines
3. **Load Balancing**: Request scheduling in distributed systems
4. **Auto-scaling**: Adaptive resource allocation
5. **Observability**: Metrics, logging, tracing for ML systems

### Career Relevance

**Direct Applications**:
- **MLOps**: This IS model serving infrastructure
- **Distributed Training**: Understanding process/thread trade-offs applies to multi-GPU training
- **Cloud ML**: Same principles as AWS SageMaker, GCP Vertex AI
- **Real-time AI**: Low-latency serving for recommendation systems, fraud detection

**Concepts Map To**:
- Kubernetes pod scheduling → our meta-scheduler
- Ray distributed execution → our concurrency models
- TensorFlow Serving → our inference server architecture

---

## Related Academic Work

### Papers That Inspired This

1. **"Clipper: A Low-Latency Online Prediction Serving System"** (NSDI 2017)
   - Addresses model selection and batching for inference
   - Our scheduler extends their prediction caching ideas

2. **"TensorFlow: A System for Large-Scale Machine Learning"** (OSDI 2016)
   - Discusses parallelism strategies for ML workloads
   - We focus on inference, they focus on training

3. **"Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads"** (ATC 2019)
   - Studies scheduling in GPU clusters
   - Our CPU-based study is analogous

4. **"Towards ML-Centric Cloud Platforms"** (ACM Computing Surveys 2020)
   - Reviews ML infrastructure challenges
   - We tackle concurrency and scheduling specifically

### How This Extends Prior Work

**Novel Contribution**: Systematic comparison of OS concurrency primitives specifically for ML inference, with an adaptive meta-scheduler. Most prior work assumes one concurrency model; we compare all three and learn which to use.

---

## Expected Outputs


### Demonstration Plan

**Live Demo**: Show meta-scheduler adapting in real-time
1. Start with low load (fast model) → system picks threads
2. Shift to high load (slow model) → system switches to processes
3. Show metrics dashboard updating
4. Explain why the switch happened (OS concepts)



---

## Why This Project Aligns With My Goals

I'm pursuing a career in AI/ML engineering, specifically focusing on MLOps and production machine learning systems. While many students concentrate solely on model development and algorithms, I recognize that **deployment and infrastructure are where most AI projects fail in production**.

### Current Gap in Knowledge
- **What I know**: Machine learning algorithms, model training with Python (scikit-learn, basic PyTorch)
- **What I'm missing**: Understanding of operating systems infrastructure that powers ML at scale
- **This project addresses**: How OS-level decisions (concurrency, scheduling, memory management) affect real ML workloads

### Learning Objectives
By completing this project, I aim to confidently answer:
- Why do production inference servers struggle under load?
- How do I profile and optimize concurrent systems?
- What architecture decisions matter for ML deployment?
- When should I use threads vs processes vs async in production ML systems?

### Career Direction
**Target Role**: MLOps Engineer or ML Infrastructure Engineer at scale-ups or tech companies building production AI platforms (similar to teams at companies running large-scale ML inference systems).

**Why This Matters**: The difference between academic ML and production ML is the infrastructure layer. This project is my bridge from "can train models" to "can deploy models reliably at scale with proper systems engineering."

### Current Preparation
- **Systems Knowledge**: Basic understanding of processes, threads, memory from coursework; ready to dive deep
- **Linux Experience**: Minimal currently, but prepared to learn (installing WSL2, learning command line tools as part of this project)
- **ML Background**: Comfortable with Python ML libraries; have worked on classification and regression projects
- **Time Commitment**: Dedicating 15-20 hours/week to this project alongside coursework

I'm excited about this intersection of OS and ML because it directly maps to real-world infrastructure challenges in AI deployment.

---

## Conclusion

This project sits at the intersection of operating systems and machine learning infrastructure. It's fundamentally an OS project - we're studying how concurrency primitives affect performance - but uses ML workloads because they're realistic, measurable, and career-relevant.

**The OS part**: Building concurrent systems, measuring context switching, understanding scheduling, analyzing synchronization overhead.

**The ML part**: Serving real models, predicting performance, building an intelligent meta-scheduler.

**The learning**: By the end, I'll understand why Kubernetes schedules pods the way it does, why Ray uses processes for actors, and why every ML platform struggles with the same concurrency questions we're studying here.

I'm excited to explore this. Looking forward to discussing scope and feasibility.

---

**Appendix A: Quick Reference - Key Metrics**

| Metric | How to Measure | Why It Matters |
|--------|----------------|----------------|
| Latency (p50, p95, p99) | Time from request arrival to response | User experience |
| Throughput (req/sec) | Completed requests per second | System capacity |
| CPU Utilization (%) | `psutil.cpu_percent()` | Efficiency |
| Context Switch Rate | `perf stat -e context-switches` | OS overhead |
| Lock Contention Time | Custom instrumentation around locks | Synchronization cost |
| Memory Footprint (MB) | `psutil.Process().memory_info()` | Resource usage |
| Cache Miss Rate (%) | `perf stat -e cache-misses` | Memory hierarchy effects |
| IPC Overhead (μs) | Timing pipe/queue operations | Process communication cost |

**Appendix B: Simplified Architecture Diagram**

```
                     Load Generator (Locust)
                              │
                              ▼
                   ┌──────────────────────┐
                   │   Load Balancer      │
                   │  (Meta-Scheduler)    │
                   └──────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Threaded   │   │   Process   │   │    Async    │
    │   Server    │   │   Server    │   │   Server    │
    └─────────────┘   └─────────────┘   └─────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
                   ┌──────────────────────┐
                   │    ML Models         │
                   │  (MobileNet, BERT,   │
                   │   LSTM)              │
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Metrics Collector   │
                   │  (Prometheus-style)  │
                   └──────────────────────┘
```