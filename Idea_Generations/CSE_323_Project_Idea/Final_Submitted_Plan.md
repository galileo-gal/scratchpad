# Adaptive ML Inference Server: OS Concurrency Study

**Student**: [Your Name]  
**Course**: CSE-323 Operating Systems  
**Faculty**: [Faculty Name]  
**Submission Date**: [Date]

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

### Phase 1: Multi-Model Concurrency Server (Weeks 1-4)

**Objective**: Build three versions of the same ML inference server, each using a different OS concurrency primitive.

#### Implementation A: Multi-threaded Server
- **Framework**: Flask + Python threading
- **Architecture**: Thread pool (configurable size: 2, 4, 8 threads)
- **Synchronization**: Queue for request management, Lock for shared model access
- **Key Learning**: GIL impact, thread context switching cost, lock contention

```python
# Conceptual structure
class ThreadedInferenceServer:
    def __init__(self, models, num_threads):
        self.request_queue = Queue()
        self.thread_pool = [Thread(target=self.worker) for _ in range(num_threads)]
        self.models = models  # Shared memory
        self.lock = Lock()
    
    def worker(self):
        while True:
            request = self.request_queue.get()
            with self.lock:  # Measure contention here
                result = self.models[request.model_id].predict(request.data)
            self.send_response(result)
```

#### Implementation B: Multi-process Server
- **Framework**: Python multiprocessing + message passing
- **Architecture**: Process pool (2, 4)
- **IPC**: Pipes for request/response, shared memory for models (to avoid duplication)
- **Key Learning**: Process isolation benefits, IPC overhead, memory duplication costs

```python
# Conceptual structure
class ProcessInferenceServer:
    def __init__(self, models, num_processes):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.shared_models = self.load_to_shared_memory(models)
        self.process_pool = [Process(target=self.worker) for _ in range(num_processes)]
```

#### Implementation C: Async I/O Server
- **Framework**: FastAPI + asyncio
- **Architecture**: Event loop with async/await
- **Key Learning**: When non-blocking I/O helps, when CPU-bound tasks break async model

```python
# Conceptual structure
async def handle_inference(request):
    model = models[request.model_id]
    # CPU-bound operation - this blocks the event loop
    result = await asyncio.to_thread(model.predict, request.data)
    return result
```

#### C Extension for GIL Bypass (Advanced/Optional)
- **Objective**: For CPU-intensive inference, release GIL in C extension
- **Implementation**: Wrap critical NumPy operations in Py_BEGIN_ALLOW_THREADS / Py_END_ALLOW_THREADS
- **Measurement**: Compare threaded server with/without GIL release

**Deliverable**: Three working servers, instrumented for metrics collection

---

### Phase 2: ML Workload Characterization (Weeks 3-6)

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

## Questions for Faculty

1. **Scope Preference**: Should I target Tier 2 (realistic) or Tier 3 (ambitious with risk)?
2. **C Extension**: Is bypassing GIL in scope, or should I stick to pure Python?
3. **Evaluation**: Would you prefer depth (one concurrency model, very detailed) or breadth (all three, less deep)?
4. **Collaboration**: Can this extend beyond one semester if results are promising?
5. **Resources**: Any lab machines available, or should I run everything on my laptop?

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