# CUDA graph union memory: research and implementation plan

Status: research proposal; source and API review complete, GPU validation pending.
Baseline reviewed: `9d164f0a44` in this repository; CUDA requirements pin PyTorch
2.14.0. The memory and performance targets below are not measured results.

Reduce GPU memory attributable to a collection of captured CUDA graphs while
preserving its supported batch shapes, numerical behavior, and replay performance.
The working hypothesis is that mutually exclusive executions can reuse an arena
whose capacity approaches the largest required workspace among those executions.
Whether the maximum-token graph supplies that capacity is an experiment, not an
assumption of the implementation.

GPU and LLM independence is a project requirement. Derive arena layouts and
eligibility from allocation sizes, lifetimes, alignment, and execution dependencies
for the active configuration. Keep device allocation details behind a backend
interface and avoid GPU-name or model-name rules. Each deployment can have a
different memory plan while using the same planning and management technique.
The first implementation uses CUDA graph APIs. Portability to another execution
backend requires equivalent allocation and ordering capabilities and separate
validation; the planner's interface should make these requirements explicit.

Here, **union memory** means overlaying temporary allocations of executions that
cannot overlap onto shared GPU storage. Each shape keeps its own captured kernels
and launch parameters. Replaying the largest graph for every smaller batch is a
different strategy with different computation and padding costs.

## What the checkout already does

| Existing behavior | Local source |
| --- | --- |
| Uses a global graph pool and weak references for captured outputs | [CUDAGraphWrapper](../../vllm/compilation/cuda_graph.py) |
| Captures larger token counts first within each mode | [CudagraphDispatcher.get_capture_descs](../../vllm/v1/cudagraph_dispatcher.py) |
| Estimates shared memory separately from incremental per-graph overhead | [GPUModelRunner.profile_cudagraph_memory](../../vllm/v1/worker/gpu_model_runner.py) |
| Shares pools in model runner V2 too | [CudaGraphManager](../../vllm/v1/worker/gpu/cudagraph_utils.py) |
| Routes graph allocations through a separate allocator path for sleep support | [capture_pool](../../vllm/compilation/cudagraph_pool.py) |
| Captures a configured or generated set of token sizes, not necessarily every integer | [CompilationConfig](../../vllm/config/compilation.py) |

The profiler in `gpu_model_runner.py` samples up to two decoder descriptors per
mode and extrapolates later costs. Model runner V2's profiler in
`gpu/cudagraph_utils.py` samples FULL graphs but measures PIECEWISE, encoder, and
speculator captures in full. Both use a minimum per-graph allowance in their
extrapolation; this is not evidence of an irreducible CUDA requirement. Research
measurements must cover every descriptor and graph object, including pieces of
piecewise graphs. Record which runner produced each result.

Simply sharing a pool or changing capture order would reproduce existing behavior.
The potential contribution is a demonstrated reduction in the residual memory
growth through better allocation layouts, lifetimes, or workspace ownership.

PyTorch explicitly supports sharing a pool among independent graphs that never
execute concurrently. Their replay order can vary, but replay can overwrite a
previous graph's outputs. Dependent graph sequences have additional ordering and
lifetime constraints. Thus arbitrary sequential batch selection is already a
supported pattern; our obligation is to preserve output consumption and storage
lifetimes under a new layout. See [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#sharing-memory-across-captures).

Related mechanisms to compare before choosing an implementation include
[PyTorch CUDAGraph Trees](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html),
which track live allocations when reusing a pool, and
[CUDA graph memory nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#graph-memory-nodes),
which permit physical memory reuse for suitable graph allocation lifetimes.
CUDA graph memory nodes and PyTorch's ordinary graph pool are different allocation
mechanisms. Neither establishes that a new vLLM allocator will save memory.
Verify API availability against the eventual pinned PyTorch/CUDA versions.

## Memory model and design contract

For accounting, use nonoverlapping categories:

```text
process GPU memory = model/KV/runtime baseline
                   + persistent graph buffers
                   + reusable graph workspace, including allocator slack
                   + graph executable/driver and other external allocations
```

An execution family groups complete executions eligible to overlay scratch
storage. Family membership depends on the capture mode, full batch descriptor,
backend, dtype, persistent state, and concurrency schedule, not just token count.
A piecewise family includes the eager sections and tensors live between pieces.
For each family, aim for:

```text
shared arena capacity ~= max(required workspace over all family members)
```

Peak simultaneously live scratch bytes give a lower bound. Alignment,
fragmentation, and layout constraints can make the required capacity larger.
The maximum token count need not produce the maximum workspace: kernel choices,
attention metadata, and execution paths can change with shape.
The largest standalone measured workspace is an empirical reference, not a proven
minimum. Compare it under the same allocator settings and persistent-buffer policy.
Count physical backing storage once when virtual addresses alias it.

The candidate architecture keeps a graph executable per existing descriptor and
assigns reusable intermediate storage to fixed offsets in a common arena before
capture. Input/state/output buffers with longer lifetimes need explicit ownership.
Graph executable overhead remains a separate cost and may still scale with graph
count. We will not promise constant total graph memory.

Required invariants:

1. Every captured pointer remains valid and has the captured address at replay.
2. Allocations with overlapping GPU lifetimes never occupy overlapping storage.
3. Each replay writes each scratch region before reading it, with valid GPU
   ordering. Persistent values such as
   weights, KV cache, counters, and RNG state retain their intended lifetimes.
4. Arena reuse waits for graph execution and downstream consumers, including
   asynchronous output copies. Host return from `replay()` does not imply completion.
5. Any simultaneously executing work uses disjoint storage or a proven schedule.
   Serialization introduced solely for sharing must be counted as a performance cost.
6. A piecewise execution is analyzed with its eager sections and inter-piece live
   tensors. Its pieces cannot independently reset the same arena.
7. Unsupported layouts fall back to the existing capture path. A published arena
   cannot be relocated while captured graphs retain its pointers; resizing requires
   retaining old storage or rebuilding the affected graphs safely.

## Scope and milestones

The design targets GPU- and LLM-agnostic graph memory management. The first
validation case uses one available GPU, a small dense decoder, one attention
backend, one dtype, and sequential decode with `FULL_DECODE_ONLY`. This bounds
experimental complexity without defining the technique's hardware or model scope.
Pin the actual model runner and compilation mode for each experiment.

Expand to piecewise and mixed modes after the full-decode result is established.
Multi-GPU collectives, overlapping microbatches, LoRA, speculative decoding, MoE,
multimodal encoders, and sleep/wake are later compatibility work. Until validated,
those configurations retain the existing allocation path.

As resources become available, validate across at least two GPU architectures and
two model families, including a structurally different case such as MoE or a hybrid
model, plus different attention backends and full/piecewise capture modes. Report
tested combinations and capability-based restrictions explicitly. Generality is
supported by these experiments, not inferred from one successful dense model.

Indicative schedule: four to six engineering weeks for the initial prototype and
evaluation with available GPU access. Broader compatibility may require more time.
Stages depend on the previous stage's evidence; the schedule is not a promise of
a positive research result.

| Stage | Effort | Deliverable and exit condition |
| --- | --- | --- |
| 0. Reproducible baseline | 1–2 days | Pin hardware, driver, CUDA, PyTorch, vLLM revision, model revision, backend, allocator, runner, and capture settings. Reproduce capture and replay. |
| 1. Attribute memory growth | 3–5 days | Per-capture measurements and allocation traces; explain which categories grow, estimate recoverable bytes, and decide whether an arena project is justified. |
| 2. Prove sharing on a small workload | 3–5 days | A layout simulator and minimal graph collection with controlled lifetimes; demonstrate correctness and savings beyond the current shared-pool baseline. |
| 3. Integrate one vLLM path | 5–8 days | An opt-in implementation for the pinned runner, with fallback, bounded arena lifetime, diagnostics, and real-model correctness evidence. |
| 4. Evaluate and broaden | 5–8 days | Repeated memory/performance experiments, model evaluation, mixed-mode follow-up where feasible, and a written proposal with measured limitations. |

The implementation owner maintains the tooling, code, and evidence log, and selects
representative validation cases. The project owner coordinates resources and
milestone decisions. A human reviewer validates conclusions, code, and test results
before an upstream contribution. These are responsibilities, not staffing assumptions.
At each stage, record results, open questions, and the next decision in the report.

## Stage 1: the decisive experiment

Run independent fresh processes for each configuration. Hold weights, KV capacity,
maximum sequence length, attention backend, compilation, and input distribution
constant. Warm up all tested shapes consistently and separate compilation or lazy
library initialization from capture costs. Also report actual startup peak usage.

Use a small explicit set such as `1, 2, 4, 8, 16, 32, 64, 128` first, bounded by
the selected configuration. Then test the production capture set. Sweep graph
count at a fixed maximum size and maximum size at a fixed capture policy. Include
shapes near discovered kernel-selection boundaries and more than one KV length.

Compare:

- Current vLLM sharing and capture order: the primary performance/memory baseline.
- Only the maximum-token descriptor: a capacity reference with different coverage.
- Each descriptor captured alone: find the largest measured standalone workspace,
  including all pieces for a piecewise execution.
- Separate pools per shape: a diagnostic control to quantify existing sharing.
- The proposed allocator with the same graph set and dispatch coverage as baseline.

For each warmup, capture, first replay, and steady replay, record descriptor, graph
count, allocator allocated/reserved/peak bytes, pool segments, process/device memory,
capture latency, and replay latency. Trace sizes, addresses, alignment, lifetime,
stream, and ownership for allocations where tooling permits. Synchronize at memory
measurement boundaries and run without competing GPU jobs.

Measure memory in two passes: an instrumented diagnostic pass and an uninstrumented
capture/replay pass. Profiling can change memory use and must not supply the final
latency numbers. Replay every captured graph before measuring steady residency to
include lazy initialization. Sample startup peaks with sufficient resolution and
report that resolution; a before/after free-memory delta is not a peak measurement.

Do not add allocated bytes to reserved bytes: allocated bytes are included in
reserved bytes. Compare allocator snapshots with device/process measurements to
identify external allocations. Unexplained differences remain unattributed until
controlled experiments identify them. Destroying graph executables while retaining
an externally owned arena is one useful diagnostic when supported.

Track graph-owned allocations even when they occur before the capture loop, so
moving an allocation into setup cannot appear as a saving. Allocator counters and
host allocation/free traces do not establish GPU lifetimes: reconstruct stream
dependencies and output consumers as well. `cudaDeviceGetGraphMemAttribute` reports
the CUDA graph-memory-node pool; it is not a counter for all PyTorch graph storage
or graph executable overhead. See [CUDA memory accounting](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#physical-memory-footprint).

Separate two eventual deployment evaluations: fixed KV capacity to expose savings,
and fixed total GPU budget to show whether savings buy additional KV blocks or
request capacity. Automatic KV sizing can otherwise hide the memory reduction.

Decision: continue with arena work if measured avoidable workspace growth is
material relative to the measured noise and implementation cost. Record an
absolute MiB target and a fraction of graph-attributable memory before tuning.
If the pool already approaches the largest standalone workspace and
external executable overhead dominates, document that result and reconsider the
research direction. Graph-count reduction or executable updates would require a
separate design and performance argument.

## Stages 2–3: select and integrate the mechanism

Start with the smallest mechanism supported by the measurements:

1. Repair unnecessarily retained buffers or allocation-order fragmentation if
   that explains the growth. Measure this against the proposed arena approach.
2. Prototype explicit maximum-capacity buffers for a small captured workload, then
   test a layout planner that maps independent graph scratch lifetimes into one
   arena. Include a deliberately non-monotonic workspace case and a case that
   requires preserving an output beyond replay.
3. Choose between explicit workspace plumbing, a supported capture-aware allocator
   integration, or compiler memory planning based on which allocations must move.
   Verify whether a PyTorch change is needed before promising a vLLM-only solution.
   A preallocated tensor alone does not redirect arbitrary operator temporaries.
4. Consider virtual-memory or graph-memory-node integration only if simpler
   mechanisms cannot reach the measured opportunity. Validate API restrictions and
   mapping costs independently; replay-time remapping is not assumed free.

Profile and plan before the final capture, so kernel arguments contain the intended
addresses. Never reclaim live capture allocations merely by rewinding allocator
bookkeeping. Trace any internal streams and library workspaces that escape the
chosen allocator, and include their memory in results.

Planner input must identify allocation size/alignment, aliases, GPU lifetime
dependencies, and whether storage escapes the execution. Its output assigns
arena offsets and identifies allocations kept outside the arena. An independent
layout check rejects conflicting live ranges before capture. Missing lifetime or
ownership information makes an allocation ineligible for overlay.

Bind each plan to the configuration and allocation trace that produced it. A
changed capture set, backend, compilation result, or workspace requirement triggers
revalidation and, if needed, recapture or fallback. The first prototype plans all
descriptors before publishing graphs; online arena growth is later work.

The likely integration points are `CUDAGraphWrapper` and the selected runner's
capture loop/manager. Keep descriptor dispatch semantics intact. Treat
`capture_pool`, sleep support, graph destruction, and capture failure cleanup as
ownership boundaries to audit. Implement the smallest supported path first, then
evaluate whether the second runner can reuse the same abstraction.

## Validation and acceptance

Before each new test, specify its purpose, public I/O contract, failure mode, and
cheapest effective test level. Extend the existing suites in `tests/v1/cudagraph/`
and `tests/compile/fullgraph/` where they fit. Use `tests/cudagraphs/` only for
correctness cases without an appropriate existing home.

Test ascending, descending, repeated, random, and alternating minimum/maximum
shapes, including unaligned requests that dispatch to padded shapes. Compare
outputs and updated KV state with the existing graph path and an eager reference
using justified dtype tolerances and identical initial inputs and mutable state.
Restore KV and other state between equivalent reference/prototype runs; separately
exercise full sequences that carry state across steps. Check preserved outputs,
live views, graph teardown, failed capture cleanup, and fallback behavior. Use Compute Sanitizer on
the small reproducer and targeted GPU tests. Complete real-model generation and
the relevant existing model evaluation before claiming correctness.

Define metrics for the same descriptor set, persistent-buffer policy, and warmup:

```text
W_ref   = maximum standalone workspace reserved across descriptors
E_base  = max(0, baseline shared workspace reserved - W_ref)
E_union = max(0, proposed shared workspace reserved - W_ref)
```

Workspace includes allocator slack. Report graph-attributable persistent buffers
and external allocations separately, then include them in total graph memory.
These empirical metrics do not claim an optimal packing bound.

Provisional targets, to finalize after Stage 1 and before prototype tuning:

- Reduce `E_base` by at least 50%, measured as `(E_base - E_union) / E_base`,
  only when `E_base` is positive and exceeds measurement uncertainty. Aim for
  workspace within 10% of `W_ref` where layouts permit.
- Demonstrate at least 20% reduction in total graph-attributable GPU memory on
  two meaningful configurations, reporting absolute MiB and whole-process savings.
- Keep median decode latency regression below 2% and p95 regression below 5%,
  with repeated runs, noise estimates, and the same capture/dispatch coverage.
- Preserve correctness and show no unexplained memory growth during long shape
  switching. Report startup peak and capture time separately; investigate a
  capture-time increase above 20% before expanding scope.

These are project targets, not measured results. Small synthetic savings alone
are insufficient. If the available opportunity is below these targets, report it
and revise the project decision explicitly rather than changing the baseline.

Use at least five fresh-process runs per primary configuration, alternate baseline
and prototype order, and report variability and confidence intervals for paired
comparisons. Gather enough replay samples for p95. Measure both GPU execution and
host dispatch/management time; a CPU-side regression may be absent from CUDA-event
timings. Validate the actual dispatch sequence and padded sizes in every comparison.

Use a harness that exercises the modified capture manager for primary results.
`vllm bench forward` provides complementary per-shape timings, but its
[capture implementation](../../vllm/benchmarks/forward/capture.py) uses a one-shot
graph and explicitly bypasses vLLM's graph manager. Its
[setup](../../vllm/benchmarks/forward/main.py) also disables compilation. It cannot
establish the proposed memory savings or production replay latency by itself.
Use real weights for correctness and `vllm bench serve` for eventual serving
throughput, TTFT, and inter-token latency measurements.

Place graph-management experiment tooling under `benchmarks/cudagraphs/`; place
any isolated kernel benchmarks under `benchmarks/kernels/`. Store commands,
version manifests, seeds, raw measurements, and plots with each experiment report.
The final proposal should show memory versus graph count, memory versus maximum
tokens, per-shape latency, and a breakdown explaining the saved bytes.

## Next deliverables and review requirements

GPU measurements are deferred until the shared GPU resource is available.
The next deliverables can be prepared without GPU access:

1. An ownership map covering inputs, outputs, temporary tensors, backend workspaces,
   graph pools, and graph executable lifetimes in both runners.
2. A measurement harness and result schema identifying the runner, full descriptor,
   graph count, allocator/backend, measurement phase, counters, and timings.
3. A planner interface and synthetic lifetime cases for CPU-side layout checks.

Once GPU access is available, use the repository's `uv` environment workflow,
reproduce the unmodified capture path, and complete the memory-growth report before
selecting a production allocation mechanism. No GPU or model choice limits the
design; each experiment records the resources it actually uses.

Before any upstream PR, perform the required duplicate-work checks and
`pr-checklist`, and include human review, tests, model evaluation, and AI-assistance
disclosure as required by `AGENTS.md`.
