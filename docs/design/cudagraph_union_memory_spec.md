# CUDA graph union memory: software specification

Status: proposed specification; implementation and GPU validation pending.

Date: 2026-10-16.

Source plan: [research and implementation plan](cudagraph_union_memory_plan.md).

Reviewed checkout: `10164931c40fae07b99db3e6ed769f0ee05845ed`; the plan's
code baseline is `9d164f0a44`. The intervening commit adds the plan.

This document specifies the measurement tools, memory planner, and opt-in runtime
integration needed to evaluate and implement the plan. **MUST** identifies a
correctness or acceptance requirement; **SHOULD** identifies a preferred design
that needs a documented reason to change. Proposed configuration fields,
interfaces, and new file paths below do not exist yet. Performance figures are
targets, not results.

## 1. Objective and scope

Reduce the GPU memory attributable to an unchanged collection of captured CUDA
graphs by assigning temporary storage from mutually exclusive executions to a
shared arena. Each descriptor keeps its own graph executables, kernels, launch
parameters, and dispatch behavior. Total graph memory can still grow with graph
count because persistent buffers and executable/driver allocations remain.

The first runtime implementation targets `GPUModelRunner` in
[`gpu_model_runner.py`](../../vllm/v1/worker/gpu_model_runner.py), called **runner
V1** here, with `FULL_DECODE_ONLY`. This is a specification choice that narrows
integration work; Stage 1 may change the selected runner through a recorded design
decision. The architecture and measurement schema MUST accommodate **runner V2**
in [`gpu/`](../../vllm/v1/worker/gpu/cudagraph_utils.py) without reducing its richer
descriptor to V1 fields.

Initial eligibility requires CUDA, one device, sequential dense-decoder execution,
one validated attention backend/dtype combination, and complete ownership and
ordering evidence for every allocation selected for overlay. It excludes
piecewise/mixed capture, distributed collectives, overlapping microbatches,
LoRA, speculative decoding, MoE/hybrid state, multimodal encoders, sleep/wake,
weight offloading, and runtime model/workspace reconfiguration. These features
continue through the existing allocation path until individually validated.
Internal library streams also require explicit dependency evidence.

Eligibility MUST depend on capabilities and allocation contracts. GPU/model names
may identify experiments, but MUST NOT select offsets, arena sizes, or correctness
exceptions. Other execution backends require their own allocation and ordering
implementation and validation.

The project does not change batch coverage, replay the maximum-token graph for
smaller batches, reduce graph count, change model arithmetic, compress KV storage,
or promise that arbitrary PyTorch operator temporaries can use a supplied tensor.
Online arena growth and replay-time virtual-memory remapping are outside the first
implementation.

## 2. Existing behavior and ownership boundaries

The baseline is the current shared-pool implementation, including its capture
order and workspace reuse. A separate-pool implementation is only a diagnostic
control.

| Component | Current behavior | Required integration boundary |
| --- | --- | --- |
| [`CUDAGraphWrapper`](../../vllm/compilation/cuda_graph.py) | Uses the global graph pool; stores graph entries and weak output references; input storage belongs to callers | Attach strong arena ownership to captured entries; stage entries before publication; preserve caller-owned inputs and offloader ordering |
| [`CudagraphDispatcher`](../../vllm/v1/cudagraph_dispatcher.py) | Determines descriptors and padding; captures descending token sizes within a mode | Enumerate the exact descriptor set; preserve dispatch and padding decisions |
| [Runner V1 capture loop](../../vllm/v1/worker/gpu_model_runner.py) | `capture_model`, `_capture_cudagraphs`, and `_warmup_and_capture` coordinate warmup/capture; workspace locks afterward | Coordinate discovery, planning, final capture, publication, and teardown at family scope |
| [`CudaGraphManager` / `ModelCudaGraphManager`](../../vllm/v1/worker/gpu/cudagraph_utils.py) | V2 shares a pool, owns graph dictionaries and output buffers, and prepares fresh attention state for capture | Future adapter around `capture`, `run_fullgraph`, and `release_graphs`; preserve fresh metadata and full descriptor identity |
| [`WorkspaceManager`](../../vllm/v1/worker/workspace.py) | Shares scratch per `(ubatch, lane)`; supplies aligned views and keyed persistent resources; locks against growth | Reuse existing workspace plumbing where possible; distinguish scratch from persistent resources; finalize capacity before capturing pointers |
| [`capture_pool`](../../vllm/compilation/cudagraph_pool.py) | Routes captures into sleep-aware allocation and updates the NCCL graph-pool selection | Preserve allocator scope and cleanup; reject sleep/collective combinations until supported |
| Existing graph-memory profilers | V1 samples up to two descriptors per mode; V2 samples FULL and captures other categories more completely | Keep startup estimates distinct from exhaustive experimental measurements; never promote throwaway profiling graphs into production |
| [`GPUWorker`](../../vllm/v1/worker/gpu_worker.py) | Memory sizing can reserve estimated graph costs before allocating KV | Include arena planning/setup in budget accounting; compare actual KV capacity, not only requested utilization |

Both runners already lock the workspace after capture. The union path MUST prevent
growth once the first final graph captures an arena pointer, even if the existing
global workspace lock occurs later. A weak tensor reference does not establish
storage ownership or GPU completion.

| Storage | Owner and minimum lifetime | Union policy |
| --- | --- | --- |
| Weights, KV, mutable model state, RNG/counters | Existing model/runner owners across steps | External to the arena; preserve updates and addresses |
| Input, position, slot mapping, and attention metadata buffers | Runner/backend through all graph users | External unless an explicit shorter lifetime is proven; refresh contents using existing ordering |
| Outputs and live views | Runner/caller through downstream consumers | External in the first implementation; copying from scratch requires a measured, ordered copy |
| Explicit backend scratch and intermediate tensors | Arena owner through captured graph lifetime; contents through their GPU consumers | Eligible only with known sizes, aliases, initialization, and dependencies |
| Keyed persistent workspace resources | `WorkspaceManager` or backend resource owner | External even when allocated during warmup |
| Uncontrolled library/operator workspaces | Their allocator/library | Retain on the existing path and count their bytes; reject the family if interactions cannot be isolated |
| Graph executable and driver storage | Graph object/runtime | Separate memory category; no claim that arena packing removes it |

Independent, nonconcurrent PyTorch graphs can share a pool, with care for outputs
that another replay can overwrite. Thus varying sequential batch order is not
itself a new capability. See [PyTorch pool-sharing semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#sharing-memory-across-captures).

## 3. Requirements

| ID | Requirement |
| --- | --- |
| R1 | Preserve the full resolved capture set, graph pieces, dispatch decisions, padding, model results, and mutable-state updates. |
| R2 | Every captured address MUST remain valid and unchanged until its last graph and consumer release it. |
| R3 | Distinct allocations that can be live simultaneously MUST occupy disjoint physical storage. Unknown ordering counts as possible overlap. |
| R4 | Every scratch byte read MUST be initialized during that invocation by a preceding GPU write, or be explicitly classified as persistent storage. |
| R5 | Reuse MUST wait for execution and all relevant consumers, including other streams and asynchronous copies. Host return from replay is insufficient. |
| R6 | Plan every member before final capture; independently validate the plan and final allocation binding before publishing the family. |
| R7 | Bind plans to configuration, compilation, allocation contracts, and storage generations. A mismatch MUST prevent replay of a stale graph. |
| R8 | Unsupported configurations and safely recoverable pre-publication failures MUST use the existing path with a recorded reason. Device faults MUST propagate. |
| R9 | Separate persistent buffers, reusable workspace including slack, and external graph allocations; count every physical allocation once. |
| R10 | Primary measurements MUST use the modified production capture manager, identical coverage, fixed KV capacity, and fresh processes. |
| R11 | The planner MUST be deterministic and testable on CPU; allocation and completion operations MUST be behind a backend interface. |
| R12 | The default path MUST remain unchanged. Enabled replay MUST avoid tracing, layout construction, allocation, and added device-wide synchronization. |

## 4. Architecture and mechanism selection

The design has four logical components. They need not become four packages.

1. **Runner adapter:** enumerates descriptors, discovers storage contracts, stages
   captures, and observes dispatch and consumer completion.
2. **CPU planner and verifier:** normalize aliases and lifetimes, assign offsets,
   and verify a layout independently of the placement algorithm.
3. **Storage backend:** supplies stable backing storage, binds eligible allocation
   sites, validates capture-time requests, and releases storage after retirement.
4. **Experiment harness:** launches controlled runs, collects raw measurements,
   validates comparisons, and produces evidence reports.

Proposed source locations are
`vllm/compilation/cudagraph_memory.py` for contracts/lifecycle and
`vllm/compilation/cudagraph_memory_planner.py` for the CPU planner. CUDA-specific
operations belong behind the storage backend and the repository's platform
abstractions. Runner adapters stay near their existing capture managers. Reuse
`WorkspaceManager` for allocation sites it already owns; avoid introducing a
second independent owner for the same storage.

The production allocation mechanism remains a Stage 1/2 decision. The mechanism
report MUST identify allocation sites, controllable bytes, unsupported bytes,
ownership, APIs used, and whether a PyTorch/compiler change is necessary.

| Candidate | Evidence required before selection |
| --- | --- |
| Retention/allocation-order repair | Trace identifies avoidable retention or fragmentation and a minimal change reduces it beyond the existing baseline |
| Explicit workspace plumbing | Identified operators accept supplied storage without changing semantics; all affected requests fit the verified plan |
| Capture-aware allocator integration | The pinned allocator supports interception at the required granularity, graph ownership, safe frees, aliases, and stream dependencies |
| Compiler memory planning | Generated code can bind planned storage and preserve alias and graph-break semantics; compilation changes are versioned |
| Virtual memory or graph memory nodes | Simpler choices leave material recoverable bytes; API compatibility, physical accounting, and mapping/synchronization costs are measured |

A preallocated byte tensor does not redirect unmodified ATen, Triton, cuBLAS,
attention, or other library allocations. An allocator that only receives segment
requests cannot be assumed to control individual tensor lifetimes. Stage 2 MUST
prove the selected binding mechanism with actual captures before Stage 3 promises
a vLLM-only implementation.

Compare the selected design with [PyTorch CUDAGraph Trees](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html),
which account for live allocations when managing a shared graph pool. Record why
the existing integration does not already provide the needed saving. CUDA graph
memory nodes are a separate mechanism with GPU-ordered allocation/free lifetimes
and potential physical-memory reuse; they are not the ordinary PyTorch graph pool.
See [NVIDIA graph memory nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#graph-memory-nodes).

## 5. Identity and input contracts

### 5.1 Execution family

An `ExecutionFamily` contains complete executions that may share an arena. It
includes device/context, runner, runtime mode, backend versions and capabilities,
dtype/quantization, compiled-program identity, persistent-storage generation,
ownership policy, and concurrency schedule. The first adapter creates one family
for its eligible FULL decode descriptors and guarantees one ordered execution
lane. If that guarantee is unavailable, it uses the existing path.

Each member keeps its native descriptor:

| Runner | Required descriptor fields at this revision |
| --- | --- |
| V1 | Runtime mode plus `BatchDescriptor(num_tokens, num_reqs, uniform, has_lora, num_active_loras)` |
| V2 | `BatchExecutionDescriptor(cg_mode, num_tokens, num_reqs, uniform_token_count, max_query_len, num_active_loras, num_ubatches)` |

Serialize the runner and descriptor schema version with these fields. Null values
retain their native meaning. Never key a plan by token count alone. Future
descriptor additions MUST invalidate an older incompatible schema.

The family fingerprint MUST cover the resolved descriptor set, mode/runner,
model/config revision, tensor/KV layouts, backend and allocator settings, dtype,
compilation flags and artifact digest, relevant software/device capabilities,
workspace lane/concurrency policy, trace digest, planner version, and storage
backend version. Record runtime storage identities/generations separately from
the portable hash. Raw device pointers are process-local diagnostics, not reusable
cache keys. Persisted plans are diagnostic artifacts in the first release and
MUST NOT bypass discovery or capture validation in another process.

### 5.2 Allocation and lifetime records

Discovery produces a versioned `FamilyTrace` with one execution DAG per member.
The DAG represents GPU ordering, including stream events and downstream consumers.
A single total sequence is sufficient only when the adapter proves one ordered
stream covers all accesses. CPU allocation/free timestamps are diagnostic data,
not GPU liveness evidence.

| Record | Required content |
| --- | --- |
| `AllocationSpec` | Stable allocation-site ID and occurrence, member ID, backing-storage/alias-group ID, requested and rounded bytes, alignment, device, owner, storage class, controllable binding ID |
| `AliasSpec` | Root storage ID, byte offset, accessible span, dtype/shape/stride constraints; a view extends the root's lifetime |
| `AccessSpec` | Allocation/byte range, operation ID, read/write/read-modify-write kind, initialization evidence, dependency nodes |
| `DependencySpec` | Ordered source/target operation IDs and evidence: stream order, graph edge, event wait, or explicit owner contract |
| `EscapeSpec` | External owner, consumers, completion condition, retention policy, and reason for exclusion |
| `TraceCoverage` | Observed/declared allocation sites, missing information, discovery method, and which contracts were checked during final capture |

Use bytes for all storage arithmetic. Allocation IDs MUST survive address changes
between discovery and final capture; raw addresses and request order alone are
insufficient identifiers unless a backend proves a stable contract. Store
allocation occurrences separately when an address is reused during discovery.

Alias groups MUST be canonicalized before placement. Unsupported strides or
unknown aliases exclude the storage from overlay; uncertainty that could alias an
eligible region rejects the family. Missing lifetime or ownership information
never becomes a guessed end-of-life timestamp. Uninitialized padding read by a
kernel is part of the initialization contract, even when logically masked output
elements are ignored.

### 5.3 Planning and backend interfaces

The following signatures describe proposed contracts, not an existing API:

```text
assess(configuration, runner_capabilities, backend_capabilities)
    -> Eligibility(supported, reason_codes)
discover(family) -> FamilyTrace
plan(trace, limits) -> MemoryPlan | PlanRejection
verify(trace, plan) -> VerificationReport

backend.reserve(plan) -> ArenaOwner
backend.bind_capture(owner, member_plan) -> CaptureBindingScope
backend.validate_capture(scope) -> CaptureValidation
backend.retire(owner, completion) -> RetirementHandle

capture_family(family, plan, owner) -> StagedGraphFamily
publish(staged_family) -> PublishedGraphFamily
drain(published_family) -> Completion
```

`MemoryPlan` contains the fingerprint, one placement per eligible root storage and
member, excluded allocations/reasons, arena size/alignment, external byte counts,
dependency proof references, and verifier result. `ArenaOwner` retains backing
storage and backend resources. Each published graph family holds a strong
reference to it; tensor views follow the chosen backend's storage ownership rules.

`CaptureBindingScope` MUST satisfy each planned request with the specified address,
check size/alignment/alias/access contracts, and preserve ordinary allocation for
declared external sites. A changed, missing, or unexpected managed request rejects
the capture. It MUST NOT silently allocate a replacement and publish a graph under
the old plan. Any backend unable to validate its binding declares itself
unsupported. Completeness of kernel access information remains a backend contract;
the CPU verifier cannot infer it from an allocator trace.

## 6. Layout algorithm and invariants

The first planner SHOULD use deterministic greedy placement. An optimal packing
solver is unnecessary for the initial experiment.

1. Validate integer bounds, sizes, alignments, DAG acyclicity, alias spans, and
   ownership. Normalize root storage and union the lifetimes of all its views.
2. Classify persistent/escaping/uncontrolled allocations as external. Reject
   unsafe interactions and missing evidence for any remaining allocation.
3. Within each member, construct a conflict relation. Two root lifetimes conflict
   unless all accesses to one are proven to finish before the other begins,
   including writes that initialize reused bytes. Unordered accesses conflict.
4. Place roots in descending rounded size, breaking ties by stable ID. Choose the
   lowest aligned offset that does not overlap a conflicting placement. Allow
   different members to use different offsets over the same arena base.
5. Compute member high-water marks and the family capacity. Run a separate
   pairwise verifier from the trace and placements, not from the planner's cached
   conflict decisions. Reject violations before reserving device memory.

For a strictly exclusive family with all escapes external:

```text
member_bytes[d] = max(offset[d, a] + reserved_span[d, a])
arena_bytes     = backend_round_up(max_d(member_bytes[d]), arena_alignment)
pointer[d, a]   = fixed_arena_base + offset[d, a]
```

Empty layouts require zero arena capacity and no dummy dereferenceable pointer.
The verifier checks nonnegative offsets, overflow, address alignment including
alias offsets, bounds, exact alias relationships, and nonoverlap for every
conflicting pair. It also checks that each planned byte read has a dominating
initialization or an external persistent owner. Partial initialization must cover
the actual read ranges. A conservative whole-allocation lifetime is acceptable.

Sequential family members can overlay their complete scratch layouts because
scratch values do not survive an invocation. Persistent state may carry between
steps through external buffers. Capacity MUST consider all descriptors; the
maximum-token member need not require the most memory. Separate concurrently
executing families require disjoint arenas unless a later design proves an
inter-family schedule. Cross-family overlay is excluded initially.

Report peak live scratch bytes as a lower bound only where the dependency model
supports that calculation; otherwise report a clearly identified bound. Report
alignment and packing slack separately. Maximum standalone reserved workspace is
an empirical allocator reference, not a mathematical lower bound or optimality
claim. Arena caps are resource limits: exceeding one rejects the plan without
truncating storage or dropping descriptors.

## 7. Capture, replay, and retirement

### 7.1 Startup transaction

```text
DISABLED -> existing capture
ASSESS -> DISCOVER -> PLAN -> VERIFY -> RESERVE -> CAPTURE -> VALIDATE -> READY
                  recoverable rejection -> CLEANUP -> existing capture
READY -> DRAINING -> RETIRED
```

Before `READY`, the family is private to initialization. The coordinator MUST:

1. Resolve descriptors and compilation, initialize persistent storage, and warm
   every relevant shape and backend path. Complete lazy initialization separately
   from final capture. Record its cost in startup measurements.
2. Discover contracts with representative KV/input conditions. A sampled execution
   alone is not proof for a data-dependent allocation path; require a static bound
   or reject that path. The final plan uses actual production storage/layouts,
   never the startup profiler's minimal throwaway KV bindings.
3. Plan and verify every member; reserve final storage once; prevent relocation
   before any final graph captures its pointers. Restore mutable state affected
   by warmup/discovery so final initialization preserves baseline behavior.
4. Capture into a private graph-entry collection with the existing descriptor and
   mode order. Validate the final allocation contract for every member and retain
   all required owners. Capture scratch and outputs may need initialized setup
   values even before the first replay.
5. Validate every staged graph with controlled inputs; include first-replay lazy
   costs. Restore mutated KV/RNG/counters after validation. Publish the entire
   family only after all checks succeed, then lock its storage for its lifetime.

Existing capture loops publish entries incrementally. The union adapter MUST add
a family-level staging/publication boundary; inserting a partially initialized
entry into the production lookup is insufficient. In the first implementation,
one failed member causes the whole decoder family to use existing capture. Partial
union/default membership within that family is deferred.

Cleanup MUST restore capture-enable flags, allocator routing, wrapper/manager
references, workspace state, and counters. Release staged graphs before their
arena, and only after outstanding GPU work and views are safe. Scope cleanup to
the transaction; a failed family MUST NOT clear unrelated live graph managers.

### 7.2 Replay and output consumption

The normal dispatcher selects the same descriptor as before. The published entry
selects its immutable plan/arena generation without tracing or planning. Input
updates precede launch using the existing GPU ordering.

The first release relies on the runner's proven single execution lane. Arena
reuse follows all accesses to its scratch. Same-stream ordering can establish
this without a host synchronization. Any permitted side-stream consumer requires
an event dependency before the next writer. Unknown consumers make overlay
ineligible. Events, lookup tables, and ownership records are prepared at startup;
do not allocate them on every replay.

Outputs remain outside the arena by default. If a prototype copies scratch-backed
results into owned outputs, the copy must finish reading scratch before reuse;
its storage and latency count in results. Preserve the baseline output-lifetime
contract, including retained views and asynchronous D2H copies. Holding a Python
object, dropping a reference, or returning from `replay()` does not establish GPU
completion.

No replay-time resizing, allocator rewinding, pointer patching, or virtual-address
remapping is allowed in the first implementation. Unexpected descriptor or
generation mismatches MUST be detected before launching affected work. Requests
that the existing dispatcher routes to eager execution keep that behavior and
MUST NOT accidentally use union scratch outside its ownership protocol.

### 7.3 Invalidation and failure behavior

Capture-set changes, recompilation, backend/dtype changes, KV or workspace
relocation, and execution-schedule changes invalidate the family. Stop new uses,
drain GPU work and consumers, destroy graph handles and dependent views, then
release the arena. Rebuild from discovery or enter existing capture at a safe
initialization boundary. The first release does not perform online rebuilds.

| Condition | Required action |
| --- | --- |
| Unsupported feature/API or incomplete ownership evidence | Select existing capture before publication; log a stable reason code |
| Invalid layout, changed capture allocation, or recoverable reservation failure | Dispose of the staged family; retry the existing path once if the context is healthy |
| Existing-path capture also fails | Propagate the original/combined diagnostic; never claim successful fallback |
| Stale configuration/storage detected before launch | Block stale replay and require safe reinitialization; preserve ordinary eager dispatch only where already supported |
| CUDA illegal access, assertion, or other device-context failure | Propagate and use the existing worker failure/restart behavior; do not replay a fallback graph in a possibly corrupted context |
| Teardown with live consumers | Drain or retain ownership until consumers release; never free optimistically |

Reason codes SHOULD include `unsupported_runner`, `unsupported_feature`,
`unsupported_allocator`, `unknown_lifetime`, `unknown_alias`,
`uncontrolled_workspace`, `layout_conflict`, `capture_contract_mismatch`,
`arena_limit`, and `stale_generation`. Include the affected family/member and site
where known. A union benchmark that falls back is a failed/unsupported union run,
not a successful optimized measurement.

## 8. Configuration and observability

Stage 3 proposes one experimental field on `CompilationConfig`:

```text
cudagraph_memory_mode: "default" | "union" = "default"
```

`default` executes the current path without discovery or union metadata overhead.
`union` requests capability assessment and the transaction above. Existing config
validation handles unknown values. Graph mode `NONE` keeps eager behavior and
reports union as inactive. Diagnostics record requested and effective memory mode
and every fallback reason. No HTTP request/response schema changes are required.

Discovery, strict failure-on-fallback, output paths, and experimental controls
belong to the benchmark harness rather than new serving flags. An optional arena
limit can remain an internal/harness planning limit initially. If storage binding
changes generated code, its configuration MUST participate in compilation cache
identity; otherwise document why the compiler cache can be shared while keeping
the separate memory-plan fingerprint.

Startup diagnostics MUST report family/plan IDs, descriptor and executable counts,
arena reserved bytes, external persistent/workspace bytes, excluded sites,
discovery/planning/capture time, and effective backend. Per-replay allocation
traces are debug-only. Ordinary telemetry SHOULD use aggregate counts; avoid a
new high-cardinality metric for each allocation site.

## 9. Measurement harness

### 9.1 Execution protocol

Place proposed tools under `benchmarks/cudagraphs/`: `run.py`, runner adapters,
versioned result schemas, and `report.py`. The harness MUST call the selected
runner's real capture/dispatch path. The existing
[`vllm bench forward` capture](../../vllm/benchmarks/forward/capture.py) bypasses
that manager, and its [setup](../../vllm/benchmarks/forward/main.py) disables
compilation; it is only a complementary per-shape timing tool for this project.

Run each control and replicate in a fresh process:

| Variant | Purpose |
| --- | --- |
| `shared_baseline` | Current sharing, capture order, and complete descriptor set; primary baseline |
| `max_tokens_only` | Maximum-token descriptor capacity reference; explicitly different coverage |
| `standalone` | One full descriptor per process, including all its pieces; obtain maximum standalone workspace |
| `separate_pools` | Separate pool per complete descriptor; diagnostic value of existing sharing |
| `union` | Same coverage and dispatch as `shared_baseline`; reject the comparison if any required member falls back |

First use a feasible explicit set such as `1, 2, 4, 8, 16, 32, 64, 128`, then the
resolved production set. Sweep graph count at a fixed maximum and maximum size at
a fixed capture policy. Include kernel-selection boundaries, more than one KV
length, and ascending/descending/repeated/random/alternating extreme sequences.
Record the actual padded descriptor selected for unaligned requests. References
with reduced coverage MUST NOT be used as primary performance baselines.

Hold model revision/weights, KV capacity, maximum sequence length, compile/backend
settings, inputs, mutable-state initialization, allocator policy, warmup, and
output-retention policy constant. Keep `empty_cache()` and garbage-collection
schedules identical across variants and record them; additional trimming between
captures changes the experiment. Prefer fixed `kv_cache_memory_bytes` for primary
runs and verify actual per-group KV block counts and bytes. In this checkout that
setting skips the startup graph estimator in `GPUWorker`; record whether the
estimator ran rather than assuming it did. A separate fixed-total-budget study
may enable automatic KV sizing to measure additional capacity.

Use two passes: instrumented diagnostics and uninstrumented capture/replay. Sample
startup process/device memory at a recorded interval; label sampled peaks as
sampled lower bounds on the true peak. Allocator peak counters have their own
coverage and reset boundaries. Synchronize memory snapshots at defined phase
boundaries. Do not put measurement synchronizations into the timed replay loop or
use diagnostic profiler timings as final latency numbers.

Measure setup, each warmup/capture, every member's first replay, steady state,
shape-switch stress, and teardown. Replay all graphs before steady residency.
Measure GPU execution with events, CPU enqueue/management time with a monotonic
clock, and completed-step wall time separately; do not add overlapping host/GPU
durations. Record harness overhead and time boundaries. Run without competing GPU
jobs, recording any observed contention or clock/power changes.

### 9.2 Result schema and artifacts

All records MUST have `schema_version`, `run_id`, `pair_id`, `variant`, `pass`, and
an explicit success/failure/unsupported status. Byte fields are integer bytes;
time fields identify their units. Unknown or unsupported counters are `null` with
a reason, never zero. Preserve raw signed memory deltas even when noise makes
them negative.

| Artifact | Minimum required fields |
| --- | --- |
| `manifest.json` | Exact command/environment, code revision and dirty-diff digest, Python/PyTorch/CUDA/driver/backend versions, GPU identity/capabilities, model/weights revision, dtype, allocator settings, trimming/GC schedules, runner, compilation settings, capture policy and resolved descriptors, seeds, actual KV layout/capacity, instrumentation and peak-sampling interval |
| `events.jsonl` | Monotonic timestamp and clock domain, process/rank/device identity, phase/sequence/replay-sample IDs, runner/family/plan/member IDs, native descriptor, request and padded tokens, graph object/piece IDs and counts, requested/effective mode, allocator counters, pool IDs/segments, arena bytes, process/device counters, raw latency values and measurement method, consumer-completion boundary, fallback/error reason |
| `allocations.jsonl` | Versioned allocation/alias/access/dependency/escape records from Section 5, evidence source, and coverage gaps |
| `plan.json` | Fingerprint, offsets, sizes/alignment, external allocations, capacity/slack, verifier report, selected mechanism/version |
| `summary.json` | Baseline pairing, coverage validation, memory-category totals and uncertainty, latency distributions and confidence intervals, correctness/eval status, acceptance decisions |
| Report and plots | Commands, assumptions, unexplained bytes, memory versus graph count and maximum tokens, per-shape latency, saved-byte breakdown, startup cost, tested compatibility matrix |

Count descriptors and actual graph executables separately, especially for
piecewise, encoder, and speculator captures. Include graph-owned allocations made
before the capture loop; moving work into setup MUST NOT count as a saving.
Artifact generation MUST reject mixed schema versions or incompatible manifests
unless an explicit conversion/comparison rule exists. Result tools run on CPU and
produce reports from saved data without loading the model. Preserve every raw
timing sample with its warmup/measurement classification, and record the mapping
between sampler and worker clocks so startup samples can be joined to phases.

### 9.3 Memory accounting

Use disjoint ownership categories:

```text
process GPU memory = model/KV/runtime baseline
                   + graph-attributable persistent buffers
                   + reusable graph workspace, including allocator slack
                   + graph executable/driver and other external allocations
```

Reserved allocator bytes include allocated bytes; never add them. Arena storage
already included in allocator counters is not added a second time. Count physical
backing once even when multiple virtual addresses or views reference it. Annotate
each counter's allocator/device/process scope and backend validity.

When segments mix persistent and temporary allocations, disclose the attribution
rule and uncertainty instead of assigning all reserved slack to scratch. Compare
snapshots and controlled process/device deltas to identify external memory. Any
unexplained residual stays unattributed; it is not automatically executable
overhead. A total graph-memory acceptance claim requires an explained breakdown
or uncertainty bounds that cannot change the decision.

`cudaDeviceGetGraphMemAttribute` measures CUDA graph-memory-node storage, not all
PyTorch graph pools or all executable overhead. Use it only for that mechanism.
See [NVIDIA physical memory accounting](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#physical-memory-footprint).

## 10. Validation specification

Before adding a test, record its module purpose, public I/O contract, failure
mode, and cheapest effective test level. Reuse nearby fixtures. Do not require a
GPU for arithmetic, eligibility, schema validation, or lifecycle tests that can
use a fake backend.

| Test group | Observable behavior and minimum cases | Preferred location |
| --- | --- | --- |
| Planner/verifier | Known ordered lifetimes reuse offsets; simultaneous/unordered lifetimes do not; aliases extend ownership; alignment/overflow/bounds are enforced; independently reject deliberately corrupt layouts | New `tests/v1/cudagraph/test_cudagraph_memory_planner.py` for the new pure CPU contract |
| Planner edge cases | Empty input, zero-size requests, oversized allocation, deterministic result, nonmonotonic shape workspace, incomplete dependency/alias/initialization evidence | Same planner suite |
| Family lifecycle | No partial publication; failure on each capture position cleans owners/flags; repeated teardown is safe; stale generation blocks launch; fallback once and report reason | Extend [`test_cudagraph_manager.py`](../../tests/v1/cudagraph/test_cudagraph_manager.py) where adapter-independent fixtures fit |
| Dispatch/config | Exact and padded descriptors unchanged; same tokens with different request/mode fields stay distinct; default/unsupported configurations retain existing behavior | Extend [`test_cudagraph_dispatch.py`](../../tests/v1/cudagraph/test_cudagraph_dispatch.py) and [`test_cudagraph_mode.py`](../../tests/v1/cudagraph/test_cudagraph_mode.py) |
| Existing workspace ownership | Scratch versus persistent resources, distinct lanes, no growth after final binding, live views survive permitted uses | Extend [`test_workspace.py`](../../tests/v1/worker/test_workspace.py) |
| GPU graph correctness | Alternating/random/repeated shapes; preserved outputs/live views; ordered asynchronous consumer; teardown and recoverable capture failure; no stale-address reuse | Extend [`test_full_cudagraph.py`](../../tests/compile/fullgraph/test_full_cudagraph.py) or [`test_basic_correctness.py`](../../tests/compile/fullgraph/test_basic_correctness.py) |
| Future piecewise support | Tensor lives through graph A, eager attention, and graph B; no reset between pieces | Extend [`test_multiple_graphs.py`](../../tests/compile/fullgraph/test_multiple_graphs.py) and [`test_breakable_cudagraph.py`](../../tests/v1/cudagraph/test_breakable_cudagraph.py) before enabling support |
| Harness/reporting | Correct units/nulls, no double counting, mismatched coverage rejected, fallback excluded from union success, reproducible paired summaries | Small CPU tests next to the new harness/schema, following repository test placement conventions |

Minimum synthetic planning examples MUST include: two ordered allocations that
can share one span; two unordered stream branches that cannot; a late view that
extends a root lifetime; and three descriptors whose middle token count needs
the largest workspace. Also test read-modify-write without per-replay
initialization, and an output retained beyond the next replay.

GPU validation compares union, existing graphs, and eager execution with identical
weights, inputs, and initial KV/RNG/counter state. Restore state before each
equivalent run; separately test complete stateful sequences. Compare tensor
outputs and updated KV/state with predeclared dtype-appropriate tolerances and
record maximum errors. Shape, layout, finite-value checks, and preservation of
retained results are mandatory. Numerical nondeterminism must be characterized
against the baseline before relaxing a tolerance.

Run real-weight generation and a relevant existing model evaluation from
`tests/evals/` before claiming model correctness. Record dataset/version, prompt
settings, seeds, metric, baseline variance, and accepted accuracy difference
before tuning. Run NVIDIA Compute Sanitizer on the minimal reproducer and
targeted GPU tests, recording tool/version/checker coverage; a clean sanitizer
run supplements the lifetime proof and model checks.

For leak/stress validation, the initial harness default SHOULD exercise at least
10,000 completed shape switches after all descriptors are warm, sampling every
1,000 steps. Require no unexplained growth above a predeclared noise bound and
successful resource retirement. These counts are proposed test parameters,
subject to recorded adjustment for workload duration, not claims of exhaustive
correctness.

## 11. Acceptance criteria and statistics

For the same descriptors, allocator settings, output policy, and warmup:

```text
W_ref   = max(standalone workspace reserved for each complete descriptor)
E_base  = max(0, shared_baseline workspace reserved - W_ref)
E_union = max(0, union workspace reserved - W_ref)
excess_reduction = (E_base - E_union) / E_base
graph_reduction  = (graph_memory_base - graph_memory_union) / graph_memory_base
```

Compute excess reduction only when `E_base` is positive and above measurement
uncertainty. Otherwise mark it not applicable and report absolute values. Use
paired configurations to propagate uncertainty in references and comparisons.

| Gate | Acceptance evidence |
| --- | --- |
| Correctness | R1–R8 hold; all required CPU/GPU/stateful/model checks pass; no unexplained sanitizer failures |
| Recoverable workspace | Provisional target: at least 50% reduction in `E_base`; aim within 10% of `W_ref` where layouts permit |
| Practical memory saving | Provisional target: at least 20% total graph-attributable reduction on two meaningful configurations; report absolute MiB and whole-process savings |
| Replay performance | Completed decode-step median regression below 2%, p95 below 5%, for matching coverage; report GPU execution and CPU enqueue time separately |
| Startup and residency | Report actual sampled startup peak and capture time; investigate capture-time growth above 20%; no unexplained long-run growth |
| Compatibility | Every supported combination has evidence; unsupported cases visibly use existing allocation |

Stage 1 MUST freeze an absolute MiB opportunity target, a graph-memory fraction,
measurement noise bounds, and the final acceptance thresholds before prototype
tuning. A smaller opportunity requires an explicit project decision, not a
changed baseline or an undefined percentage. Two meaningful configurations may
support initial value evidence on available hardware; broad portability claims
also require the cross-hardware/model matrix in Section 12.

The primary latency endpoint is completed decode-step wall time from runner input
preparation through its normal downstream output-consumption boundary. Record the
exact boundary for the selected runner and use it in both variants. The proposed
2%/5% gates apply to each preregistered per-descriptor workload and the declared
shape-switching mixture; a favorable mixture cannot hide a failing shape. GPU
execution and CPU enqueue distributions explain the result and identify costs
outside event timing. Freeze workload weights and any revised endpoint/gating
policy at Stage 1, before tuning. Use the runner's normal completion mechanism;
do not add a device-wide synchronization to every timed step.

Use at least five fresh-process baseline/union pairs per primary configuration,
alternating their run order. Use the same seeded dispatch sequence within each
pair. Record replay sample count and length; gather enough samples for a stable
p95. Report per-run distributions and 95% confidence intervals for paired
differences/ratios, using runs as independent units rather than treating every
correlated replay as an independent experiment. Increase repetitions when the
interval is inconclusive. The proposed acceptance rule is that the relevant
confidence bound meets the memory or latency threshold; otherwise label the
result inconclusive or failed, as appropriate.

## 12. Delivery stages and decisions

| Stage | Deliverables | Exit decision |
| --- | --- | --- |
| 0: baseline, 1–2 days | Ownership inventory, pinned environment/configuration manifest, reproducible current capture and replay | Baseline works and coverage/KV controls are verified |
| 1: attribution, 3–5 days | Harness/schema, all-descriptor traces, memory-category report, opportunity/noise analysis, frozen targets | Continue only for material avoidable workspace growth; otherwise document the result and stop/reframe arena work |
| 2: mechanism, 3–5 days | CPU planner/verifier, synthetic lifetime cases, minimal real CUDA captures, mechanism/API decision | Demonstrate correct binding and savings beyond shared pools; explicitly resolve any PyTorch/compiler dependency |
| 3: V1 integration, 5–8 days | Opt-in field, transaction, immutable ownership, fallback, diagnostics, tests and real-model evidence | All initial-path invariants and acceptance checks pass |
| 4: evaluation, 5–8 days | Repeated results, plots, serving evaluation where applicable, compatibility report | Human review accepts measured value and limitations; scope expansion gets its own evidence |

The four-to-six-week estimate depends on GPU access and positive intermediate
results. CPU deliverables can proceed now: the ownership inventory template,
harness/result schemas, planner contracts, simulator, and fake-backend lifecycle
tests. GPU mechanisms and performance claims remain gated by measurement.

After the first FULL-decode result, evaluate V2, then complete-execution piecewise
planning and mixed modes. Before enabling each deferred feature, document its
storage owners, concurrency, invalidation behavior, and tests. Broader evidence
SHOULD cover at least two GPU architectures, two model families including a
structurally different MoE/hybrid case, multiple attention backends, and full and
piecewise modes. Use the same planner technique with configuration-specific plans.

Fixed-total-budget evaluation reports whether saved memory buys additional KV
blocks or request capacity. Eventual `vllm bench serve` runs report throughput,
TTFT, and inter-token latency alongside workload details. A per-shape forward
microbenchmark cannot substitute for this or for manager memory measurements.

The implementation owner maintains tooling and raw evidence; the project owner
coordinates resources and stage decisions; a human reviewer checks conclusions
and every contributed line. Reports MUST retain negative findings and unresolved
attribution. Use the repository's `uv`/`.venv/bin/python` workflow, relevant tests
and lint hooks, and incremental compilation workflow if native code is added.
Before an upstream PR, complete duplicate-work checks and `pr-checklist`, human
review/testing, model-evaluation reporting, and AI-assistance disclosure required
by [`AGENTS.md`](../../AGENTS.md).

## 13. Open decisions and risk closure

| Decision/risk | Required resolution |
| --- | --- |
| Residual memory may be executable overhead rather than recoverable scratch | Stage 1 category attribution and controlled teardown experiments; stop arena work if opportunity is insufficient |
| Exact PyTorch/CUDA binding mechanism is unproven | Stage 2 API/version probe and capture reproducer; record upstream dependency before committing to production integration |
| Trace does not expose library streams, accesses, or aliases | Obtain explicit backend contracts or exclude storage/family; allocator timestamps alone cannot justify reuse |
| A later shape requires larger workspace | Discover every descriptor and verify final requests; never resize published storage |
| Outputs or state survive replay | Keep them external or implement ordered, costed ownership/copies; test downstream consumers and live views |
| Discovery and staging increase startup peak | Measure all setup/discovery/final capture bytes, release throwaway resources safely, enforce resource limits and recovery |
| Faster GPU timing hides CPU or serialization cost | Measure host management and completed-step latency with matching dispatch, plus serving evaluation |
| First configuration does not generalize | Capability checks and an explicit tested matrix; no GPU-name/model-name planning rules |
| Accuracy tolerances and final memory opportunity are not yet known | Freeze them from baseline evidence before tuning; keep acceptance pending until recorded |

Completion of this specification establishes implementable contracts and research
gates. It does not establish that an arena will improve the current allocator or
that the provisional performance targets are achievable.
