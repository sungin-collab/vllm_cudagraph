# CUDA graph union memory: implementation status

Date: 2026-10-16. Source baseline: `3e99205337`.

The [specification](cudagraph_union_memory_spec.md) now has CPU prerequisites:
immutable trace and descriptor contracts, a deterministic planner, an independent
layout verifier, backend capability assessment, staged family capture/retirement,
a saved-trace simulator, strict reporting records, and paired result summaries.
The production allocator and serving configuration are unchanged. This is not a
completed Stage 3 implementation or evidence of GPU savings.

## Ownership inventory

| Path | Existing storage and owner | Integration obligation |
| --- | --- | --- |
| V1 `CUDAGraphWrapper` | Graph entries own executable handles; outputs may be weak references; input storage is caller-owned | Stage every member before publishing, attach arena ownership, retain inputs externally, preserve offloader stream ordering |
| V1 `GPUModelRunner.capture_model` | Dispatcher supplies the capture set; each descriptor warms and captures before the next; workspace is locked after the loop | Discover every shape first, freeze any planned backing before its first final capture, restore state after discovery, scope cleanup to the family |
| V2 `CudaGraphManager` | `graphs` owns executables; `pool` is shared; `release_graphs` clears graph handles | Future adapter must preserve the full native descriptor and drain before retirement |
| V2 `ModelCudaGraphManager` | `hidden_states`, auxiliary hidden states and intermediate tensors are persistent output buffers copied into during capture | Keep outputs external and account for their storage and copies; prepare fresh attention metadata for final capture |
| Both runners' input/attention/KV buffers | Runner, model state, block tables and backend owners retain addresses across invocations | Record storage generations and preserve normal input/consumer ordering; exclude from overlay |
| `WorkspaceManager.get_simultaneous` | One scratch backing per `(ubatch, lane)`; aligned views; growth replaces backing before the final workspace lock | Reuse this owner for any future workspace binding; finalize capacity before capture and never unlock while a plan is live |
| `WorkspaceManager.get_persistent_resource` | Keyed resources retained per `(ubatch, lane)` | Exclude from scratch planning even if created during warmup |
| `capture_pool` | Existing allocator routing selects normal or sleep-aware pools and updates NCCL pool selection | Preserve routing on cleanup; sleep and collectives are initially ineligible |
| Operator/library scratch | ATen, generated kernels, attention libraries or their allocators | No redirection exists in this implementation; obtain explicit allocation/access contracts before binding |
| Graph executables/driver allocations | Runtime and graph handles | Count separately; an arena cannot remove them |

This inventory identifies boundaries, not allocation liveness evidence. No
allocation site has yet been certified for production overlay. Existing workspace
sharing already reuses scratch, so a new byte arena alone does not demonstrate
an improvement over the baseline.

## Implemented contracts and limits

`cudagraph_memory.py` records V1 and V2 descriptors without collapsing them to
token count. Portable fingerprints include configuration/layout/backend/software
and device capability identity, trace evidence, and planner/backend versions.
Runtime configuration, storage, arena and execution-lane generations are checked
before launch. Missing generation changes cannot be inferred from raw pointers:
the future adapter must update generations whenever storage or configuration
changes.

`cudagraph_memory_planner.py` normalizes direct contiguous aliases and all their
accesses, validates acyclic GPU ordering, proves byte-range initialization, and
uses conservative whole-root lifetimes. Greedy placement orders roots by reserved
size and stable ID. Different members overlay one arena; its capacity covers all
members, including a middle-sized descriptor that requires the most memory.
A separate verification pass reconstructs evidence and checks every conflicting
pair, bounds, alignment, exclusions, and fingerprint. Incomplete alias/access
contracts reject planning rather than guessing timestamps. More general strides
and alias chains are deliberately unsupported.

`CaptureTransaction` accepts a backend protocol rather than a tensor allocator.
It stages all graphs, validates the entire family, and only then returns a
published lookup. Recoverable rejections drain, destroy handles/views, and release
storage before invoking the baseline callback once. Device faults propagate.
Failed draining retains ownership on the transaction/family for explicit cleanup;
there is no destructor that optimistically frees GPU resources. Backend reserve
must clean partial reservation failures; backend destroy/release must be
idempotent. Backend capture must keep partial handles on its owner, validate all
binding requests, and restore capture flags/routing on every exit. These are
backend obligations tested with a fake implementation, not CUDA guarantees.

A future runner must serialize capture/replay/retirement on the declared lane,
retain transaction ownership on cleanup failure, restore mutated model state,
and prevent external consumers from accessing scratch. The CPU layer cannot
verify hidden kernel accesses or create these ordering guarantees itself.

The [tools README](../../benchmarks/cudagraphs/README.md) documents simulation,
reporting inputs and CPU validation. The reporting schema is a summary contract;
the production manifest/event collector is still pending. Reports cannot satisfy
acceptance without that collector and its raw evidence.

## Validation and blocked stages

The local environment has no usable NVIDIA driver (`nvidia-smi` fails) and no
PyTorch installation. Tests use the repository `.venv/bin/python`, with CPU test
and lint dependencies installed through `uv`. No GPU binding, baseline memory
attribution, CUDA correctness, sanitizer, real-weight generation or model
evaluation has run. No memory or latency improvement is claimed.

CPU tests cover ordered/unordered lifetimes, late aliases, retained outputs,
partial initialization, read-modify-write, empty layouts, overflow/alignment,
nonmonotonic descriptor capacity, deterministic layouts, randomized intervals,
corrupt-plan rejection, eligibility, failure at every capture position, stale
generations, repeated retirement, blocked cleanup, schema integrity, measurement
pairing, statistics, and simulator execution. The suite is wired into the CPU V1
Buildkite job.

Local validation completed:

- `.venv/bin/python -m pytest --noconftest tests/v1/cudagraph/test_cudagraph_memory_planner.py -q`: 63 passed.
- All configured pre-commit checks on the changed files passed, including mypy
  for Python 3.11 and Buildkite test tethering.
- The report CLI decoded and summarized five synthetic baseline/union pairs;
  these are tool checks, not GPU experiment results.

The next stage requires a GPU environment and an all-descriptor production
capture collector with fixed actual KV capacity. Measure attribution and freeze
opportunity/noise targets before selecting a mechanism. Then implement and
validate a real backend on minimal CUDA captures against the current shared pool.
Only positive Stage 1/2 evidence permits adding the serving opt-in, production V1
adapter, budget accounting, model evaluation, sanitizer runs and performance
claims. The specification's Stage 1/2 exit decisions remain open.
