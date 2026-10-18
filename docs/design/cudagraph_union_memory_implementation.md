# CUDA graph union memory: implementation status

Updated: 2026-10-18. CPU foundation commit: `8c008ea589`.

The [specification](cudagraph_union_memory_spec.md) now has CPU prerequisites:
immutable trace and descriptor contracts, a deterministic planner, an independent
layout verifier, backend capability assessment, staged family capture/retirement,
a saved-trace simulator, strict reporting records, and paired result summaries.
Diagnostic production collection and an explicit CUDA binding probe are also
implemented, with initial GPU validation described below.
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
reporting inputs and CPU validation. The reporting schema is a summary contract. The diagnostic production collector
now writes manifests, raw events, allocator snapshots and records. Incomplete
physical ownership/access attribution remains explicitly unknown in reports.

## Completed diagnostic collection and explicit binding

`benchmarks/cudagraphs/collect.py` launches fresh processes for the shared baseline,
maximum-token reference, every standalone descriptor and separate-pool controls.
`production.py` observes the real V1 capture manager and dispatcher, retains full
native descriptors, checks captured membership and pool isolation, records fixed
KV allocation configuration, and saves allocator/device counters through startup,
capture, first replay, shape switching and retirement. NVML samples use the
worker's monotonic clock; foreign GPU processes invalidate the run. Success records
are published only after cleanup completes. Serving defaults are unchanged.

The collector currently supports one GPU and a dense text decoder using FULL
uniform decode. It runs diagnostic dummy steps, including padded request shapes,
and synchronizes at measurement boundaries. These timings are not completed
serving-step latency, and dummy replay is not a model-accuracy evaluation. Actual
KV bytes currently come from the runner's allocated KV configuration; independent
storage-level deduplication remains to be added. Allocator snapshots do not prove
kernel accesses or disjoint workspace ownership, so the four memory attribution
categories remain unknown. The diagnostic reporter preserves that uncertainty
instead of inferring savings from reserved bytes or reduced graph coverage.

`binding.py` supplies a narrow explicit CUDA backend for declared contiguous
allocations, aliases and serial operations. It verifies the plan, binds views into
an aligned arena, keeps outputs external, captures every member transactionally,
checks storage generations and stream identity, and drains before graph/storage
retirement. The caller must attest complete kernel accesses and stream usage;
request validation cannot discover hidden accesses or redirect arbitrary ATen
allocations. This is an executable Stage 2 mechanism probe, not the production
allocation adapter. Its per-replay identity hashing is diagnostic overhead.

## Validation

The installed environment now supports CUDA outside the filesystem sandbox:
PyTorch `2.14.0+cu126`, CUDA runtime 12.6, NVIDIA driver 550.144.03 and an A100
PCIe 40 GB. The editable vLLM installation and stable native extension load.
Transformers, Triton, FlashInfer, pytest and pre-commit are available. GPU 0 was
used for these checks; other devices were left to their existing workloads.

Completed checks:

- The targeted planner, binding, reporter and collector suite passed **101 tests**,
  with one CUDA test skipped inside the sandbox. The binding test also has a CUDA
  path and is wired into GPU CI. Running the binding suite on GPU 0 outside the
  sandbox passed all **9 tests**, including the real CUDA probe.
- The real CUDA binding probe completed **10,000 shape switches**, exact output
  checks, scratch poisoning and retirement. Its three synthetic members share a
  512-byte logical arena (767-byte aligned backing) with 1,408 bytes of separate
  external outputs. These sizes are not savings versus the existing shared pool.
- Compute Sanitizer memcheck on 100 probe switches reported **zero errors**.
- A real-weight Llama 3.1 8B diagnostic smoke run successfully captured four FULL
  decode descriptors (1, 2, 4, 8 tokens), replayed each plus 100 shape switches and
  retired the graphs. The KV allocation configuration was 256 MiB / 128 blocks.
  This used BF16, Triton attention, compilation mode 0 and
  `VLLM_USE_FLASHINFER_SAMPLER=0`; the installed nvcc rejected FlashInfer's
  `--compress-mode=size` option when its sampler was enabled.
- All configured pre-commit checks on the changed files passed, including mypy
  and Buildkite test tethering.

Raw local evidence from the completed runs is retained at
`/tmp/union-binding-probe-stress.json`, `/tmp/union-binding-probe-memcheck.json`,
`/tmp/union-binding-memcheck.log` and `/tmp/union-llama-smoke-3/`. The latter contains
job inputs, process logs, manifests, events and allocator snapshots. These local
artifacts are not committed benchmark fixtures and may disappear on temp cleanup.
The [tools README](../../benchmarks/cudagraphs/README.md) provides reproduction
commands. No GPU memory savings, serving latency improvement or model evaluation
result is claimed.

## Remaining gates

1. Complete repeated, all-descriptor production controls with frozen opportunity
   and noise targets. Validate physical KV storage, attribute allocation ownership
   and lifetimes, and compute the standalone workspace reference and shared-pool
   excess with uncertainty. The smoke run alone cannot decide Stage 1.
2. Certify explicit allocation/access contracts for any production candidate and
   demonstrate material recovery beyond the existing shared pool. The synthetic
   binding probe establishes only its own narrow contract.
3. Only after positive Stage 1/2 evidence, implement the production V1 adapter,
   serving opt-in, accounting and fallback integration, then run real-weight
   correctness/model evaluations, sanitizer coverage and paired performance tests.

Stage 3 remains unimplemented. Existing workspace sharing already reuses scratch;
adding an arena without these measurements would not satisfy the specification.
