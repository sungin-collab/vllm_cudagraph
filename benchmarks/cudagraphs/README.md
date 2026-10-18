# CUDA graph union-memory tools

These tools implement the planner, diagnostic collection, reporting and explicit
CUDA binding probe for the
[union-memory specification](../../docs/design/cudagraph_union_memory_spec.md).
They leave serving defaults unchanged. The probe allocates CUDA storage but does
not demonstrate savings against the production shared pool. Production union
binding remains gated on positive Stage 1/2 evidence.

## Simulate a layout

From the repository root, using the repository's `uv` virtual environment:

```bash
.venv/bin/python -m benchmarks.cudagraphs.run \
  benchmarks/cudagraphs/examples/ordered.json \
  --output /tmp/union-plan.json
```

The example declares two 64-byte allocations whose GPU accesses are ordered.
Their offsets are both zero; the default backend alignment rounds arena capacity
to 256 bytes. This is a synthetic layout, not a measurement of the existing shared
CUDA graph pool. The output explicitly records `artifact_kind: cpu_simulation`
and pending GPU binding validation. An invalid plan exits with status 1 and a
reason-coded rejection artifact. Output files must not already exist.

A trace is the JSON representation of `FamilyTrace` in
`vllm/compilation/cudagraph_memory.py`. It includes the full native descriptor,
portable configuration identity, stable allocation occurrences, byte accesses,
aliases, GPU dependency evidence, escapes and coverage attestations. Schema
versions are mandatory; unknown fields and duplicate JSON keys are rejected.
Persisted plans are diagnostic artifacts and cannot authorize capture in another
process.

The planner accepts complete FULL executions with mutually exclusive members.
Allocation alignment must be a power of two. Aliases must be contiguous views
rooted directly in known storage; other layouts reject the trace. Every scratch
read requires complete byte-range initialization by preceding writes in that
invocation. Read-modify-write does not initialize previously undefined data.
Unordered operations conflict, including two accesses in the same operation.

External exclusions describe logical storage per member. They are **not** a
physical-memory total: a collector must deduplicate shared backing across members
before constructing measured memory categories. Requested bytes and reservation
slack likewise must not be added to each other.

## Report saved measurements

`schema.py` defines a strict `RunRecord` reporting input. The diagnostic collector
emits manifests, raw events and allocator snapshots alongside these records;
physical allocation/access attribution remains pending. `controls_digest` covers model/weights,
KV/layout limits, compilation/backend/allocator settings, warmup, output retention,
GC/trimming policy, instrumentation, completed-step boundary, and workload policy.
Run identity and treatment are excluded from that digest. `manifest_digest`
identifies the complete manifest; the actual seeded dispatch has its own digest.
Hashes identify controls; they do not prove the collector measured them correctly.

```bash
.venv/bin/python -m benchmarks.cudagraphs.report \
  /path/to/baseline-record.json /path/to/union-record.json \
  --output /tmp/union-summary.json --seed 0
```

Pass all independent pairs for one configuration together. The primary comparison
requires `shared_baseline` and `union`, identical capture order/coverage, graph
counts, actual KV bytes/block counts, and matching dispatch within each pair.
Fallback is never successful union execution. Diagnostic and timing passes cannot
be mixed. Reduced-coverage references are rejected as primary baselines.

Memory categories are disjoint physical byte deltas: persistent buffers, reusable
workspace including slack and the arena, external allocations, and unattributed
residual. Unknown values require a reason and remain unknown in totals. Signed
measurement deltas are preserved. Allocated/reserved counters must not be summed;
raw counter snapshots belong in the separate event log.

Timing reports retain per-run median/p95 values for each workload, descriptor and
method. Only steady samples are summarized; warmup and first replay remain in the
raw records. At least 100 steady samples per workload/run are required. Intervals
use a deterministic percentile bootstrap over independent paired runs, with at
least five pairs; they do not treat individual replay samples as independent.
Measurement uncertainty is reported separately from sampling intervals, so these
intervals alone must not be used for acceptance. GPU, enqueue and completed-step
times are never added together. Correctness and acceptance remain explicit.

The diagnostic collector runs fresh processes, checks GPU contention, observes the
production manager and records startup/teardown. It resolves the shared baseline
first, then reverses the diagnostic control order on alternating repetitions.
This is not the alternating baseline/union timing experiment required for final
acceptance. Allocation liveness attribution, timing passes, plots and production
acceptance gates remain pending.

## Collect production diagnostic controls

Provide a JSON object of `LLM` options, using cached real weights. For example:

```json
{
  "model": "/path/to/cached/llama-3.1-8b",
  "dtype": "bfloat16",
  "max_model_len": 128,
  "max_num_batched_tokens": 128,
  "max_num_seqs": 8,
  "attention_config": {"backend": "TRITON_ATTN"},
  "compilation_config": {"mode": 0}
}
```

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python -m benchmarks.cudagraphs.collect \
  --engine-config /path/to/engine.json --capture-sizes 1,2,4,8 \
  --kv-cache-memory-bytes 268435456 --repetitions 5 \
  --stress-steps 10000 --output-dir /path/to/new-results

.venv/bin/python -m benchmarks.cudagraphs.report --diagnostics \
  /path/to/new-results/*/record.json --output /tmp/diagnostic-summary.json
```

The output directory must be new. Defaults include shared baseline, maximum-token,
every standalone descriptor and separate-pool controls. For a smoke run use
`--variants shared_baseline --repetitions 1 --stress-steps 100`. The collector
requires an idle CUDA device, NVML and a dense text decoder on the V1 runner;
distributed execution, MoE, multimodal, hybrid, LoRA and speculative decoding are
unsupported. Explicit KV bytes avoid the startup graph-memory estimator. KV bytes
and blocks in records come from the runner's allocated cache configuration;
independent physical-storage verification remains pending.

Every run retains status/reason, software/source/environment identity, resolved
and captured descriptors, graph handles/pools, sampled device counters and full
allocator snapshots. Sampling peaks are lower bounds. An unsupported or failed
run exits nonzero and retains its artifacts. Memory attribution remains unknown
until ownership and access evidence is available. Diagnostic dummy steps exercise
dispatch/capture, not model accuracy or normal serving output consumers.

## Run the explicit CUDA binding probe

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.cudagraphs.binding_probe \
  --switches 10000 --output /tmp/binding-probe.json

CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --error-exitcode 99 \
  --log-file /tmp/binding-memcheck.log \
  .venv/bin/python -m benchmarks.cudagraphs.binding_probe \
  --switches 100 --output /tmp/binding-memcheck.json
```

The probe binds declared fill/copy kernels to planned addresses, includes a
nonmonotonic member capacity and contiguous alias, keeps outputs external, poisons
scratch and checks exact values. It validates storage generations and drains before
retirement. It cannot redirect ordinary production allocations or establish their
access contracts. Output explicitly records `production_capture: false` and no
savings claim.

## Tests

```bash
.venv/bin/python -m pytest --noconftest \
  tests/v1/cudagraph/test_cudagraph_memory_planner.py \
  tests/v1/cudagraph/test_cudagraph_memory_binding.py \
  tests/benchmarks/test_cudagraph_memory_report.py \
  tests/benchmarks/test_cudagraph_memory_collect.py -q
```

`--noconftest` avoids the repository-wide model fixtures, which import PyTorch.
These test modules use no fixtures from them. The collector adapter test imports
vLLM and the binding test imports PyTorch; only the planner/report modules are
dependency-light. `cpu.py` loads the unchanged source
modules under a private package to avoid `vllm.__init__` initializing PyTorch;
it does not mock their behavior or affect normal production imports. The same
suite is wired into the existing CPU V1 CI job with its normal environment.
