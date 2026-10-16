# CUDA graph union-memory tools

These tools implement the CPU prerequisites of the
[union-memory specification](../../docs/design/cudagraph_union_memory_spec.md).
They do not change production capture, allocate CUDA storage, or demonstrate GPU
memory savings. Production collection and binding remain gated on the Stage 1/2
measurements described in the specification.

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

`schema.py` defines a strict `RunRecord` reporting input. A future production
collector must emit the detailed manifests and raw events from Section 9 of the
specification alongside these records. `controls_digest` covers model/weights,
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

The collector must enforce fresh processes, alternate treatment order, control
GPU contention, capture all descriptors using the production manager, preserve
state, and measure startup/teardown. Those collection operations, standalone
references, excess-workspace statistics, plots and acceptance gates are pending.

## CPU tests

```bash
.venv/bin/python -m pytest --noconftest \
  tests/v1/cudagraph/test_cudagraph_memory_planner.py -q
```

`--noconftest` avoids the repository-wide model fixtures, which import PyTorch.
The test module uses no fixtures from them. `cpu.py` loads the unchanged source
modules under a private package to avoid `vllm.__init__` initializing PyTorch;
it does not mock their behavior or affect normal production imports. The same
suite is wired into the existing CPU V1 CI job with its normal environment.
