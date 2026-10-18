# CUDA graph union memory: development and team handoff

Updated: 2026-10-18. Foundation checkpoint: `df108058bf`; R02 follow-up below.

This document is the working backlog and validation plan for coworkers. The
[specification](cudagraph_union_memory_spec.md) defines the required behavior and
stage gates; the [implementation notes](cudagraph_union_memory_implementation.md)
explain ownership and current limitations; the
[tool README](../../benchmarks/cudagraphs/README.md) documents command options.
Update this backlog and the implementation notes when a task changes status.

## Current state

The CPU planner, verifier, lifecycle contracts, diagnostic collector and explicit
CUDA binding probe are implemented. R02 now verifies live KV backing storage for
device-only full-attention caches. Real CUDA probe validation and real-weight
production baseline smoke runs succeeded. **Production union allocation is not
implemented, and savings beyond the existing shared CUDA graph pool have not been
established.** There is no serving opt-in to enable today.

The first proposed integration is V1 `FULL_DECODE_ONLY`, one CUDA device, sequential
dense-decoder execution and one validated attention backend/dtype combination.
The descriptor contracts also represent V2, but there is no V2 adapter. Before
production integration, record the runner decision: this project's spec selects
V1, while upstream development guidance favors V2 for new runner features.
Expanding to V2 or changing the initial runner requires an explicit design update.

| Stage | Current status | Exit condition still missing |
| --- | --- | --- |
| 0: baseline | Ownership inventory, baseline smoke and bounded KV backing verification done | A durable, reproducible evidence bundle and human review of R02 |
| 1: attribution | Collection/reporting tools implemented; measurements incomplete | All-descriptor controls, allocation attribution, noise bounds and frozen opportunity targets |
| 2: mechanism | Planner and synthetic explicit CUDA binding validated | A production allocation contract and demonstrated recovery beyond existing shared pools |
| 3: integration | Not started | Positive Stage 1/2 decisions before adding the production adapter and opt-in |
| 4: evaluation | Not started | Real-model correctness, repeated paired measurements and human acceptance |

Stages overlap in tooling work; completing a probe does not close a stage's exit
gate. If Stage 1 finds no material recoverable workspace, retain the negative
result and stop or reframe arena integration as required by the spec.

## Completed subtasks

“Done” below means the listed deliverable exists and has the stated bounded
validation. It does not mean production acceptance.

| ID | Deliverable | Code / documentation | Validation and limits |
| --- | --- | --- | --- |
| D01 | Native descriptors, immutable allocation/access contracts and fingerprints | [cudagraph_memory.py](../../vllm/compilation/cudagraph_memory.py) | CPU contract tests; V1/V2 fields retained without reducing identity to token count |
| D02 | Deterministic arena planner and independent verifier | [cudagraph_memory_planner.py](../../vllm/compilation/cudagraph_memory_planner.py) | Ordered/unordered accesses, aliases, initialization, alignment, overflow, nonmonotonic capacity and corrupt-plan tests |
| D03 | Staged publication, generation checks, fallback and retirement contracts | [cudagraph_memory.py](../../vllm/compilation/cudagraph_memory.py) | Fake-backend failures and cleanup tested; production runner wiring remains open |
| D04 | Saved-trace simulator and strict measurement schema | [run.py](../../benchmarks/cudagraphs/run.py), [schema.py](../../benchmarks/cudagraphs/schema.py) | CPU simulation/schema tests; traces cannot authorize capture in another process |
| D05 | Paired reporting and standalone/excess diagnostic summaries | [report.py](../../benchmarks/cudagraphs/report.py) | Coverage/control mismatches rejected; missing attribution stays unknown |
| D06 | Fresh-process production diagnostic controls | [collect.py](../../benchmarks/cudagraphs/collect.py), [production.py](../../benchmarks/cudagraphs/production.py) | Shared baseline, maximum-token, standalone and separate-pool tools implemented; only the shared baseline has completed a real-model smoke run |
| D07 | Explicit CUDA views, transactional graph capture and external outputs | [binding.py](../../benchmarks/cudagraphs/binding.py), [binding_probe.py](../../benchmarks/cudagraphs/binding_probe.py) | Synthetic fill/copy contracts only; 10,000 switches and targeted memcheck passed |
| D08 | Tests, CI wiring, ownership inventory and reproduction instructions | [planner tests](../../tests/v1/cudagraph/test_cudagraph_memory_planner.py), [binding tests](../../tests/v1/cudagraph/test_cudagraph_memory_binding.py), [collector tests](../../tests/benchmarks/test_cudagraph_memory_collect.py), [report tests](../../tests/benchmarks/test_cudagraph_memory_report.py) | Targeted tests and local hooks passed; remote CI execution is not asserted |
| D09 / R02 | Live KV backing deduplication and configuration reconciliation | [production.py](../../benchmarks/cudagraphs/production.py), [collector tests](../../tests/benchmarks/test_cudagraph_memory_collect.py) | Padded shared views and fail-closed rejection tested on CPU; Llama CUDA smoke verified 32 views sharing one 256 MiB backing; human review pending |

The CPU foundation is commit `8c008ea589`; collector, binding probe and diagnostic
reporting are commit `df108058bf`.

## Remaining work and assignment boundaries

Owner roles below are proposed work areas, **not assigned people**. A coworker
should claim a task with a name, branch and reviewer before changing shared files.
The project owner records stage decisions; implementation owners retain evidence;
a human reviewer checks code and conclusions.

| ID / priority | Proposed owner | Work and completion criteria | Dependencies |
| --- | --- | --- | --- |
| R01 / next | Measurement owner | Preserve raw evidence in durable team storage; record commit, dirty-source snapshot if any, environment, commands, model revision and artifact checksums. Verify another coworker can reproduce the smoke run. | Existing D06/D07 artifacts |
| R02 / review | Codex implementation; human reviewer pending; local `main` | Implemented in this follow-up: deduplicate live backing, reconcile the single-allocation contract, record per-group blocks/layout and view metadata, and reject unsupported/discrepant storage. See evidence below; reviewer closes this task. | D06; R01 evidence durability still open |
| R03 / next | Collector owner | Validate every diagnostic variant on the real manager. Audit pre-existing graph ownership, scoped teardown, pool isolation, conflicting engine controls, sampler/cleanup failures and restoration of patched methods. Record failures without publishing successful records. | D06, then R02 for trusted comparisons |
| R04 / next | Measurement + allocator owners | Run all standalone descriptors and shared/separate-pool controls; attribute persistent, reusable workspace, external and residual bytes without double counting. Obtain access/alias/dependency evidence for candidate storage. Snapshots alone are insufficient. | R01–R03 |
| R05 / gate | Project owner + reviewer | Freeze absolute MiB opportunity, graph-memory fraction, noise bounds, workloads, output boundaries and final thresholds before tuning. Compute `W_ref` and `E_base` with uncertainty; record continue/stop decision. | R04 |
| R06 / gate | Runtime + operator owners | Select production binding mechanism and exact sites; document initialization, aliases, streams, escapes and unsupported cases. Prove real kernels use planned addresses and improve on the shared-pool baseline. Record any PyTorch/compiler dependency and V1/V2 decision. | R04/R05; D07 is only a starting example |
| R07 / gated | Runtime owner | Add capability-based opt-in and complete-family discovery/capture/publication, immutable arena ownership, generation invalidation, one-time fallback and scoped retirement. Preserve existing output, workspace and allocator routing contracts. | Positive R05/R06 decisions |
| R08 / gated | Runtime + validation owners | Integrate arena/external/executable accounting and startup-budget behavior. Verify fixed KV primary comparisons, discovery peak, failure recovery, retained outputs and asynchronous consumers. | R07 |
| R09 / gated | Validation owner | Compare union, existing graphs and eager execution with restored KV/RNG/state; run real-weight generation, relevant model evaluation, production sanitizer coverage and long-run stress. | R07/R08 |
| R10 / gated | Performance owner | Implement an uninstrumented timing pass and alternating fresh-process baseline/union pairs. Measure per-descriptor and mixture completed-step, GPU and enqueue distributions; record confidence intervals and all failures. | R05, R07–R09 |
| R11 / final | Project owner + reviewer | Review two meaningful configurations, plots, absolute and fractional savings, startup/residency and compatibility limits. Record acceptance, inconclusive result or rejection. | R09/R10 |
| R12 / deferred | Future feature owners | Evaluate V2, piecewise/mixed execution, additional backends/hardware and other features with their own ownership/concurrency evidence; assess fixed-total-budget and serving capacity. | Initial acceptance and separate design/test scope |

R01–R03 and code-level investigation of allocation contracts can proceed together.
Keep one owner for edits to `collect.py`/`production.py`; coordinate schema changes
with the reporting owner. R07 must not bypass the measurement/mechanism gates.
Do not select offsets or eligibility by model/GPU names, resize a published arena,
or treat fewer captured graphs as a successful union implementation.

## Validation already completed

These are historical local results from the checkpoint work, not tests rerun by
this documentation change. CPU and GPU test counts overlap and must not be added
as a count of distinct tests.

| Check | Result | What it establishes / does not establish |
| --- | --- | --- |
| Planner + binding + reporter + collector suite | 101 passed, 1 CUDA test skipped in sandbox | CPU behavior and adapter contract checks; no full production union path |
| Binding suite on GPU 0 | 9 passed | Includes real CUDA probe test |
| Explicit binding stress | 10,000 switches, exact output checks, retired successfully | Synthetic address binding and output preservation; no production savings claim |
| Compute Sanitizer memcheck | 100 probe switches; zero errors | Targeted synthetic probe only; not all sanitizer checkers or production kernels |
| Llama 3.1 8B baseline smoke | Four FULL descriptors (1, 2, 4, 8), first replays plus 100 switches, retirement succeeded | Real-weight capture/dispatch; dummy inputs do not establish model accuracy or serving latency |
| Pre-commit checks | Passed, including mypy and Buildkite test tethering | Local checks only |

The smoke run used BF16, Triton attention, compilation mode 0, maximum sequence
length/batched tokens 128, maximum sequences 8 and configured KV storage of 256 MiB
with 128 blocks. The installed stack was PyTorch `2.14.0+cu126`, CUDA runtime 12.6,
NVIDIA driver 550.144.03 and A100 PCIe 40 GB. The editable vLLM and stable native
extension loaded. This is a tested configuration, not a minimum-version guarantee.

`VLLM_USE_FLASHINFER_SAMPLER=0` was used because the installed nvcc rejected the
FlashInfer sampler JIT option `--compress-mode=size`. Keep this control identical
between comparison variants and record it in the manifest. GPU 0 was available
for those runs; check availability again rather than assuming it remains idle.

### Evidence locations and portability

| Local artifact | Contents |
| --- | --- |
| `/tmp/union-binding-probe-stress.json` | Synthetic trace, verified plan, 10,000-switch result and retirement |
| `/tmp/union-binding-probe-memcheck.json` | 100-switch sanitizer probe result |
| `/tmp/union-binding-memcheck.log` | Memcheck zero-error summary |
| `/tmp/union-llama-smoke-3/` | Collection index, job config, worker log, manifest, event log and allocator snapshots |

The production run ID is `5ac2899675c54085bc632bd41964e381`; its pair ID is
`18adb83f66e349f5a1555783647f5fed`. These artifacts existed when this handoff was
written. They are local, uncommitted and vulnerable to `/tmp` cleanup; a Git clone
does not contain them. R01 must move/copy them to agreed team storage and record
that location and hashes. Preserve failed runs as well as successes. Older probe
and smoke artifacts preceded the final checkpoint; their source/environment
manifests and recorded dirty-state identity take precedence over this document's
checkpoint label. Reproduce from a clean commit for formal comparisons.

### R02 implementation and validation handoff

The collector previously summed `KVCacheTensor.size`, although all placements
describe the same backing allocation in `allocate_kv_cache`. It now counts live
`untyped_storage()` bytes once per device/base pointer and requires exactly one
backing of the configured size. It checks device, scheduler block count and
layer/view coverage; records group blocks/layout, configured placements, storage
identity, view offsets and byte strides; and rejects host groups, unknown layouts,
non-full-attention specs, split kernel-block views, missing views and unexplained
storage discrepancies.
Layout padding is included; allocator reservation slack is excluded. This checks
the current CUDA allocator contract, not arbitrary virtual-memory aliases or
kernel access lifetimes. Serving code and allocation behavior are unchanged.

CPU test design: the collector maps configuration and live views to measurement
controls or an explicit rejection. The regression guards against counting aliased
storage repeatedly; real CPU tensors with padded strided views are the cheapest
effective level. Existing collector tests also cover mismatched backing sizes,
extra allocations, unsupported views and worker rejection without a success record.

The four-suite command in the coworker runbook passed **110 tests**, with the
existing CUDA binding test skipped inside the sandbox. All configured pre-commit
checks on the five changed files passed, including mypy and CI test tethering.
The CUDA collector check below ran outside the sandbox; the skipped binding test
was not rerun for this collector-only change. No model evaluation was run.

The real-weight check used the earlier Llama engine options, GPU 0, BF16, Triton
attention, compilation mode 0 and the same FlashInfer sampler workaround:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python -m benchmarks.cudagraphs.collect \
  --engine-config /tmp/union-r02-engine.json --capture-sizes 1,2,4,8 \
  --kv-cache-memory-bytes 268435456 --variants shared_baseline \
  --repetitions 1 --stress-steps 100 --output-dir /tmp/union-r02-smoke
```

Result: successful capture/replay/retirement of four descriptors; 32 live views
deduplicate to one 268,435,456-byte backing, matching configuration. The cache has
128 blocks, 16 tokens/block, 65,536 bytes/layer/page and layout `LBNHC`. The saved
job contains the full engine options, including the pinned cached model revision.
This is a diagnostic smoke check, not an all-control comparison, model evaluation
or evidence of graph-memory savings. It ran from checkpoint `324ecfb2c5` with
source digest `329dc729dedcad6f89742d58ea12366f69e5074d83cefc7229fd927ec487880e`;
repeat from the clean follow-up commit for formal measurements.

Artifacts are local under
`/tmp/union-r02-smoke/72599ecd173b4a1b8ce170f161110a9c/`; R01 must preserve them
alongside the earlier runs. SHA-256 checksums:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `b4945b46c2d32d5c917bd04dbe1a7ff14dcb171d2c153525e16bd4228327d0f7` |
| `record.json` | `60a0b37b9deda4be1477cb50b25f2929ec62725d619828ddf9b59e4102f6f632` |
| `events.jsonl` | `c01c096532c72660d0eb3bf618fe95c076b62adc70be0dd9625d0ce6d8ff13ab` |

Next coworker actions: review R02, preserve the evidence (R01), and audit/run every
production control variant (R03). Stage 1 attribution and the R05/R06 decisions
remain prerequisites for production arena integration.

## Tests still required

| Area | Required evidence | Suggested location / level |
| --- | --- | --- |
| Collector fidelity | Physical KV equality, exact native descriptor coverage, all standalone references, isolated pools, unchanged capture order and control policy | Extend collector CPU tests; fresh-process production control runs |
| Collector failure paths | Contention, unavailable counters, unsupported config, capture/cleanup exceptions, no success artifact on failure, restored hooks and ownership-scoped teardown | Focused collector tests plus selected GPU reproductions |
| Production binding | Every chosen allocation matches size/alignment/access contract; hidden library streams excluded or ordered; per-replay initialization and live aliases proven | Minimal real-kernel GPU reproducer plus planner/verifier tests |
| Transaction and invalidation | Failure at each capture position, no partial publication, cleanup/fallback once, stale configuration/storage/lane rejection | Existing lifecycle suite and runner adapter tests |
| Output/consumer ownership | Outputs and live views survive later shapes; ordered asynchronous consumer; no arena growth/reuse before completion | Full-graph correctness suites and workspace tests |
| Dispatch and compatibility | Exact/padded descriptors unchanged; default and unsupported configurations retain existing allocation behavior | `tests/v1/cudagraph/test_cudagraph_dispatch.py` and `test_cudagraph_mode.py` |
| Model correctness | Identical inputs/weights/initial state; outputs and updated KV/state compared to baseline/eager; finite values and declared numerical tolerances; real generation and model eval | Existing full-graph tests and a relevant `tests/evals/` evaluation |
| GPU memory safety and residency | Production reproducer under applicable sanitizer checkers; at least 10,000 completed switches by default, samples each 1,000, bounded growth and successful retirement | GPU validation runs with raw logs |
| Performance and value | At least five independent alternating baseline/union pairs per configuration; identical coverage, actual KV and seeded dispatch; startup peak/capture cost and per-workload latency | New timing pass and reporting artifacts; two meaningful initial configurations |

Before adding a test, record the module's purpose, I/O contract, failure it guards
and cheapest effective level. Extend existing tests and fixtures first. A clean
sanitizer run does not replace ownership proof or model evaluation. Dummy-step
wall times from the current collector are diagnostic only; they cannot satisfy
completed-serving-step latency gates.

Spec Section 11 lists provisional targets: at least 50% recoverable-excess
reduction, at least 20% graph-attributable reduction on two meaningful
configurations, and completed-step median/p95 regressions below 2%/5%. Absolute
opportunity, noise, tolerances, workload weights and final thresholds are still
unfrozen. These figures are planning targets, not observed results or acceptance.

## Coworker runbook

Use the repository `uv` environment and `.venv/bin/python`; do not use system
Python or bare `pip`. Follow [AGENTS.md](../../AGENTS.md) for dependencies, hooks
and contributions. Do not replace an existing environment just to run these
checks. Confirm the intended device is free before CUDA work.

For the current automated coverage:

```bash
.venv/bin/python -m pytest \
  tests/v1/cudagraph/test_cudagraph_memory_planner.py \
  tests/v1/cudagraph/test_cudagraph_memory_binding.py \
  tests/benchmarks/test_cudagraph_memory_report.py \
  tests/benchmarks/test_cudagraph_memory_collect.py -q

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/cudagraph/test_cudagraph_memory_binding.py -q
```

The first command skips its CUDA case if CUDA is unavailable. In the GPU job,
confirm that case actually runs. In a sandbox without device access, run GPU work
in the project's authorized GPU environment; skip results are not GPU evidence.

Use the [tool README](../../benchmarks/cudagraphs/README.md) for the engine-config
example, collector, diagnostic reporter, 10,000-switch probe and sanitizer
commands. Start with `--variants shared_baseline --repetitions 1 --stress-steps 100`
to check the environment, then run the full control matrix. Use a new output path
for every run. The collector defaults to five repetitions and 10,000 switches;
reserve enough GPU time before launching all standalone controls. It supports
only diagnostic variants today, not a `union` treatment or final timing pass.

Formal experiments should start from a clean, shared commit. Store artifacts
outside the source checkout: untracked files enter the collector's source digest,
so generating results inside the checkout can make otherwise matching runs look
like different source states. Retain resolved native descriptors, actual capture
and dispatch records, KV controls, raw counters, logs, failure reasons and exact
commands. Never edit manifests to make mismatched runs comparable.

## Handoff and review workflow

1. Claim an R-task with owner, reviewer, branch and dependencies. Keep measurement
   and runtime work separate until the prerequisite evidence is available.
2. Agree on schema/API changes before parallel edits to shared modules. Keep
   schema versions, fixtures, producer and reporter behavior consistent.
3. Submit a focused commit with commands/results and artifact locations. Mark
   tests as passed, failed, skipped or not run; state the reason for gaps.
4. Run relevant tests and `.venv/bin/pre-commit run` on staged files. Include agent
   attribution and sign-off when applicable. Have a human review every changed
   line before proposing an upstream contribution.
5. Update task status and the implementation notes with the commit and evidence.
   The reviewer closes a task only when its completion criteria are met; the
   project owner records each continue/stop stage decision.

Suggested task update template:

```text
Task ID / status:
Owner / reviewer / branch:
Commit and scope:
Dependencies and decision needed:
Commands / results (including failures and skips):
Artifact location / checksums / environment identity:
Known limitations:
Next action:
```

Before any upstream PR or re-review, perform the repository's duplicate-work
checks, apply `pr-checklist`, include human review/testing and model-evaluation
results where required, and disclose AI assistance. This handoff and checkpoint
push are project collaboration artifacts, not a claim of upstream readiness.
