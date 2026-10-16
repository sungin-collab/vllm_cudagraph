# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests: trace -> verified layout or explicit rejection.

Guard against unsafe storage reuse, stale plans and premature arena release.
Synthetic DAGs and a fake backend are the cheapest level for these invariants;
CUDA binding and model correctness require separate GPU validation.
"""

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmarks.cudagraphs.cpu import memory as m
from benchmarks.cudagraphs.cpu import planner as p

pytestmark = pytest.mark.cpu_test


def descriptor(tokens=1, **overrides):
    fields = dict(
        runtime_mode="FULL",
        num_tokens=tokens,
        num_reqs=tokens,
        uniform=True,
        has_lora=False,
        num_active_loras=0,
    )
    fields.update(overrides)
    return m.Descriptor("v1", tuple(sorted(fields.items())))


def member(*, tokens=1, sizes=(64, 64), ordered=True):
    allocations = tuple(
        m.AllocationSpec(
            str(i), f"site{i}", 0, size, size, 16, "backend", "scratch", f"binding{i}"
        )
        for i, size in enumerate(sizes)
    )
    operations = tuple(f"{i}{suffix}" for i in range(len(sizes)) for suffix in "wr")
    dependencies = [
        m.DependencySpec(f"{i}w", f"{i}r", "stream_order") for i in range(len(sizes))
    ]
    if ordered:
        dependencies.extend(
            m.DependencySpec(f"{i - 1}r", f"{i}w", "event_wait")
            for i in range(1, len(sizes))
        )
    accesses = tuple(
        m.AccessSpec(str(i), f"{i}{suffix}", 0, size, kind)
        for i, size in enumerate(sizes)
        for suffix, kind in (("w", "write"), ("r", "read"))
    )
    sites = tuple(a.site_id for a in allocations)
    return m.MemberTrace(
        descriptor(tokens),
        operations,
        tuple(dependencies),
        allocations,
        (),
        accesses,
        (),
        m.TraceCoverage(
            sites, sites, "explicit accesses v1", "no aliases v1", "synthetic"
        ),
    )


def family(*members):
    return m.FamilyTrace(
        m.FamilyIdentity(
            model_revision="synthetic",
            config_digest="config",
            tensor_layout_digest="t",
            kv_layout_digest="kv",
            backend="fake",
            allocator="fake",
            dtype="uint8",
            compilation_digest="program",
            software_digest="versions",
            device_capabilities="cpu simulator",
            storage_backend_version="fake v1",
            device="cpu:0",
            execution_lane="single ordered lane",
            exclusive_members=True,
        ),
        tuple(members),
    )


@pytest.mark.parametrize("ordered,expected", [(True, 64), (False, 128)])
def test_only_proven_order_allows_overlay(ordered, expected):
    trace = family(member(ordered=ordered))
    layout = p.plan(trace, arena_alignment=16)
    assert layout.arena_bytes == expected
    p.verify(trace, layout)


def test_late_view_extends_root_lifetime():
    data = member()
    data = replace(
        data,
        aliases=(m.AliasSpec("view", "0", 0, 32, 16, "uint8", 1, (32,), (1,)),),
        accesses=data.accesses + (m.AccessSpec("view", "1r", 0, 32, "read"),),
    )
    assert p.plan(family(data), arena_alignment=16).arena_bytes == 128


def test_middle_descriptor_sets_capacity_and_native_fields_remain_distinct():
    trace = family(
        member(tokens=1, sizes=(64,)),
        member(tokens=4, sizes=(512,)),
        member(tokens=8, sizes=(128,)),
    )
    assert p.plan(trace).arena_bytes == 512
    same_tokens = replace(trace.members[1], descriptor=descriptor(4, num_reqs=2))
    layout = p.plan(replace(trace, members=trace.members + (same_tokens,)))
    assert len({entry.member_id for entry in layout.placements}) == 4


@pytest.mark.parametrize("kind", ["read", "read_write"])
def test_read_requires_per_invocation_initialization(kind):
    data = member(sizes=(64,))
    data = replace(data, accesses=(m.AccessSpec("0", "0w", 0, 64, kind),))
    with pytest.raises(m.UnionMemoryError, match="uninitialized_read"):
        p.plan(family(data))


def test_partial_initialization_covers_actual_read_ranges():
    data = member(sizes=(64,))
    accesses = (
        m.AccessSpec("0", "0w", 0, 32, "write"),
        m.AccessSpec("0", "0w", 32, 32, "write"),
        m.AccessSpec("0", "0r", 0, 64, "read"),
    )
    p.plan(family(replace(data, accesses=accesses)))
    with pytest.raises(m.UnionMemoryError, match="uninitialized_read"):
        p.plan(family(replace(data, accesses=accesses[1:])))


def test_unordered_write_does_not_initialize_read():
    data = replace(member(sizes=(64,)), dependencies=())
    with pytest.raises(m.UnionMemoryError, match="uninitialized_read"):
        p.plan(family(data))


@pytest.mark.parametrize("storage_class", ["persistent", "output", "uncontrolled"])
def test_persistent_and_output_storage_stay_external(storage_class):
    data = member(sizes=(64,))
    data = replace(
        data, allocations=(replace(data.allocations[0], storage_class=storage_class),)
    )
    layout = p.plan(family(data))
    assert layout.arena_bytes == 0
    assert layout.exclusions[0].reserved_bytes == 64
    assert layout.exclusions[0].reason == storage_class


def test_retained_view_excludes_its_whole_backing():
    data = member(sizes=(64,))
    data = replace(
        data,
        aliases=(m.AliasSpec("view", "0", 0, 32, 16, "uint8", 1, (32,), (1,)),),
        escapes=(m.EscapeSpec("view", "caller", "consumer event", "retained output"),),
    )
    layout = p.plan(family(data))
    assert not layout.placements
    assert layout.exclusions[0].reason == "escaping_storage"


@pytest.mark.parametrize("sizes", [(), (0,)])
def test_empty_layout_does_not_require_storage(sizes):
    assert p.plan(family(member(sizes=sizes))).arena_bytes == 0
    assert p.plan(family()).arena_bytes == 0


def test_plan_is_deterministic_under_allocation_and_access_permutations():
    data = member(sizes=(48, 32, 64), ordered=False)
    baseline = p.plan(family(data))
    reordered = replace(
        data,
        allocations=tuple(reversed(data.allocations)),
        accesses=tuple(reversed(data.accesses)),
    )
    assert p.plan(family(reordered)).placements == baseline.placements
    assert p.plan(family(data)) == baseline


@pytest.mark.parametrize("bad_size", [-1, m.MAX_BYTES + 1, True])
def test_invalid_integer_sizes_rejected(bad_size):
    with pytest.raises(m.UnionMemoryError, match="invalid_trace"):
        p.plan(family(member(sizes=(bad_size,))))


def test_arena_limit_and_alignment_overflow_rejected():
    with pytest.raises(m.UnionMemoryError, match="arena_limit"):
        p.plan(family(member()), arena_limit=32)
    with pytest.raises(m.UnionMemoryError, match="invalid_trace"):
        p.plan(family(member(sizes=(m.MAX_BYTES,))))


@pytest.mark.parametrize(
    "damage,reason",
    [
        (
            lambda d: replace(
                d,
                dependencies=d.dependencies
                + (m.DependencySpec("1r", "0w", "event_wait"),),
            ),
            "cyclic",
        ),
        (
            lambda d: replace(
                d, coverage=replace(d.coverage, gaps=("unknown stream",))
            ),
            "unknown_lifetime",
        ),
        (
            lambda d: replace(d, coverage=replace(d.coverage, alias_contract="")),
            "unknown_alias",
        ),
        (
            lambda d: replace(d, coverage=replace(d.coverage, observed_sites=())),
            "capture_contract_mismatch",
        ),
        (lambda d: replace(d, accesses=()), "unknown_lifetime"),
        (
            lambda d: replace(
                d,
                aliases=(
                    m.AliasSpec("view", "0", 60, 32, 16, "uint8", 1, (32,), (1,)),
                ),
            ),
            "unknown_alias",
        ),
    ],
)
def test_incomplete_or_invalid_evidence_is_rejected(damage, reason):
    with pytest.raises(m.UnionMemoryError, match=reason):
        p.plan(family(damage(member())))


@pytest.mark.parametrize(
    "damage",
    [
        lambda plan: replace(plan, arena_bytes=0),
        lambda plan: replace(plan, placements=plan.placements[:-1]),
        lambda plan: replace(plan, placements=plan.placements + plan.placements[:1]),
        lambda plan: replace(
            plan, placements=tuple(replace(p, offset=0) for p in plan.placements)
        ),
        lambda plan: replace(
            plan,
            placements=(replace(plan.placements[0], offset=1),) + plan.placements[1:],
        ),
        lambda plan: replace(plan, fingerprint="stale"),
        lambda plan: replace(plan, arena_alignment=1),
    ],
)
def test_independent_verifier_rejects_corrupt_layout(damage):
    trace = family(member(ordered=False))
    with pytest.raises(m.UnionMemoryError):
        p.verify(trace, damage(p.plan(trace)))


def test_random_total_order_intervals_never_overlap_live_storage():
    rng = random.Random(71)
    for _ in range(50):
        data = member(sizes=tuple(rng.randrange(1, 256) for _ in range(8)))
        nodes = tuple(str(i) for i in range(16))
        intervals = [sorted(rng.sample(range(16), 2)) for _ in range(8)]
        data = replace(
            data,
            operations=nodes,
            dependencies=tuple(
                m.DependencySpec(str(i), str(i + 1), "stream_order") for i in range(15)
            ),
            accesses=tuple(
                m.AccessSpec(str(i), str(op), 0, root.reserved_bytes, kind)
                for i, root in enumerate(data.allocations)
                for op, kind in zip(intervals[i], ("write", "read"))
            ),
        )
        layout = p.plan(family(data), arena_alignment=16)
        offsets = {int(e.allocation_id): e.offset for e in layout.placements}
        for i in range(8):
            for j in range(i):
                if max(intervals[i][0], intervals[j][0]) <= min(
                    intervals[i][1], intervals[j][1]
                ):
                    assert (
                        offsets[i] + data.allocations[i].reserved_bytes <= offsets[j]
                        or offsets[j] + data.allocations[j].reserved_bytes <= offsets[i]
                    )


class FakeBackend:
    def __init__(self, fail_at=None, error=None):
        self.events = []
        self.fail_at = fail_at
        self.error = error or m.UnionMemoryError(
            "capture_contract_mismatch", "injected"
        )
        self.owner = SimpleNamespace(
            generation=m.StorageGeneration("config", "kv", "a", "lane")
        )
        self.graphs = []

    def step(self, event):
        self.events.append(event)
        if event == self.fail_at:
            raise self.error

    def reserve(self, plan):
        self.step("reserve")
        return self.owner

    def capture(self, owner, member, plan):
        self.step(f"capture{len(self.graphs)}")
        graph = SimpleNamespace(replay=lambda: self.step("replay"))
        self.graphs.append(graph)
        return graph

    def validate(self, owner, graphs):
        assert len(graphs) == len(self.graphs)
        self.step("validate")

    def drain(self, owner):
        self.step("drain")

    def destroy(self, owner):
        self.step("destroy")
        self.graphs.clear()

    def release(self, owner):
        assert not self.graphs
        self.step("release")


@pytest.mark.parametrize(
    "fail_at", ["reserve", "capture0", "capture1", "capture2", "validate"]
)
def test_failed_family_falls_back_once_after_safe_cleanup(fail_at):
    backend = FakeBackend(fail_at)
    trace = family(*(member(tokens=t) for t in (1, 4, 8)))
    fallback = []
    result = m.CaptureTransaction(backend).capture(
        trace, p.plan(trace), lambda reason: fallback.append(reason)
    )
    assert result is None
    assert fallback == ["capture_contract_mismatch"]
    if fail_at != "reserve":
        assert backend.events[-3:] == ["drain", "destroy", "release"]


def test_device_fault_does_not_run_fallback():
    backend = FakeBackend("capture1", RuntimeError("device fault"))
    trace = family(member(tokens=1), member(tokens=4))
    with pytest.raises(RuntimeError, match="device fault"):
        m.CaptureTransaction(backend).capture(
            trace, p.plan(trace), lambda _: pytest.fail("unsafe fallback")
        )
    assert backend.events[-3:] == ["drain", "destroy", "release"]


def test_stale_generation_blocks_launch_and_retirement_is_idempotent():
    backend = FakeBackend()
    trace = family(member())
    published = m.CaptureTransaction(backend).capture(
        trace, p.plan(trace), lambda _: pytest.fail("fallback")
    )
    published.replay(trace.members[0].member_id, backend.owner.generation)
    backend.owner.generation = replace(
        backend.owner.generation, persistent_storage="new"
    )
    with pytest.raises(m.UnionMemoryError, match="stale_generation"):
        published.replay(trace.members[0].member_id, backend.owner.generation)
    assert backend.events.count("replay") == 1
    published.retire()
    published.retire()
    assert backend.events.count("release") == 1
    with pytest.raises(m.UnionMemoryError, match="stale_generation"):
        published.replay(trace.members[0].member_id, backend.owner.generation)


def test_live_consumer_prevents_release_and_retirement_can_retry():
    backend = FakeBackend()
    trace = family(member())
    published = m.CaptureTransaction(backend).capture(
        trace, p.plan(trace), lambda _: None
    )
    backend.fail_at = "drain"
    with pytest.raises(m.UnionMemoryError):
        published.retire()
    assert "destroy" not in backend.events and "release" not in backend.events
    backend.fail_at = None
    published.retire()
    assert backend.events[-3:] == ["drain", "destroy", "release"]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"runner": "v2"}, "unsupported_runner"),
        ({"platform": "cpu"}, "unsupported_allocator"),
        ({"allocation_contract": None}, "unsupported_allocator"),
        ({"unsupported_features": ("sleep",)}, "unsupported_feature"),
        ({"ordered_execution_lane": False}, "unknown_lifetime"),
        ({"validated_backend_dtype": False}, "unsupported_feature"),
        ({"device_count": 2}, "unsupported_feature"),
        ({"graph_mode": "FULL_AND_PIECEWISE"}, "unsupported_feature"),
    ],
)
def test_initial_eligibility_fails_closed(changes, reason):
    capabilities = m.RunnerCapabilities(
        "v1", "cuda", "FULL_DECODE_ONLY", 1, True, "validated contract", True, ()
    )
    assert m.assess(capabilities).supported
    result = m.assess(replace(capabilities, **changes))
    assert not result.supported and reason in result.reason_codes


def test_v2_descriptor_identity_keeps_all_native_fields():
    values = dict(
        cg_mode="FULL",
        num_tokens=8,
        num_reqs=4,
        uniform_token_count=2,
        max_query_len=2,
        num_active_loras=0,
        num_ubatches=1,
    )
    first = m.Descriptor("v2", tuple(sorted(values.items())))
    data = replace(member(tokens=8), descriptor=first)
    values["num_ubatches"] = 2
    second = m.Descriptor("v2", tuple(sorted(values.items())))
    assert first.member_id != second.member_id
    trace = family(data, replace(data, descriptor=second))
    assert len(p.plan(trace).placements) == 4
    with pytest.raises(m.UnionMemoryError, match="unsupported_runner"):
        p.plan(family(replace(data, descriptor=replace(first, schema_version=2))))


def test_failed_cleanup_retains_transaction_and_never_falls_back():
    class BusyBackend(FakeBackend):
        busy = True

        def drain(self, owner):
            self.step("drain")
            if self.busy:
                raise RuntimeError("consumer still active")

    backend = BusyBackend("capture1")
    trace = family(member(tokens=1), member(tokens=4))
    transaction = m.CaptureTransaction(backend)
    with pytest.raises(RuntimeError, match="consumer still active"):
        transaction.capture(trace, p.plan(trace), lambda _: pytest.fail("fallback"))
    assert "release" not in backend.events
    backend.busy = False
    transaction.cleanup()
    assert backend.events[-3:] == ["drain", "destroy", "release"]


def test_fallback_failure_propagates_with_contract_error_context():
    backend = FakeBackend("validate")
    trace = family(member())

    def fallback(reason):
        raise RuntimeError("baseline capture also failed")

    with pytest.raises(RuntimeError, match="baseline capture also failed") as exc:
        m.CaptureTransaction(backend).capture(trace, p.plan(trace), fallback)
    assert isinstance(exc.value.__context__, m.UnionMemoryError)


def test_saved_trace_roundtrip_is_strict(tmp_path):
    import json
    from dataclasses import asdict

    from benchmarks.cudagraphs.schema import load

    trace = family(member())
    path = tmp_path / "trace.json"
    raw = asdict(trace)
    path.write_text(json.dumps(raw))
    assert p.plan(load(path, m.FamilyTrace)) == p.plan(trace)
    raw["members"][0]["allocations"][0]["requested_bytes"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load(path, m.FamilyTrace)
    path.write_text('{"schema_version": 1, "schema_version": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load(path, m.FamilyTrace)


def measurement_pair(pair_id="pair", *, measurement_pass="diagnostic"):
    from benchmarks.cudagraphs.schema import (
        ByteMeasurement,
        Header,
        LatencySample,
        MemoryCategories,
        RunRecord,
    )

    desc = descriptor()
    zero = ByteMeasurement(0, 0, None)
    baseline = RunRecord(
        Header(
            1, pair_id + "base", pair_id, "shared_baseline", measurement_pass, "success"
        ),
        "fixed-controls",
        "manifest",
        "v1",
        (desc,),
        1,
        4096,
        (4,),
        "sequence",
        (),
        MemoryCategories(zero, ByteMeasurement(1024, 16, None), zero, zero),
        tuple(
            LatencySample("decode", desc.member_id, "steady", "completed_step", 100)
            for _ in range(100)
        ),
        "pending",
    )
    union = replace(
        baseline,
        header=replace(baseline.header, run_id=pair_id + "union", variant="union"),
        memory=replace(baseline.memory, workspace=ByteMeasurement(512, 16, None)),
        latency=tuple(replace(sample, duration_ns=101) for sample in baseline.latency),
    )
    return baseline, union


@pytest.mark.parametrize(
    "changes",
    [
        {"actual_kv_bytes": 8192},
        {"actual_kv_blocks": (8,)},
        {"dispatch_digest": "changed"},
        {"controls_digest": "changed"},
        {"executable_count": 2},
        {"fallback_reasons": ("unsupported_allocator",)},
        {"descriptors": (descriptor(4),)},
    ],
)
def test_measurement_comparison_rejects_changed_controls_and_fallback(changes):
    from benchmarks.cudagraphs.schema import validate_pair

    baseline, union = measurement_pair()
    with pytest.raises(ValueError):
        validate_pair(baseline, replace(union, **changes))


def test_unknown_and_signed_bytes_are_preserved_without_double_counting():
    from benchmarks.cudagraphs.schema import ByteMeasurement

    baseline, _ = measurement_pair()
    assert baseline.memory.total().value == 1024
    changed = replace(baseline.memory, external=ByteMeasurement(-16, 32, None))
    assert changed.total().value == 1008
    changed = replace(
        changed, external=ByteMeasurement(None, None, "no process counter")
    )
    assert changed.total().value is None
    with pytest.raises(ValueError):
        ByteMeasurement(None, None, None).validate()


def test_reporting_pairs_are_independent_units_and_warmup_is_excluded():
    from benchmarks.cudagraphs.report import summarize
    from benchmarks.cudagraphs.schema import RunRecord

    records: list[RunRecord] = []
    for i in range(5):
        base, union = measurement_pair(str(i), measurement_pass="timing")
        union = replace(
            union,
            latency=union.latency
            + (replace(union.latency[0], phase="warmup", duration_ns=100000),),
        )
        records.extend((base, union))
    result = summarize(records, seed=7)
    assert result == summarize(records, seed=7)
    assert result["independent_pairs"] == 5
    assert result["saved_bytes"]["mean"] == 512
    assert result["saved_bytes"]["ci95"] == [512, 512]
    assert result["latency_regression"][0]["median_fraction"]["mean"] == pytest.approx(
        0.01
    )
    assert result["correctness"] == "pending_or_failed"
    assert result["acceptance"].startswith("pending")


def test_simulator_cli_roundtrip(tmp_path):
    import json
    import subprocess
    import sys
    from dataclasses import asdict
    from pathlib import Path

    trace = family(member())
    source, target = tmp_path / "trace.json", tmp_path / "plan.json"
    source.write_text(json.dumps(asdict(trace)))
    command = [
        sys.executable,
        "-m",
        "benchmarks.cudagraphs.run",
        str(source),
        "--output",
        str(target),
    ]
    cwd = Path(__file__).resolve().parents[3]
    subprocess.run(command, cwd=cwd, check=True, capture_output=True)
    result = json.loads(target.read_text())
    assert result["artifact_kind"] == "cpu_simulation"
    assert result["plan"]["arena_bytes"] == 256
    assert result["gpu_binding_validation"] == "pending"
    rejected = tmp_path / "rejected.json"
    command[-1] = str(rejected)
    output = subprocess.run(
        command + ["--arena-limit", "0"], cwd=cwd, capture_output=True
    )
    assert output.returncode == 1
    assert json.loads(rejected.read_text())["reason"] == "arena_limit"
