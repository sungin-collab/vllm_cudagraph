# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic byte placement and independent verification of GPU access DAGs."""

from dataclasses import asdict, dataclass
from itertools import combinations
from math import prod

from .cudagraph_memory import (
    MAX_BYTES,
    PLANNER_VERSION,
    SCHEMA_VERSION,
    AccessSpec,
    AllocationSpec,
    Exclusion,
    FamilyTrace,
    MemberTrace,
    MemoryPlan,
    Placement,
    UnionMemoryError,
)


def _reject(reason: str, detail: str) -> None:
    raise UnionMemoryError(reason, detail)


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= MAX_BYTES:
        _reject("invalid_trace", f"{name} out of bounds: {value}")


def _alignment(value: int) -> None:
    _integer(value, "alignment", 1)
    if value & (value - 1):
        _reject("invalid_trace", "alignment must be a power of two")


def _end(offset: int, size: int) -> int:
    _integer(offset, "offset")
    _integer(size, "size")
    _integer(offset + size, "end")
    return offset + size


def _align(value: int, alignment: int) -> int:
    result = (value + alignment - 1) // alignment * alignment
    _integer(result, "aligned offset")
    return result


def _overlap(a: int, a_size: int, b: int, b_size: int) -> bool:
    return a_size > 0 and b_size > 0 and a < b + b_size and b < a + a_size


@dataclass
class _Normalized:
    roots: dict[str, AllocationSpec]
    accesses: dict[str, list[AccessSpec]]
    after: dict[str, set[str]]
    exclusions: tuple[Exclusion, ...]


def _reachability(member: MemberTrace) -> dict[str, set[str]]:
    nodes = member.operations
    if len(set(nodes)) != len(nodes) or any(not node for node in nodes):
        _reject("invalid_trace", "duplicate or empty operation")
    edges: dict[str, set[str]] = {node: set() for node in nodes}
    indegrees = dict.fromkeys(nodes, 0)
    for dep in member.dependencies:
        if (
            dep.source not in edges
            or dep.target not in edges
            or dep.evidence
            not in ("stream_order", "graph_edge", "event_wait", "owner_contract")
        ):
            _reject("unknown_lifetime", "missing dependency node or evidence")
        if dep.target not in edges[dep.source]:
            edges[dep.source].add(dep.target)
            indegrees[dep.target] += 1
    pending = [node for node in nodes if indegrees[node] == 0]
    ordered = []
    while pending:
        node = pending.pop()
        ordered.append(node)
        for target in edges[node]:
            indegrees[target] -= 1
            if indegrees[target] == 0:
                pending.append(target)
    if len(ordered) != len(nodes):
        _reject("unknown_lifetime", "cyclic GPU dependency graph")
    after: dict[str, set[str]] = {node: set() for node in nodes}
    for node in reversed(ordered):
        for target in edges[node]:
            after[node].add(target)
            after[node].update(after[target])
    return after


def _initialized(accesses: list[AccessSpec], after: dict[str, set[str]]) -> None:
    for read in accesses:
        if read.kind == "write" or read.size == 0:
            continue
        intervals = sorted(
            (write.offset, write.offset + write.size)
            for write in accesses
            if write.kind == "write" and read.operation in after[write.operation]
        )
        covered = read.offset
        for start, end in intervals:
            if start > covered:
                break
            covered = max(covered, end)
        if covered < read.offset + read.size:
            _reject("uninitialized_read", f"{read.allocation_id} at {read.operation}")


def _normalize(member: MemberTrace) -> _Normalized:
    member.descriptor.validate()
    coverage = member.coverage
    if coverage.gaps or not coverage.access_contract or not coverage.discovery_method:
        _reject("unknown_lifetime", "incomplete access coverage")
    if not coverage.alias_contract:
        _reject("unknown_alias", "missing alias contract")
    sites = {allocation.site_id for allocation in member.allocations}
    if set(coverage.declared_sites) != sites or set(coverage.observed_sites) != sites:
        _reject("capture_contract_mismatch", "incomplete allocation site coverage")
    after = _reachability(member)
    roots: dict[str, AllocationSpec] = {}
    occurrences: set[tuple[str, int]] = set()
    bindings: set[str] = set()
    for allocation in member.allocations:
        root_id = allocation.allocation_id
        if (
            not root_id
            or root_id in roots
            or not allocation.site_id
            or not allocation.owner
        ):
            _reject("invalid_trace", "missing or duplicate allocation identity/owner")
        _integer(allocation.occurrence, "occurrence")
        occurrence = (allocation.site_id, allocation.occurrence)
        if occurrence in occurrences:
            _reject("invalid_trace", "duplicate allocation occurrence")
        occurrences.add(occurrence)
        _integer(allocation.requested_bytes, "requested bytes")
        _integer(allocation.reserved_bytes, "reserved bytes")
        _alignment(allocation.alignment)
        if allocation.requested_bytes > allocation.reserved_bytes:
            _reject("invalid_trace", f"undersized storage: {root_id}")
        if allocation.storage_class not in (
            "scratch",
            "persistent",
            "output",
            "uncontrolled",
        ):
            _reject("invalid_trace", f"unknown storage class: {root_id}")
        if allocation.binding_id is not None:
            if not allocation.binding_id or allocation.binding_id in bindings:
                _reject("invalid_trace", "duplicate or empty binding identity")
            bindings.add(allocation.binding_id)
        roots[root_id] = allocation

    # References normalize to root-relative byte ranges, including late views.
    refs = {key: (key, 0, root.reserved_bytes) for key, root in roots.items()}
    for alias in member.aliases:
        if not alias.alias_id or alias.alias_id in refs or alias.root_id not in roots:
            _reject("unknown_alias", "invalid alias root or identity")
        root = roots[alias.root_id]
        if _end(alias.offset, alias.span) > root.reserved_bytes:
            _reject("unknown_alias", "alias outside root storage")
        _alignment(alias.alignment)
        _alignment(alias.itemsize)
        if (
            not alias.dtype
            or len(alias.shape) != len(alias.strides)
            or alias.offset % alias.alignment
            or root.alignment < alias.alignment
            or root.alignment < alias.itemsize
            or alias.offset % alias.itemsize
        ):
            _reject("unknown_alias", "unsupported alias layout/alignment")
        stride = 1
        for dimension, actual_stride in reversed(list(zip(alias.shape, alias.strides))):
            _integer(dimension, "dimension")
            _integer(actual_stride, "stride")
            if dimension > 1 and actual_stride != stride:
                _reject("unknown_alias", "only contiguous aliases are supported")
            stride *= dimension
        if prod(alias.shape) * alias.itemsize != alias.span:
            _reject("unknown_alias", "shape does not describe accessible span")
        refs[alias.alias_id] = (alias.root_id, alias.offset, alias.span)

    escaped: set[str] = set()
    for escape in member.escapes:
        if (
            escape.allocation_id not in refs
            or not escape.owner
            or not escape.completion
            or not escape.reason
        ):
            _reject("unknown_lifetime", "incomplete escape contract")
        escaped.add(refs[escape.allocation_id][0])
    exclusions: list[Exclusion] = []
    managed: dict[str, AllocationSpec] = {}
    for key, root in roots.items():
        reason = None
        if key in escaped:
            reason = "escaping_storage"
        elif root.storage_class != "scratch":
            reason = root.storage_class
        elif root.binding_id is None:
            reason = "uncontrolled_workspace"
        if reason is None:
            managed[key] = root
        else:
            exclusions.append(
                Exclusion(member.member_id, key, root.reserved_bytes, reason)
            )
    accesses: dict[str, list[AccessSpec]] = {key: [] for key in roots}
    for access in member.accesses:
        if access.allocation_id not in refs or access.operation not in after:
            _reject("unknown_lifetime", "access references missing storage/operation")
        root_id, offset, span = refs[access.allocation_id]
        if _end(access.offset, access.size) > span:
            _reject("invalid_trace", "access outside storage/view")
        if access.kind not in ("read", "write", "read_write"):
            _reject("invalid_trace", "unknown access kind")
        if access.size:
            accesses[root_id].append(
                AccessSpec(
                    root_id,
                    access.operation,
                    offset + access.offset,
                    access.size,
                    access.kind,
                )
            )
    for key, root in managed.items():
        if root.reserved_bytes and not accesses[key]:
            _reject("unknown_lifetime", f"no accesses for {key}")
        _initialized(accesses[key], after)
    return _Normalized(
        managed,
        accesses,
        after,
        tuple(sorted(exclusions, key=lambda entry: entry.allocation_id)),
    )


def _conflict(normalized: _Normalized, left: str, right: str) -> bool:
    a = normalized.accesses[left]
    b = normalized.accesses[right]
    return not (
        all(y.operation in normalized.after[x.operation] for x in a for y in b)
        or all(x.operation in normalized.after[y.operation] for x in a for y in b)
    )


def _validate_family(trace: FamilyTrace) -> None:
    if trace.schema_version != SCHEMA_VERSION:
        _reject("invalid_trace", "unsupported trace schema")
    if trace.identity.exclusive_members is not True:
        _reject("unsupported_feature", "family members may execute concurrently")
    for key, value in asdict(trace.identity).items():
        if key != "exclusive_members" and (not isinstance(value, str) or not value):
            _reject("invalid_trace", f"missing family identity: {key}")
    members = [member.member_id for member in trace.members]
    if len(set(members)) != len(members):
        _reject("invalid_trace", "duplicate descriptor")
    if len({member.descriptor.runner for member in trace.members}) > 1:
        _reject("unsupported_runner", "mixed runner family")
    for member in trace.members:
        member.descriptor.validate()
        values = dict(member.descriptor.fields)
        mode = values.get("runtime_mode", values.get("cg_mode"))
        if mode != "FULL":
            _reject(
                "unsupported_feature", "initial planner requires complete FULL graphs"
            )


def plan(
    trace: FamilyTrace, *, arena_alignment: int = 256, arena_limit: int = MAX_BYTES
) -> MemoryPlan:
    """Pack exclusive members independently using stable descending-size order."""
    _validate_family(trace)
    _alignment(arena_alignment)
    _integer(arena_limit, "arena limit")
    placements: list[Placement] = []
    exclusions: list[Exclusion] = []
    capacity = 0
    for member in trace.members:
        normalized = _normalize(member)
        exclusions.extend(normalized.exclusions)
        placed: list[Placement] = []
        conflicts = {
            frozenset((a, b))
            for a, b in combinations(normalized.roots, 2)
            if _conflict(normalized, a, b)
        }
        for key, root in sorted(
            normalized.roots.items(),
            key=lambda item: (-item[1].reserved_bytes, item[0]),
        ):
            arena_alignment = max(arena_alignment, root.alignment)
            blockers = [
                entry
                for entry in placed
                if frozenset((key, entry.allocation_id)) in conflicts
            ]
            candidates = {0} | {
                _align(entry.offset + entry.reserved_bytes, root.alignment)
                for entry in blockers
            }
            offset = next(
                candidate
                for candidate in sorted(candidates)
                if all(
                    not _overlap(
                        candidate,
                        root.reserved_bytes,
                        entry.offset,
                        entry.reserved_bytes,
                    )
                    for entry in blockers
                )
            )
            capacity = max(capacity, _end(offset, root.reserved_bytes))
            placed.append(Placement(member.member_id, key, offset, root.reserved_bytes))
        placements.extend(placed)
    capacity = _align(capacity, arena_alignment)
    if capacity > arena_limit:
        _reject("arena_limit", f"requires {capacity} bytes, limit {arena_limit}")
    result = MemoryPlan(
        trace.fingerprint,
        tuple(sorted(placements, key=lambda p: (p.member_id, p.allocation_id))),
        tuple(sorted(exclusions, key=lambda e: (e.member_id, e.allocation_id))),
        capacity,
        arena_alignment,
    )
    verify(trace, result)
    return result


def verify(trace: FamilyTrace, memory_plan: MemoryPlan) -> None:
    """Reject corrupt layouts using fresh trace normalization and pairwise checks.

    This does not trust the placer's conflict cache, sizes, exclusions or capacity.
    It cannot prove that a backend supplied complete kernel-access contracts.
    """
    _validate_family(trace)
    if (
        memory_plan.fingerprint != trace.fingerprint
        or memory_plan.planner_version != PLANNER_VERSION
    ):
        _reject("stale_generation", "plan does not match trace/planner")
    _integer(memory_plan.arena_bytes, "arena bytes")
    _alignment(memory_plan.arena_alignment)
    if memory_plan.arena_bytes % memory_plan.arena_alignment:
        _reject("layout_conflict", "unaligned arena capacity")
    placements: dict[tuple[str, str], Placement] = {}
    for entry in memory_plan.placements:
        placement_key = (entry.member_id, entry.allocation_id)
        if placement_key in placements:
            _reject("layout_conflict", "duplicate placement")
        placements[placement_key] = entry
    expected: set[tuple[str, str]] = set()
    exclusions: list[Exclusion] = []
    for member in trace.members:
        normalized = _normalize(member)
        exclusions.extend(normalized.exclusions)
        for key, root in normalized.roots.items():
            lookup = (member.member_id, key)
            expected.add(lookup)
            if lookup not in placements:
                _reject("layout_conflict", f"missing placement: {key}")
            entry = placements[lookup]
            end = _end(entry.offset, entry.reserved_bytes)
            if (
                entry.reserved_bytes != root.reserved_bytes
                or end > memory_plan.arena_bytes
                or entry.offset % root.alignment
                or memory_plan.arena_alignment < root.alignment
            ):
                _reject("layout_conflict", f"invalid span/alignment: {key}")
        for left, right in combinations(normalized.roots, 2):
            a = placements[(member.member_id, left)]
            b = placements[(member.member_id, right)]
            # Recompute ordering from the trace, never the placement decisions.
            if _overlap(
                a.offset, a.reserved_bytes, b.offset, b.reserved_bytes
            ) and _conflict(normalized, left, right):
                _reject("layout_conflict", f"overlapping live roots: {left}, {right}")
    if set(placements) != expected:
        _reject("layout_conflict", "unexpected placement")
    if memory_plan.arena_bytes and not any(
        entry.reserved_bytes for entry in placements.values()
    ):
        _reject("layout_conflict", "empty layout must not reserve storage")
    if tuple(sorted(exclusions, key=lambda e: (e.member_id, e.allocation_id))) != (
        memory_plan.exclusions
    ):
        _reject("layout_conflict", "incorrect external storage accounting")
