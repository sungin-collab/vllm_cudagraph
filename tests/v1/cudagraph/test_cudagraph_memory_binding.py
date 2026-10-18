# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit bindings reject changed requests before publishing CUDA graphs.

The CPU validator guards allocation/access completeness; the GPU probe checks
actual addresses, outputs and retirement using tiny real captures.
"""

from dataclasses import replace

import pytest

from benchmarks.cudagraphs.binding import RequestValidator, access_fingerprint
from benchmarks.cudagraphs.binding_probe import probe_trace, run_probe
from benchmarks.cudagraphs.cpu import memory, planner


def complete_requests(validator):
    for allocation in validator.member.allocations:
        validator.request(
            allocation, access_fingerprint(validator.member, allocation.allocation_id)
        )
    for alias in validator.member.aliases:
        validator.alias(alias)


def test_complete_explicit_contract_and_nonmonotonic_arena():
    trace = probe_trace()
    layout = planner.plan(trace)
    assert layout.arena_bytes == 512
    for member in trace.members:
        validator = RequestValidator(member)
        complete_requests(validator)
        for operation in member.operations:
            validator.operation(
                operation, tuple(a for a in member.accesses if a.operation == operation)
            )
        validator.finish()
    assert all(p.offset == 0 for p in layout.placements)


@pytest.mark.parametrize(
    "field,value",
    [
        ("requested_bytes", 32),
        ("alignment", 32),
        ("binding_id", "new"),
        ("occurrence", 1),
    ],
)
def test_changed_allocation_request_is_rejected(field, value):
    member = probe_trace().members[0]
    validator = RequestValidator(member)
    changed = replace(member.allocations[0], **{field: value})
    with pytest.raises(memory.UnionMemoryError, match="capture_contract_mismatch"):
        validator.request(changed, access_fingerprint(member, changed.allocation_id))


def test_repeated_request_or_changed_access_fingerprint_is_rejected():
    member = probe_trace().members[0]
    allocation = member.allocations[0]
    validator = RequestValidator(member)
    with pytest.raises(memory.UnionMemoryError, match="changed access contract"):
        validator.request(allocation, "old access contract")
    validator.request(allocation, access_fingerprint(member, allocation.allocation_id))
    with pytest.raises(memory.UnionMemoryError, match="repeated"):
        validator.request(
            allocation, access_fingerprint(member, allocation.allocation_id)
        )


def test_missing_requests_operations_and_changed_alias_are_rejected():
    validator = RequestValidator(probe_trace().members[0])
    with pytest.raises(memory.UnionMemoryError, match="missing"):
        validator.finish()
    complete_requests(validator)
    with pytest.raises(memory.UnionMemoryError, match="missing"):
        validator.finish()
    with pytest.raises(memory.UnionMemoryError, match="alias"):
        validator.alias(replace(validator.member.aliases[0], offset=4))


def test_operation_order_and_access_changes_are_rejected():
    validator = RequestValidator(probe_trace().members[0])
    complete_requests(validator)
    with pytest.raises(memory.UnionMemoryError, match="unexpected operation"):
        validator.operation("b.write", ())
    with pytest.raises(memory.UnionMemoryError, match="unexpected operation"):
        validator.operation("a.write", ())


def test_real_cuda_bindings_preserve_outputs_across_shape_switches():
    import torch

    if torch.version.hip is not None or not torch.cuda.is_available():
        pytest.skip("CUDA device required for explicit binding probe")
    result = run_probe(switches=100, seed=17)
    assert result["retired"]
    assert result["completed_shape_switches"] == 100
    assert result["plan"]["arena_bytes"] == 512
    assert result["production_capture"] is False
    assert result["savings_vs_shared_baseline"] is None
