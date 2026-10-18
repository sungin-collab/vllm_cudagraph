# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 1 reporting must preserve coverage, attribution, and uncertainty.

Pure CPU records exercise the public summary interface: incomplete references or
incompatible controls cannot produce an apparent workspace opportunity.
"""

from dataclasses import asdict, replace

import pytest

from benchmarks.cudagraphs.cpu import memory
from benchmarks.cudagraphs.report import summarize_diagnostics
from benchmarks.cudagraphs.schema import (
    ByteMeasurement,
    Header,
    MemoryCategories,
    RunRecord,
)


def descriptor(tokens, requests=None):
    return memory.Descriptor(
        "v1",
        tuple(
            sorted(
                {
                    "runtime_mode": "FULL",
                    "num_tokens": tokens,
                    "num_reqs": tokens if requests is None else requests,
                    "uniform": True,
                    "has_lora": False,
                    "num_active_loras": 0,
                }.items()
            )
        ),
    )


@pytest.fixture
def diagnostic_records():
    descriptors = tuple(descriptor(tokens) for tokens in (1, 4, 8))
    zero = ByteMeasurement(0, 0, None)
    baseline = RunRecord(
        Header(1, "baseline", "group", "shared_baseline", "diagnostic", "success"),
        "fixed-controls",
        "manifest",
        "v1",
        descriptors,
        len(descriptors),
        4096,
        (4,),
        "sequence",
        (),
        MemoryCategories(zero, ByteMeasurement(1000, 20, None), zero, zero),
        (),
        "pending",
    )
    standalone = [
        replace(
            baseline,
            header=replace(
                baseline.header, run_id=f"standalone-{i}", variant="standalone"
            ),
            descriptors=(desc,),
            executable_count=1,
            dispatch_digest=f"standalone-{i}",
            memory=replace(
                baseline.memory, workspace=ByteMeasurement(size, noise, None)
            ),
        )
        for i, (desc, size, noise) in enumerate(
            zip(descriptors, (100, 700, 500), (5, 30, 50))
        )
    ]
    return [baseline, *standalone]


def test_workspace_reference_uses_every_descriptor_and_propagates_uncertainty(
    diagnostic_records,
):
    baseline, *standalone = diagnostic_records
    maximum = replace(
        standalone[-1],
        header=replace(
            standalone[-1].header, run_id="max-tokens", variant="max_tokens_only"
        ),
    )
    separate = replace(
        baseline,
        header=replace(baseline.header, run_id="separate", variant="separate_pools"),
        memory=replace(baseline.memory, workspace=ByteMeasurement(9000, 20, None)),
    )
    result = summarize_diagnostics([*diagnostic_records, maximum, separate])
    group = result["groups"][0]
    assert group["standalone_workspace_reference_bytes"] == asdict(
        ByteMeasurement(700, 30, None)
    )
    assert group["baseline_excess_workspace_bytes"] == asdict(
        ByteMeasurement(300, 50, None)
    )
    assert group["excess_above_uncertainty"] is True
    assert len(group["records"]) == 6
    assert "saved_bytes" not in result
    assert result["acceptance"].startswith("pending:")


@pytest.mark.parametrize(
    "category,measurement",
    [
        ("workspace", ByteMeasurement(None, None, "no workspace counter")),
        ("workspace", ByteMeasurement(-10, 20, None)),
        ("external", ByteMeasurement(None, None, "no process counter")),
        ("unattributed", ByteMeasurement(32, 0, None)),
        ("unattributed", ByteMeasurement(0, 32, None)),
    ],
)
def test_unresolved_attribution_preserves_raw_data_and_unknown_opportunity(
    diagnostic_records, category, measurement
):
    reference = diagnostic_records[1]
    diagnostic_records[1] = replace(
        reference, memory=replace(reference.memory, **{category: measurement})
    )
    group = summarize_diagnostics(diagnostic_records)["groups"][0]
    assert group["records"][1]["memory"][category] == asdict(measurement)
    assert group["standalone_workspace_reference_bytes"]["value"] is None
    assert group["baseline_excess_workspace_bytes"]["uncertainty"] is None
    assert group["excess_above_uncertainty"] is None


def test_uncertain_or_negative_excess_is_not_a_proven_opportunity(diagnostic_records):
    baseline = diagnostic_records[0]
    diagnostic_records[0] = replace(
        baseline,
        memory=replace(baseline.memory, workspace=ByteMeasurement(690, 20, None)),
    )
    group = summarize_diagnostics(diagnostic_records)["groups"][0]
    assert group["baseline_excess_workspace_bytes"] == asdict(
        ByteMeasurement(0, 40, None)
    )
    assert group["excess_above_uncertainty"] is False


def test_missing_standalone_references_remain_unknown(diagnostic_records):
    group = summarize_diagnostics(diagnostic_records[:1])["groups"][0]
    assert group["standalone_workspace_reference_bytes"] == asdict(
        ByteMeasurement(None, None, "no standalone descriptor references")
    )
    assert group["baseline_excess_workspace_bytes"]["value"] is None


@pytest.mark.parametrize("fallback", [False, True])
def test_unsuccessful_reference_cannot_establish_workspace_opportunity(
    diagnostic_records, fallback
):
    reference = diagnostic_records[1]
    diagnostic_records[1] = (
        replace(reference, fallback_reasons=("unsupported_allocator",))
        if fallback
        else replace(reference, header=replace(reference.header, status="failure"))
    )
    group = summarize_diagnostics(diagnostic_records)["groups"][0]
    assert group["standalone_workspace_reference_bytes"]["value"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"controls_digest": "other-configuration"},
        {"actual_kv_bytes": 8192},
        {"actual_kv_blocks": (8,)},
    ],
)
def test_diagnostic_controls_require_identical_configuration_and_kv(
    diagnostic_records, changes
):
    diagnostic_records[1] = replace(diagnostic_records[1], **changes)
    with pytest.raises(ValueError, match="Incompatible diagnostic controls"):
        summarize_diagnostics(diagnostic_records)


@pytest.mark.parametrize("duplicate", [False, True])
def test_partial_or_duplicate_standalone_coverage_is_rejected(
    diagnostic_records, duplicate
):
    records = diagnostic_records[:-1]
    if duplicate:
        reference = diagnostic_records[1]
        records.append(
            replace(reference, header=replace(reference.header, run_id="copy"))
        )
    with pytest.raises(ValueError, match="cover each descriptor once"):
        summarize_diagnostics(records)


def test_same_tokens_with_distinct_native_fields_need_separate_references(
    diagnostic_records,
):
    baseline, reference = diagnostic_records[:2]
    original = descriptor(4, 4)
    other = descriptor(4, 2)
    baseline = replace(baseline, descriptors=(original, other), executable_count=2)
    reference = replace(reference, descriptors=(original,))
    with pytest.raises(ValueError, match="cover each descriptor once"):
        summarize_diagnostics([baseline, reference])


def test_reduced_coverage_control_cannot_impersonate_separate_pool_baseline(
    diagnostic_records,
):
    reference = diagnostic_records[-1]
    separate = replace(
        reference, header=replace(reference.header, variant="separate_pools")
    )
    with pytest.raises(ValueError, match="Incompatible separate-pool coverage"):
        summarize_diagnostics([diagnostic_records[0], separate])


def test_maximum_token_control_must_use_an_actual_maximum_descriptor(
    diagnostic_records,
):
    reference = diagnostic_records[1]
    maximum = replace(
        reference, header=replace(reference.header, variant="max_tokens_only")
    )
    with pytest.raises(ValueError, match="Maximum-token reference"):
        summarize_diagnostics([diagnostic_records[0], maximum])


@pytest.mark.parametrize(
    "change", [{"measurement_pass": "timing"}, {"variant": "union"}]
)
def test_stage1_report_rejects_timing_and_union_treatments(diagnostic_records, change):
    baseline = diagnostic_records[0]
    diagnostic_records[0] = replace(baseline, header=replace(baseline.header, **change))
    with pytest.raises(ValueError):
        summarize_diagnostics(diagnostic_records)
