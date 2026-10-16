# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict CPU artifact decoding and comparable-run records.

A result is an attributed measurement, not raw allocated+reserved counter sums.
Collectors must retain raw events/manifests alongside these reporting inputs.
"""

import json
import math
import types
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from benchmarks.cudagraphs.cpu import memory

if TYPE_CHECKING:
    from vllm.compilation.cudagraph_memory import Descriptor
else:
    Descriptor = memory.Descriptor


def decode(record_type, value):
    """Decode JSON records without coercing bools/floats to byte counts."""
    origin = get_origin(record_type)
    args = get_args(record_type)
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                return decode(choice, value)
            except ValueError:
                continue
        raise ValueError(f"Value does not match {record_type}")
    if origin is Literal:
        if any(type(value) is type(choice) and value == choice for choice in args):
            return value
    elif origin is tuple:
        if not isinstance(value, list):
            raise ValueError("Expected JSON array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], item) for item in value)
        if len(value) == len(args):
            return tuple(decode(kind, item) for kind, item in zip(args, value))
    elif isinstance(record_type, type) and is_dataclass(record_type):
        if not isinstance(value, dict):
            raise ValueError("Expected JSON object")
        declared = {field.name: field for field in fields(record_type)}
        if value.keys() - declared.keys():
            raise ValueError("Unknown record fields")
        required = {
            name
            for name, field in declared.items()
            if field.default is MISSING and field.default_factory is MISSING
        }
        if "schema_version" in declared:
            required.add("schema_version")
        if required - value.keys():
            raise ValueError(f"Missing fields: {sorted(required - value.keys())}")
        hints = get_type_hints(record_type)
        return record_type(
            **{key: decode(hints[key], item) for key, item in value.items()}
        )
    elif record_type is float and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    elif type(value) is record_type:
        return value
    raise ValueError(f"Value does not match {record_type}")


def load(path: Path, record_type):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return decode(
        record_type, json.loads(path.read_text(), object_pairs_hook=unique_object)
    )


@dataclass(frozen=True)
class Header:
    schema_version: int
    run_id: str
    pair_id: str
    variant: Literal[
        "shared_baseline", "max_tokens_only", "standalone", "separate_pools", "union"
    ]
    measurement_pass: Literal["diagnostic", "timing"]
    status: Literal["success", "failure", "unsupported"]


@dataclass(frozen=True)
class ByteMeasurement:
    """A signed delta; unknown values carry a reason rather than a fake zero."""

    value: int | None
    uncertainty: int | None
    unknown_reason: str | None

    def validate(self) -> None:
        if self.value is None:
            if not self.unknown_reason or self.uncertainty is not None:
                raise ValueError(
                    "Unknown bytes require a reason and unknown uncertainty"
                )
        elif (
            type(self.value) is not int
            or type(self.uncertainty) is not int
            or self.uncertainty < 0
            or self.unknown_reason is not None
        ):
            raise ValueError("Measured bytes require integer uncertainty")


@dataclass(frozen=True)
class MemoryCategories:
    """Disjoint physical bytes, including arena exactly once in workspace."""

    persistent: ByteMeasurement
    workspace: ByteMeasurement
    external: ByteMeasurement
    unattributed: ByteMeasurement

    def total(self) -> ByteMeasurement:
        categories = (self.persistent, self.workspace, self.external, self.unattributed)
        for category in categories:
            category.validate()
        if any(category.value is None for category in categories):
            return ByteMeasurement(None, None, "incomplete memory attribution")
        return ByteMeasurement(
            sum(cast(int, category.value) for category in categories),
            sum(cast(int, category.uncertainty) for category in categories),
            None,
        )


@dataclass(frozen=True)
class LatencySample:
    workload: str
    member_id: str
    phase: Literal["warmup", "first_replay", "steady", "stress"]
    method: Literal["gpu_event", "cpu_enqueue", "completed_step"]
    duration_ns: int


@dataclass(frozen=True)
class RunRecord:
    header: Header
    # Hash of the fixed experiment controls, excluding treatment/run identity.
    controls_digest: str
    manifest_digest: str
    runner: Literal["v1", "v2"]
    descriptors: tuple[Descriptor, ...]
    executable_count: int
    actual_kv_bytes: int
    actual_kv_blocks: tuple[int, ...]
    dispatch_digest: str
    fallback_reasons: tuple[str, ...]
    memory: MemoryCategories
    latency: tuple[LatencySample, ...]
    correctness: Literal["passed", "failed", "pending"]

    def validate(self) -> None:
        if self.header.schema_version != memory.SCHEMA_VERSION:
            raise ValueError("Unsupported artifact schema")
        if not all(
            (
                self.header.run_id,
                self.header.pair_id,
                self.controls_digest,
                self.manifest_digest,
                self.dispatch_digest,
            )
        ):
            raise ValueError("Missing run/control identity")
        if not self.descriptors:
            raise ValueError("No resolved capture descriptors")
        for descriptor in self.descriptors:
            descriptor.validate()
            if descriptor.runner != self.runner:
                raise ValueError("Runner mismatch")
        ids = {descriptor.member_id for descriptor in self.descriptors}
        if len(ids) != len(self.descriptors):
            raise ValueError("Duplicate capture descriptor")
        counts = (self.executable_count, self.actual_kv_bytes, *self.actual_kv_blocks)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Invalid graph/KV count")
        if self.header.status == "success" and self.executable_count < len(ids):
            raise ValueError("Incomplete executable coverage")
        self.memory.total()
        for sample in self.latency:
            if (
                sample.member_id not in ids
                or not sample.workload
                or type(sample.duration_ns) is not int
                or sample.duration_ns < 0
            ):
                raise ValueError("Invalid latency sample/descriptor")
        if (
            self.header.variant == "union"
            and self.header.status == "success"
            and self.fallback_reasons
        ):
            raise ValueError("Union fallback cannot be reported as success")


def validate_pair(baseline: RunRecord, union: RunRecord) -> None:
    for record in (baseline, union):
        record.validate()
        if record.header.status != "success" or record.fallback_reasons:
            raise ValueError(
                "Primary comparison requires successful runs without fallback"
            )
    if (
        baseline.header.variant != "shared_baseline"
        or union.header.variant != "union"
        or baseline.header.pair_id != union.header.pair_id
        or baseline.header.run_id == union.header.run_id
    ):
        raise ValueError("Invalid baseline/union pairing")
    for field in (
        "controls_digest",
        "runner",
        "descriptors",
        "executable_count",
        "actual_kv_bytes",
        "actual_kv_blocks",
        "dispatch_digest",
    ):
        if getattr(baseline, field) != getattr(union, field):
            raise ValueError(f"Incompatible comparison: {field}")
    if baseline.header.measurement_pass != union.header.measurement_pass:
        raise ValueError("Cannot mix diagnostic and timing passes")
