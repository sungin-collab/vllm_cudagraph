# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal explicit CUDA binding probe; not a production memory benchmark."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from benchmarks.cudagraphs.binding import ExplicitCudaBackend, access_fingerprint
from benchmarks.cudagraphs.cpu import memory as m
from benchmarks.cudagraphs.cpu import planner


def probe_trace(software: str = "unexecuted", device: str = "cuda:0"):
    """Three complete serial contracts, including a nonmonotonic capacity."""
    members = []
    for tokens, size in ((1, 64), (4, 512), (8, 128)):
        fields = dict(
            runtime_mode="FULL",
            num_tokens=tokens,
            num_reqs=tokens,
            uniform=True,
            has_lora=False,
            num_active_loras=0,
        )
        allocations = tuple(
            m.AllocationSpec(
                name, name, 0, size, size, 16, "explicit probe", kind, name
            )
            for name, kind in (
                ("a", "scratch"),
                ("b", "scratch"),
                ("out_a", "output"),
                ("out_b", "output"),
            )
        )
        operations = ("a.write", "a.copy", "b.write", "b.copy")
        dependencies = tuple(
            m.DependencySpec(left, right, "stream_order")
            for left, right in zip(operations, operations[1:])
        )
        accesses = (
            m.AccessSpec("a", "a.write", 0, size, "write"),
            m.AccessSpec("a.view", "a.copy", 0, size, "read"),
            m.AccessSpec("out_a", "a.copy", 0, size, "write"),
            m.AccessSpec("b", "b.write", 0, size, "write"),
            m.AccessSpec("b", "b.copy", 0, size, "read"),
            m.AccessSpec("out_b", "b.copy", 0, size, "write"),
        )
        sites = tuple(a.site_id for a in allocations)
        members.append(
            m.MemberTrace(
                m.Descriptor("v1", tuple(sorted(fields.items()))),
                operations,
                dependencies,
                allocations,
                (
                    m.AliasSpec(
                        "a.view", "a", 0, size, 16, "float32", 4, (size // 4,), (1,)
                    ),
                ),
                accesses,
                (),
                m.TraceCoverage(
                    sites,
                    sites,
                    "fill_ writes all elements; copy_ reads/writes all elements v1",
                    "one contiguous full-span alias v1",
                    "explicit probe source",
                ),
            )
        )
    return m.FamilyTrace(
        m.FamilyIdentity(
            "synthetic",
            "explicit fill/copy v1",
            "contiguous float32",
            "no KV",
            "explicit CUDA probe",
            "torch CUDA allocator",
            "float32",
            "no compilation",
            software,
            "single CUDA stream",
            "explicit binding v1",
            device,
            "one stream",
            True,
        ),
        tuple(members),
    )


def run_probe(*, device: int = 0, switches: int = 10000, seed: int = 0):
    import torch

    if torch.version.hip is not None or not torch.cuda.is_available():
        raise m.UnionMemoryError("unsupported_allocator", "CUDA is unavailable")
    torch.accelerator.set_device_index(device)
    stream = torch.cuda.Stream(device=device)
    trace = probe_trace(
        f"torch={torch.__version__};cuda={torch.version.cuda}", f"cuda:{device}"
    )
    plan = planner.plan(trace)
    programs = {}
    external = {}
    validators = {}
    for index, member in enumerate(trace.members):
        key = member.member_id
        size = member.allocations[0].requested_bytes // 4
        external[key] = {
            name: torch.empty(size, dtype=torch.float32, device=f"cuda:{device}")
            for name in ("out_a", "out_b")
        }

        def program(session, member=member, value=index + 1):
            tensors = {
                allocation.allocation_id: session.request(
                    allocation,
                    (allocation.requested_bytes // 4,),
                    torch.float32,
                    access_fingerprint(member, allocation.allocation_id),
                )
                for allocation in member.allocations
            }
            a_view = session.alias(member.aliases[0], torch.float32)
            for name in member.operations:
                accesses = tuple(a for a in member.accesses if a.operation == name)
                with session.operation(name, accesses):
                    if name == "a.write":
                        tensors["a"].fill_(value)
                    elif name == "a.copy":
                        tensors["out_a"].copy_(a_view)
                    elif name == "b.write":
                        tensors["b"].fill_(value + 10)
                    else:
                        tensors["out_b"].copy_(tensors["b"])

        def validate(outputs, value=index + 1):
            for name, expected in (("out_a", value), ("out_b", value + 10)):
                torch.testing.assert_close(
                    outputs[name],
                    torch.full_like(outputs[name], expected),
                    rtol=0,
                    atol=0,
                )

        programs[key] = program
        validators[key] = validate

    transaction = None
    family = None
    with torch.cuda.stream(stream):
        backend = ExplicitCudaBackend(trace, programs, external, validators, stream)
        transaction = m.CaptureTransaction(backend)

        def reject(reason):
            raise m.UnionMemoryError(reason, "probe cannot fall back")

        try:
            family = transaction.capture(trace, plan, reject)
            assert backend.owner is not None
            generation = backend.owner.generation
            keys = family.member_ids
            rng = random.Random(seed)
            for index in range(switches):
                if index % 4 == 0:
                    key = keys[0]
                elif index % 4 == 1:
                    key = keys[-1]
                else:
                    key = rng.choice(keys)
                family.replay(key, generation)
            stream.synchronize()
            for key in keys:
                validators[key](
                    {name: tensor.cpu() for name, tensor in external[key].items()}
                )
            assert backend.owner.raw is not None
            result = {
                "schema_version": m.SCHEMA_VERSION,
                "artifact_kind": "explicit_cuda_binding_probe",
                "status": "success",
                "production_capture": False,
                "savings_vs_shared_baseline": None,
                "device": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "seed": seed,
                "completed_shape_switches": switches,
                "trace": asdict(trace),
                "plan": asdict(plan),
                "backing_allocation_bytes": backend.owner.raw.numel(),
                "external_output_bytes": sum(
                    value.numel() * value.element_size()
                    for values in external.values()
                    for value in values.values()
                ),
                "validation": "exact values, poisoned scratch, retained outputs",
            }
        finally:
            if family is not None:
                family.retire()
            elif transaction is not None:
                transaction.cleanup()
    assert backend.owner is not None
    result["retired"] = backend.owner.released
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--switches", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.switches < 1:
        parser.error("--switches must be positive")
    # Reserve the artifact path before launching any GPU work.
    with args.output.open("x") as output:
        try:
            result = run_probe(
                device=args.device, switches=args.switches, seed=args.seed
            )
        except Exception as error:
            result = {
                "schema_version": m.SCHEMA_VERSION,
                "artifact_kind": "explicit_cuda_binding_probe",
                "status": "failed",
                "reason": getattr(error, "reason", type(error).__name__),
                "detail": str(error),
            }
            json.dump(result, output, indent=2)
            output.write("\n")
            raise
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
