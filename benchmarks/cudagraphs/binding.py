# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental explicit CUDA bindings; never intercept arbitrary allocations.

Callbacks must attest their complete accesses and use only the supplied stream.
Request checks verify these declarations, not hidden kernel accesses. This is a
Stage 2 probe backend, not a production allocation contract or savings claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from benchmarks.cudagraphs.cpu import memory, planner

if TYPE_CHECKING:
    import torch


def access_fingerprint(member, allocation_id: str) -> str:
    """Identify the complete declared root/alias access contract."""
    aliases = tuple(a for a in member.aliases if a.root_id == allocation_id)
    refs = {allocation_id, *(a.alias_id for a in aliases)}
    return memory.digest(
        {
            "allocation": allocation_id,
            "aliases": [asdict(a) for a in aliases],
            "accesses": [asdict(a) for a in member.accesses if a.allocation_id in refs],
            "dependencies": [asdict(d) for d in member.dependencies],
            "coverage": asdict(member.coverage),
        }
    )


class RequestValidator:
    """Validate explicit requests and a serial operation schedule on CPU."""

    def __init__(self, member):
        self.member = member
        self.allocations = {a.allocation_id: a for a in member.allocations}
        self.aliases = {a.alias_id: a for a in member.aliases}
        self.requested: set[str] = set()
        self.requested_aliases: set[str] = set()
        self.operations: list[str] = []

    @staticmethod
    def reject(detail: str) -> None:
        raise memory.UnionMemoryError("capture_contract_mismatch", detail)

    def request(self, allocation, fingerprint: str) -> None:
        key = allocation.allocation_id
        if key in self.requested or self.allocations.get(key) != allocation:
            self.reject(f"unexpected, repeated or changed request: {key}")
        if fingerprint != access_fingerprint(self.member, key):
            self.reject(f"changed access contract: {key}")
        self.requested.add(key)

    def alias(self, alias) -> None:
        if (
            alias.alias_id in self.requested_aliases
            or self.aliases.get(alias.alias_id) != alias
            or alias.root_id not in self.requested
        ):
            self.reject(f"unexpected or changed alias: {alias.alias_id}")
        self.requested_aliases.add(alias.alias_id)

    def operation(self, name: str, accesses: tuple) -> None:
        index = len(self.operations)
        expected = tuple(a for a in self.member.accesses if a.operation == name)
        if (
            index >= len(self.member.operations)
            or self.member.operations[index] != name
            or accesses != expected
            or any(
                a.allocation_id not in self.requested | self.requested_aliases
                for a in accesses
            )
        ):
            self.reject(f"unexpected operation or accesses: {name}")
        self.operations.append(name)

    def finish(self) -> None:
        if (
            self.requested != self.allocations.keys()
            or self.requested_aliases != self.aliases.keys()
            or tuple(self.operations) != self.member.operations
        ):
            self.reject("missing allocation, alias or operation")


def _tensor_identity(tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


class CudaArenaOwner:
    def __init__(self, trace, plan, stream, external):
        self.trace = trace
        self.plan = plan
        self.stream = stream
        self.external = external
        self.raw = None
        self.arena = None
        self.graphs: list[BoundGraph] = []
        self.views: list[Any] = []
        self.drained = False
        self.destroyed = False
        self.released = False

    @property
    def generation(self):
        return memory.StorageGeneration(
            configuration=self.trace.fingerprint,
            persistent_storage=memory.digest(
                [
                    (member, key, _tensor_identity(value))
                    for member, values in sorted(self.external.items())
                    for key, value in sorted(values.items())
                ]
            ),
            arena=memory.digest(
                None if self.arena is None else _tensor_identity(self.arena)
            ),
            execution_lane=str(self.stream.cuda_stream),
        )

    def check_stream(self) -> None:
        import torch

        if torch.cuda.current_stream(self.stream.device) != self.stream:
            raise memory.UnionMemoryError("unknown_lifetime", "wrong execution stream")


class BindingSession:
    def __init__(self, owner: CudaArenaOwner, member):
        self.owner = owner
        self.member = member
        self.validator = RequestValidator(member)
        self.views: dict[str, Any] = {}
        self.placements = {
            p.allocation_id: p
            for p in owner.plan.placements
            if p.member_id == member.member_id
        }

    def request(self, allocation, shape: tuple[int, ...], dtype, fingerprint: str):
        """Bind one declared contiguous tensor; every site must call this."""
        self.owner.check_stream()
        self.validator.request(allocation, fingerprint)
        if (
            any(type(dim) is not int or dim < 0 for dim in shape)
            or prod(shape) * dtype.itemsize != allocation.requested_bytes
        ):
            self.validator.reject("request shape/dtype byte size changed")
        key = allocation.allocation_id
        if key in self.placements:
            placement = self.placements[key]
            assert self.owner.arena is not None
            result = (
                self.owner.arena[
                    placement.offset : placement.offset + allocation.requested_bytes
                ]
                .view(dtype)
                .reshape(shape)
            )
        else:
            result = self.owner.external[self.member.member_id][key]
            if tuple(result.shape) != shape or result.dtype != dtype:
                self.validator.reject("external tensor layout changed")
        if result.numel() and result.data_ptr() % allocation.alignment:
            self.validator.reject("unaligned binding")
        self.views[key] = result
        self.owner.views.append(result)
        return result

    def alias(self, alias, dtype):
        import torch

        self.owner.check_stream()
        self.validator.alias(alias)
        if (
            str(dtype).removeprefix("torch.") != alias.dtype
            or dtype.itemsize != alias.itemsize
        ):
            self.validator.reject("alias dtype changed")
        root = self.views[alias.root_id].view(-1).view(torch.uint8)
        result = root[alias.offset : alias.offset + alias.span].view(dtype)
        result = result.reshape(alias.shape)
        if tuple(result.stride()) != alias.strides or (
            result.numel() and result.data_ptr() % alias.alignment
        ):
            self.validator.reject("alias strides/alignment changed")
        self.views[alias.alias_id] = result
        self.owner.views.append(result)
        return result

    @contextmanager
    def operation(self, name: str, accesses: tuple):
        self.owner.check_stream()
        self.validator.operation(name, accesses)
        yield
        self.owner.check_stream()


class BoundGraph:
    def __init__(self, owner: CudaArenaOwner, handle, outputs):
        self.owner = owner
        self.handle = handle
        self.outputs = MappingProxyType(dict(outputs))
        self.generation = owner.generation

    def replay(self):
        self.owner.check_stream()
        if self.handle is None or self.owner.generation != self.generation:
            raise memory.UnionMemoryError("stale_generation", "binding storage changed")
        self.owner.drained = False
        self.handle.replay()
        return self.outputs


class ExplicitCudaBackend:
    """Single-stream prototype with caller-owned external outputs.

    Programs must declare all accesses, retain no scratch, mutate no external
    state other than outputs, and launch no work on other streams. Validators
    receive CPU copies of external outputs after completion. Only this narrow
    explicit protocol is supported.
    """

    def __init__(
        self,
        trace,
        programs: Mapping[str, Callable[[BindingSession], None]],
        external: Mapping[str, Mapping[str, torch.Tensor]],
        validators: Mapping[str, Callable[[Mapping[str, torch.Tensor]], None]],
        stream,
    ):
        self.trace = trace
        self.programs = dict(programs)
        self.external = {key: dict(value) for key, value in external.items()}
        self.validators = dict(validators)
        self.stream = stream
        self.owner: CudaArenaOwner | None = None
        members = {m.member_id for m in trace.members}
        if any(set(mapping) != members for mapping in (programs, external, validators)):
            RequestValidator.reject("missing or extra member binding")
        for member in trace.members:
            if member.escapes or any(
                dep.evidence != "stream_order" for dep in member.dependencies
            ):
                raise memory.UnionMemoryError("unknown_lifetime", "not a serial probe")
            positions = {name: i for i, name in enumerate(member.operations)}
            if any(
                positions[d.source] >= positions[d.target] for d in member.dependencies
            ):
                RequestValidator.reject("operation schedule violates dependencies")
            roots = {a.allocation_id: a for a in member.allocations}
            if any(
                alias.offset + alias.span > roots[alias.root_id].requested_bytes
                for alias in member.aliases
            ):
                RequestValidator.reject("alias references reservation slack")

    def reserve(self, plan) -> CudaArenaOwner:
        import torch

        planner.verify(self.trace, plan)
        if self.owner is not None:
            raise RuntimeError("Backend instances are single-use")
        if torch.version.hip is not None or not torch.cuda.is_available():
            raise memory.UnionMemoryError("unsupported_allocator", "CUDA unavailable")
        owner = CudaArenaOwner(self.trace, plan, self.stream, self.external)
        owner.check_stream()
        output_storage = set()
        for member in self.trace.members:
            expected = {
                p.allocation_id
                for p in plan.exclusions
                if p.member_id == member.member_id
            }
            external = self.external[member.member_id]
            if external.keys() != expected:
                RequestValidator.reject("external allocations do not match exclusions")
            for allocation in member.allocations:
                if allocation.allocation_id not in external:
                    continue
                value = external[allocation.allocation_id]
                if (
                    allocation.storage_class != "output"
                    or not value.is_contiguous()
                    or value.device != self.stream.device
                    or value.numel() * value.element_size()
                    != allocation.requested_bytes
                    or allocation.requested_bytes != allocation.reserved_bytes
                ):
                    RequestValidator.reject("invalid external output")
                storage = value.untyped_storage().data_ptr()
                if storage in output_storage:
                    RequestValidator.reject("external outputs share storage")
                output_storage.add(storage)
        raw = torch.empty(
            plan.arena_bytes + plan.arena_alignment - 1 if plan.arena_bytes else 0,
            dtype=torch.uint8,
            device=self.stream.device,
        )
        padding = -raw.data_ptr() % plan.arena_alignment
        owner.raw = raw
        owner.arena = raw[padding : padding + plan.arena_bytes]
        self.owner = owner
        return owner

    def capture(self, owner: CudaArenaOwner, member, plan):
        import torch

        if (
            owner is not self.owner
            or plan != owner.plan
            or member not in self.trace.members
        ):
            RequestValidator.reject("capture member/plan changed")
        owner.check_stream()
        for _ in range(2):
            warmup = BindingSession(owner, member)
            self.programs[member.member_id](warmup)
            warmup.validator.finish()
        owner.stream.synchronize()
        session = BindingSession(owner, member)
        outputs = owner.external[member.member_id]
        graph = BoundGraph(owner, torch.cuda.CUDAGraph(), outputs)
        # Retain even a partially captured handle until transaction cleanup.
        owner.graphs.append(graph)
        with torch.cuda.graph(graph.handle, stream=owner.stream):
            self.programs[member.member_id](session)
            session.validator.finish()
        return graph

    def validate(self, owner: CudaArenaOwner, graphs) -> None:
        owner.check_stream()
        assert owner.arena is not None
        if set(graphs) != set(self.programs):
            RequestValidator.reject("partial capture family")
        sequence = list(graphs) + list(reversed(graphs)) + list(graphs)
        for index, key in enumerate(sequence):
            owner.arena.fill_(0xA5 if index % 2 else 0x5A)
            for output in graphs[key].outputs.values():
                output.fill_(0xA5 if index % 2 else 0x5A)
            graphs[key].replay()
            owner.stream.synchronize()
            self.validators[key](
                {name: tensor.cpu() for name, tensor in graphs[key].outputs.items()}
            )
        # Each member has its own external outputs, which must survive switches.
        for key, graph in graphs.items():
            self.validators[key](
                {name: tensor.cpu() for name, tensor in graph.outputs.items()}
            )

    def drain(self, owner: CudaArenaOwner) -> None:
        owner.stream.synchronize()
        owner.drained = True

    def destroy(self, owner: CudaArenaOwner) -> None:
        if not owner.drained:
            raise RuntimeError("Drain must complete before graph destruction")
        for graph in owner.graphs:
            if graph.handle is not None:
                graph.handle.reset()
                graph.handle = None
        owner.graphs.clear()
        owner.views.clear()
        owner.destroyed = True

    def release(self, owner: CudaArenaOwner) -> None:
        if not owner.destroyed:
            raise RuntimeError("Destroy graphs and views before releasing the arena")
        owner.arena = None
        owner.raw = None
        owner.released = True
