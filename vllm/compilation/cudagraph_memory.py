# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for experimental CUDA graph union memory.

These records describe backend-proven GPU accesses, not allocator timestamps.
No production allocation backend is enabled by importing this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal, Protocol

SCHEMA_VERSION = 1
PLANNER_VERSION = 1
MAX_BYTES = (1 << 63) - 1


class UnionMemoryError(ValueError):
    """A recoverable contract rejection before publication."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class Descriptor:
    """Full native descriptor; fields must match the declared runner schema."""

    runner: Literal["v1", "v2"]
    fields: tuple[tuple[str, int | bool | str | None], ...]
    schema_version: int = SCHEMA_VERSION

    @property
    def member_id(self) -> str:
        return digest(asdict(self))

    def validate(self) -> None:
        expected = {
            "v1": {
                "runtime_mode",
                "num_tokens",
                "num_reqs",
                "uniform",
                "has_lora",
                "num_active_loras",
            },
            "v2": {
                "cg_mode",
                "num_tokens",
                "num_reqs",
                "uniform_token_count",
                "max_query_len",
                "num_active_loras",
                "num_ubatches",
            },
        }
        if self.schema_version != SCHEMA_VERSION or self.runner not in expected:
            raise UnionMemoryError("unsupported_runner", str(self))
        names = [key for key, _ in self.fields]
        if names != sorted(expected[self.runner]):
            raise UnionMemoryError("unsupported_runner", "descriptor schema changed")
        values = dict(self.fields)
        for key, value in self.fields:
            if key in ("runtime_mode", "cg_mode"):
                if value not in ("FULL", "PIECEWISE"):
                    raise UnionMemoryError("unsupported_runner", "invalid runtime mode")
            elif key in ("uniform", "has_lora"):
                if type(value) is not bool:
                    raise UnionMemoryError("unsupported_runner", key)
            elif value is None:
                if key not in ("num_reqs", "uniform_token_count", "max_query_len"):
                    raise UnionMemoryError("unsupported_runner", key)
            elif type(value) is not int or value < 0:
                raise UnionMemoryError("unsupported_runner", key)
        if type(values["num_tokens"]) is not int or values["num_tokens"] <= 0:
            raise UnionMemoryError("unsupported_runner", "invalid token count")
        if self.runner == "v2" and values["num_ubatches"] == 0:
            raise UnionMemoryError("unsupported_runner", "invalid microbatch count")


@dataclass(frozen=True)
class FamilyIdentity:
    """Portable configuration identity; pointers belong in runtime generations."""

    model_revision: str
    config_digest: str
    tensor_layout_digest: str
    kv_layout_digest: str
    backend: str
    allocator: str
    dtype: str
    compilation_digest: str
    software_digest: str
    device_capabilities: str
    storage_backend_version: str
    device: str
    execution_lane: str
    exclusive_members: bool


@dataclass(frozen=True)
class AllocationSpec:
    allocation_id: str
    site_id: str
    occurrence: int
    requested_bytes: int
    reserved_bytes: int
    alignment: int
    owner: str
    storage_class: Literal["scratch", "persistent", "output", "uncontrolled"]
    binding_id: str | None = None


@dataclass(frozen=True)
class AliasSpec:
    alias_id: str
    root_id: str
    offset: int
    span: int
    alignment: int
    dtype: str
    itemsize: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]


@dataclass(frozen=True)
class AccessSpec:
    allocation_id: str
    operation: str
    offset: int
    size: int
    kind: Literal["read", "write", "read_write"]


@dataclass(frozen=True)
class DependencySpec:
    source: str
    target: str
    evidence: Literal["stream_order", "graph_edge", "event_wait", "owner_contract"]


@dataclass(frozen=True)
class EscapeSpec:
    allocation_id: str
    owner: str
    completion: str
    reason: str


@dataclass(frozen=True)
class TraceCoverage:
    """A backend attests completeness; the planner cannot infer kernel accesses."""

    declared_sites: tuple[str, ...]
    observed_sites: tuple[str, ...]
    access_contract: str
    alias_contract: str
    discovery_method: str
    gaps: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemberTrace:
    descriptor: Descriptor
    operations: tuple[str, ...]
    dependencies: tuple[DependencySpec, ...]
    allocations: tuple[AllocationSpec, ...]
    aliases: tuple[AliasSpec, ...]
    accesses: tuple[AccessSpec, ...]
    escapes: tuple[EscapeSpec, ...]
    coverage: TraceCoverage

    @property
    def member_id(self) -> str:
        return self.descriptor.member_id


@dataclass(frozen=True)
class FamilyTrace:
    identity: FamilyIdentity
    members: tuple[MemberTrace, ...]
    schema_version: int = SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        return digest({"trace": asdict(self), "planner_version": PLANNER_VERSION})


@dataclass(frozen=True)
class Placement:
    member_id: str
    allocation_id: str
    offset: int
    reserved_bytes: int


@dataclass(frozen=True)
class Exclusion:
    member_id: str
    allocation_id: str
    reserved_bytes: int
    reason: str


@dataclass(frozen=True)
class MemoryPlan:
    fingerprint: str
    placements: tuple[Placement, ...]
    exclusions: tuple[Exclusion, ...]
    arena_bytes: int
    arena_alignment: int
    planner_version: int = PLANNER_VERSION


@dataclass(frozen=True)
class StorageGeneration:
    """Process-local generations checked before every launch."""

    configuration: str
    persistent_storage: str
    arena: str
    execution_lane: str


class CapturedGraph(Protocol):
    def replay(self) -> object: ...


class ArenaOwner(Protocol):
    """Retains storage and dependent views until retirement has completed."""

    @property
    def generation(self) -> StorageGeneration: ...


class UnionMemoryBackend(Protocol):
    """Backend obligations at the transaction boundary.

    Capture checks every request (including missing/extra sites, aliases, sizes,
    access contracts and pointers), preserves external allocations, and restores
    capture scopes/flags on failure. Failed capture must retain partial handles
    on the owner until drain/destroy. Validation includes controlled replay and
    restoration of KV/RNG/counters. Drain waits for all accesses and consumers;
    destroy removes all handles/views before release frees backing storage.
    Only proven recoverable failures use UnionMemoryError; device faults escape.
    Reserve must clean its own partial resources if it raises before returning.
    """

    def reserve(self, plan: MemoryPlan) -> ArenaOwner: ...

    def capture(
        self, owner: ArenaOwner, member: MemberTrace, plan: MemoryPlan
    ) -> CapturedGraph: ...

    def validate(
        self, owner: ArenaOwner, graphs: Mapping[str, CapturedGraph]
    ) -> None: ...

    def drain(self, owner: ArenaOwner) -> None: ...

    def destroy(self, owner: ArenaOwner) -> None: ...

    def release(self, owner: ArenaOwner) -> None: ...


class PublishedGraphFamily:
    """Immutable lookup with strong ownership and explicit, retryable retirement.

    The runner must serialize replay and retirement on its declared lane. It must
    retain this object until retire succeeds; there is no optimistic finalizer.
    """

    def __init__(
        self,
        backend: UnionMemoryBackend,
        owner: ArenaOwner,
        graphs: Mapping[str, CapturedGraph],
    ) -> None:
        self._backend = backend
        self._owner = owner
        self._generation = owner.generation
        self._entries = dict(graphs)
        self._graphs = MappingProxyType(self._entries)
        self._state = "ready"

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(self._graphs)

    def replay(self, member_id: str, generation: StorageGeneration) -> object:
        if (
            self._state != "ready"
            or generation != self._generation
            or self._owner.generation != self._generation
        ):
            raise UnionMemoryError("stale_generation", "reinitialization required")
        if member_id not in self._graphs:
            raise UnionMemoryError("capture_contract_mismatch", "unknown descriptor")
        return self._graphs[member_id].replay()

    def retire(self) -> None:
        if self._state == "retired":
            return
        if self._state in ("ready", "draining"):
            self._state = "draining"
            self._backend.drain(self._owner)
            self._state = "destroying"
        if self._state == "destroying":
            self._backend.destroy(self._owner)
            self._entries.clear()
            self._state = "releasing"
        self._backend.release(self._owner)
        self._state = "retired"


class CaptureTransaction:
    """Stage a complete family; retain failed resources until safe cleanup.

    Keep the transaction alive if cleanup fails. Backend destroy/release must be
    idempotent so retirement can be retried. Unknown exceptions never fall back.
    """

    def __init__(self, backend: UnionMemoryBackend) -> None:
        self._backend = backend
        self._pending: PublishedGraphFamily | None = None
        self._used = False

    def cleanup(self) -> None:
        if self._pending is not None:
            self._pending.retire()
            self._pending = None

    def capture(
        self,
        trace: FamilyTrace,
        plan: MemoryPlan,
        fallback: Callable[[str], object],
    ) -> PublishedGraphFamily | object:
        from .cudagraph_memory_planner import verify

        if self._used:
            raise RuntimeError("CaptureTransaction is single-use")
        self._used = True
        try:
            verify(trace, plan)
            owner = self._backend.reserve(plan)
            # The backend also owns partially captured handles on failure.
            self._pending = PublishedGraphFamily(self._backend, owner, {})
            for member in trace.members:
                self._pending._entries[member.member_id] = self._backend.capture(
                    owner, member, plan
                )
            self._backend.validate(owner, self._pending._graphs)
            if owner.generation != self._pending._generation:
                raise UnionMemoryError("stale_generation", "storage changed in capture")
            published = self._pending
        except UnionMemoryError as error:
            self.cleanup()
            return fallback(error.reason)
        except BaseException:
            # A device fault may prevent draining. Retain the owner if that fails.
            self.cleanup()
            raise
        self._pending = None
        return published


@dataclass(frozen=True)
class Eligibility:
    supported: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class RunnerCapabilities:
    runner: str
    platform: str
    graph_mode: str
    device_count: int
    ordered_execution_lane: bool
    allocation_contract: str | None
    validated_backend_dtype: bool
    unsupported_features: tuple[str, ...]


def assess(capabilities: RunnerCapabilities) -> Eligibility:
    """Conservative initial-path gate; names of GPUs/models are never inputs.

    Adapters must enumerate active deferred features, including collectives,
    ubatching, LoRA, speculation, MoE/hybrid state, multimodal, sleep, offload and
    reconfiguration. A contract is issued only after actual binding validation.
    """
    reasons = []
    if capabilities.runner != "v1":
        reasons.append("unsupported_runner")
    if capabilities.platform != "cuda" or not capabilities.allocation_contract:
        reasons.append("unsupported_allocator")
    if (
        capabilities.graph_mode != "FULL_DECODE_ONLY"
        or capabilities.device_count != 1
        or not capabilities.validated_backend_dtype
        or capabilities.unsupported_features
    ):
        reasons.append("unsupported_feature")
    if not capabilities.ordered_execution_lane:
        reasons.append("unknown_lifetime")
    return Eligibility(not reasons, tuple(reasons))
