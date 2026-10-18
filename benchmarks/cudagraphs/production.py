# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic adapter around the unchanged production V1 capture/dispatch path."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict
from unittest.mock import patch

from benchmarks.cudagraphs.collect import (
    Unsupported,
    request_sequence,
    select_descriptors,
)
from benchmarks.cudagraphs.cpu import memory
from benchmarks.cudagraphs.schema import (
    ByteMeasurement,
    Header,
    MemoryCategories,
    RunRecord,
    decode,
)


def native_descriptor(mode, descriptor) -> dict:
    value = memory.Descriptor(
        "v1", tuple(sorted({"runtime_mode": mode.name, **asdict(descriptor)}.items()))
    )
    value.validate()
    # JSON normalization preserves exact equality with saved standalone requests.
    return json.loads(json.dumps(asdict(value)))


def kv_storage_inventory(config, caches, device) -> dict:
    """Reconcile live KV views with the single-backing CUDA allocation contract.

    Counts tensor storage bytes, including layout padding, once per backing.
    Allocator reservation slack and driver allocations are outside this count.
    """
    import torch

    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout

    placements = config.kv_cache_tensors
    groups = config.kv_cache_groups
    if (
        not placements
        or not groups
        or config.num_blocks <= 0
        or config.kv_cache_layout not in KVCacheLayout.__members__
        or any(p.host_resident for p in placements)
        or any(
            g.host_resident or not isinstance(g.kv_cache_spec, FullAttentionSpec)
            for g in groups
        )
    ):
        raise Unsupported(
            "unsupported_kv_layout", "Expected device-only full-attention KV views"
        )
    # allocate_kv_cache gives every placement the size of the same backing.
    sizes = {p.size for p in placements}
    if len(sizes) != 1 or next(iter(sizes)) <= 0:
        raise Unsupported(
            "unsupported_kv_layout", "Expected one configured KV backing size"
        )
    configured_bytes = next(iter(sizes))
    layer_names = [name for group in groups for name in group.layer_names]
    placed_names = [name for placement in placements for name in placement.layers]
    if (
        len(layer_names) != len(set(layer_names))
        or len(placed_names) != len(set(placed_names))
        or set(layer_names) != set(placed_names)
        or len(caches) != len(layer_names)
    ):
        raise Unsupported(
            "unsupported_kv_layout", "KV layer coverage does not match configured views"
        )

    storages: dict[tuple[str, int], dict] = {}
    views = []
    device = torch.device(device)
    for index, cache in enumerate(caches):
        if (
            not isinstance(cache, torch.Tensor)
            or cache.device != device
            or cache.layout != torch.strided
            or cache.numel() == 0
            or cache.ndim != 4
            or cache.shape[0] != config.num_blocks
        ):
            raise Unsupported(
                "unsupported_kv_layout",
                f"KV view {index} must be a strided [B,H,N,C] tensor on {device}",
            )
        storage = cache.untyped_storage()
        key = (str(cache.device), storage.data_ptr())
        if key not in storages:
            storages[key] = {
                "storage_id": len(storages),
                "device": key[0],
                "data_ptr": key[1],
                "bytes": storage.nbytes(),
            }
        elif storages[key]["bytes"] != storage.nbytes():
            raise Unsupported("kv_storage_mismatch", "Inconsistent KV storage extent")
        views.append(
            {
                "runner_index": index,
                "storage_id": storages[key]["storage_id"],
                "shape": list(cache.shape),
                "stride_bytes": [s * cache.element_size() for s in cache.stride()],
                "offset_bytes": cache.storage_offset() * cache.element_size(),
                "dtype": str(cache.dtype),
            }
        )
    actual_bytes = sum(s["bytes"] for s in storages.values())
    if len(storages) != 1 or actual_bytes != configured_bytes:
        raise Unsupported(
            "kv_storage_mismatch",
            f"Expected one {configured_bytes}-byte KV backing; observed "
            f"{len(storages)} backings totaling {actual_bytes} bytes",
        )
    return {
        "bytes": actual_bytes,
        "configured_backing_bytes": configured_bytes,
        "measurement": "deduplicated live untyped storage; excludes allocator slack",
        "num_blocks": config.num_blocks,
        "groups": [list(group.layer_names) for group in groups],
        "layout": config.kv_cache_layout,
        "group_details": [
            {
                "group_id": index,
                "layers": list(group.layer_names),
                "num_blocks": config.num_blocks,
                "block_size_tokens": group.kv_cache_spec.block_size,
                "page_size_bytes": group.kv_cache_spec.page_size_bytes,
                "layout": config.kv_cache_layout,
            }
            for index, group in enumerate(groups)
        ],
        "placements": [asdict(p) for p in placements],
        "storages": list(storages.values()),
        "views": views,
    }


class DeviceSampler:
    """NVML samples share the worker clock and never synchronize GPU execution."""

    def __init__(self, artifacts, device_uuid: str, interval_ms: int):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        if not device_uuid.startswith(("GPU-", "MIG-")):
            device_uuid = "GPU-" + device_uuid
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
        except Exception:
            pynvml.nvmlShutdown()
            raise
        self.artifacts = artifacts
        self.interval = interval_ms / 1000
        self.stop_event = threading.Event()
        self.foreign_processes: set[int] = set()
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self) -> dict:
        processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        foreign = [p.pid for p in processes if p.pid != os.getpid()]
        self.foreign_processes.update(foreign)
        usage = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        own = [p.usedGpuMemory for p in processes if p.pid == os.getpid()]
        # NVML may return an unavailable sentinel rather than an integer count.
        process_bytes = (
            sum(own) if all(0 <= size <= usage.total for size in own) else None
        )
        return {
            "device_used_bytes": usage.used,
            "device_total_bytes": usage.total,
            "process_used_bytes": process_bytes,
            "process_counter_reason": None
            if process_bytes is not None
            else "NVML unavailable",
            "foreign_pids": foreign,
            "gpu_utilization_percent": self.nvml.nvmlDeviceGetUtilizationRates(
                self.handle
            ).gpu,
            "driver_version": self.nvml.nvmlSystemGetDriverVersion(),
        }

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.artifacts.event("startup_sample", **self.sample())
            except self.nvml.NVMLError as error:
                self.errors.append(str(error))
                self.artifacts.event(
                    "startup_sample", status="failure", reason=str(error)
                )
                return
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        initial = self.sample()
        self.artifacts.event("device_preflight", **initial)
        if self.foreign_processes:
            raise Unsupported(
                "gpu_contention",
                f"GPU has other processes: {sorted(self.foreign_processes)}",
            )
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.nvml.nvmlShutdown()


class CaptureObserver:
    def __init__(self, job, artifacts):
        self.job = job
        self.artifacts = artifacts
        self.runner = None
        self.resolved: list[dict] = []
        self.selected: list[dict] = []
        self.entries: list[dict] = []
        self.dispatches: list[dict] = []
        self.wrappers = []
        self.pool_snapshots: dict[str, list[dict]] = {}
        self.capture_seen = False

    def snapshot(self, phase: str, **data) -> None:
        import torch

        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info()
        counters = torch.accelerator.memory_stats()
        self.artifacts.event(
            phase,
            device="cuda:0",
            consumer_completion="device synchronized diagnostic boundary",
            allocator_allocated_bytes=counters.get("allocated_bytes.all.current"),
            allocator_reserved_bytes=counters.get("reserved_bytes.all.current"),
            allocator_peak_reserved_bytes=counters.get("reserved_bytes.all.peak"),
            device_free_bytes=free,
            device_total_bytes=total,
            **data,
        )
        # Allocator segments are diagnostics, not proof of GPU liveness.
        segments = torch.cuda.memory_snapshot()
        self.pool_snapshots[phase] = segments
        self.artifacts.write(
            f"snapshot-{phase}.json",
            {
                "status": "success",
                "segments": segments,
                "coverage_gaps": ["No kernel accesses, aliases or GPU lifetime proof"],
            },
        )

    def inventory(self) -> list[dict]:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        entries = []
        self.wrappers = []
        selected_ids = {decode(memory.Descriptor, d).member_id for d in self.selected}
        for wrapper in list(CUDAGraphWrapper._all_instances):
            found = False
            for descriptor, entry in wrapper.concrete_cudagraph_entries.items():
                value = native_descriptor(wrapper.runtime_mode, descriptor)
                member_id = decode(memory.Descriptor, value).member_id
                if member_id in selected_ids and entry.cudagraph is not None:
                    found = True
                    entries.append(
                        {
                            "descriptor": value,
                            "member_id": member_id,
                            "graph_id": str(id(entry.cudagraph)),
                            "pool_id": list(entry.cudagraph.pool()),
                        }
                    )
            if found:
                self.wrappers.append(wrapper)
        return entries

    def capture(self, runner, original):
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.platforms import current_platform

        if self.capture_seen:
            raise Unsupported(
                "capture_contract_mismatch", "More than one production capture"
            )
        self.capture_seen = True
        self.runner = runner
        model = runner.model_config
        if model.is_multimodal_model or model.is_moe or model.is_hybrid:
            raise Unsupported(
                "unsupported_feature",
                "Collector initially requires a dense text decoder",
            )
        groups = runner.cudagraph_dispatcher.get_capture_descs()
        self.resolved = [
            native_descriptor(mode, desc) for mode, descs in groups for desc in descs
        ]
        self.selected = select_descriptors(
            self.resolved, self.job["variant"], self.job["selected_descriptor"]
        )
        if any(mode != CUDAGraphMode.FULL for mode, _ in groups):
            raise Unsupported(
                "unsupported_feature", "Resolved capture includes non-FULL graphs"
            )
        selected_groups = [
            (
                mode,
                [
                    desc
                    for desc in descs
                    if native_descriptor(mode, desc) in self.selected
                ],
            )
            for mode, descs in groups
        ]
        warmup_and_capture = runner._warmup_and_capture
        count = 0

        def observe_member(desc, cudagraph_runtime_mode, *args, **kwargs):
            nonlocal count
            mode = cudagraph_runtime_mode
            value = native_descriptor(mode, desc)
            self.snapshot(f"member-{count}-before", descriptor=value)
            started = time.monotonic_ns()
            with ExitStack() as stack:
                if self.job["variant"] == "separate_pools":
                    pool = current_platform.graph_pool_handle()
                    for wrapper in list(CUDAGraphWrapper._all_instances):
                        stack.enter_context(patch.object(wrapper, "graph_pool", pool))
                warmup_and_capture(desc, mode, *args, **kwargs)
            self.snapshot(
                f"member-{count}-after",
                descriptor=value,
                warmup_capture_wall_ns=time.monotonic_ns() - started,
            )
            count += 1

        self.snapshot("capture-before")
        started = time.monotonic_ns()
        with (
            patch.object(runner, "_warmup_and_capture", observe_member),
            patch.object(
                runner.cudagraph_dispatcher,
                "get_capture_descs",
                return_value=selected_groups,
            ),
        ):
            result = original(runner)
        self.entries = self.inventory()
        ids = {entry["member_id"] for entry in self.entries}
        expected = {decode(memory.Descriptor, d).member_id for d in self.selected}
        if ids != expected:
            raise Unsupported(
                "capture_contract_mismatch",
                "Captured members differ from selected descriptors",
            )
        if self.job["variant"] == "separate_pools":
            pools: dict[str, set[tuple]] = {}
            for entry in self.entries:
                pools.setdefault(entry["member_id"], set()).add(tuple(entry["pool_id"]))
            if any(len(value) != 1 for value in pools.values()) or len(
                set.union(*pools.values())
            ) != len(pools):
                raise Unsupported(
                    "capture_contract_mismatch",
                    "Separate-pool control did not isolate members",
                )
        self.snapshot(
            "capture-after",
            graphs=self.entries,
            capture_wall_ns=time.monotonic_ns() - started,
        )
        return result

    def replay(self) -> None:
        import torch

        runner = self.runner
        assert runner is not None
        tokens = sorted(dict(d["fields"])["num_tokens"] for d in self.selected)
        sequence = tokens + request_sequence(
            tokens, self.job["stress_steps"], self.job["seed"]
        )
        if len(self.selected) == 1:
            # Reduced coverage cannot service arbitrary padded requests.
            sequence = [tokens[0]] * len(sequence)
        determine = runner._determine_batch_execution_and_padding
        current_request = 0

        def observe_dispatch(*args, **kwargs):
            result = determine(*args, **kwargs)
            mode, desc = result[:2]
            selected = native_descriptor(mode, desc)
            if selected not in self.selected:
                raise Unsupported(
                    "dispatch_mismatch", "Replay selected an uncaptured descriptor"
                )
            self.dispatches.append(
                {"request_tokens": current_request, "descriptor": selected}
            )
            return result

        with patch.object(
            runner, "_determine_batch_execution_and_padding", observe_dispatch
        ):
            for index, current_request in enumerate(sequence):
                before = len(self.dispatches)
                started = time.monotonic_ns()
                outputs = runner._dummy_run(
                    current_request,
                    uniform_decode=True,
                    allow_microbatching=False,
                    skip_eplb=True,
                    remove_lora=False,
                )
                # Diagnostic endpoint excludes normal serving consumers/sampling.
                torch.accelerator.synchronize()
                del outputs
                if len(self.dispatches) != before + 1:
                    raise Unsupported(
                        "dispatch_mismatch", "Expected one dispatch per dummy step"
                    )
                self.artifacts.event(
                    "first_replay" if index < len(tokens) else "shape_switch",
                    sequence_id=index,
                    **self.dispatches[-1],
                    dummy_step_wall_ns=time.monotonic_ns() - started,
                    latency_method=(
                        "diagnostic synchronized dummy step; not serving latency"
                    ),
                )
                if index == len(tokens) - 1:
                    self.snapshot("all-members-replayed")
                elif (index + 1) % 1000 == 0:
                    self.snapshot(f"stress-{index + 1}")
        self.snapshot("steady-resident")

    def retire(self) -> None:
        import torch

        self.snapshot("teardown-before")
        torch.accelerator.synchronize()
        # Only wrappers captured by this adapter; no global clear_all_graphs.
        for wrapper in self.wrappers:
            wrapper.clear_graphs()
        self.wrappers.clear()
        self.snapshot("teardown-after")


def collect(job: dict, artifacts, manifest: dict) -> RunRecord:
    import torch

    import vllm
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    manifest["cuda_runtime"] = torch.version.cuda
    if not torch.cuda.is_available():
        raise Unsupported(
            "cuda_unavailable",
            "PyTorch cannot initialize CUDA; check driver/runtime and device access",
        )
    properties = torch.cuda.get_device_properties(0)
    manifest["gpu"] = {
        "name": properties.name,
        "uuid": str(properties.uuid),
        "capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }
    observer = CaptureObserver(job, artifacts)
    sampler = DeviceSampler(artifacts, str(properties.uuid), job["sample_interval_ms"])
    llm = None
    try:
        sampler.start()
        observer.snapshot("startup-before")
        original = GPUModelRunner.capture_model
        with patch.object(
            GPUModelRunner,
            "capture_model",
            lambda runner: observer.capture(runner, original),
        ):
            llm = vllm.LLM(**job["engine_options"])
        if observer.runner is None:
            raise Unsupported(
                "unsupported_runner", "Production V1 capture was not observed"
            )
        manifest["resolved_descriptors"] = observer.resolved
        manifest["captured_descriptors"] = observer.selected
        manifest["graph_executables"] = observer.entries
        config = observer.runner.kv_cache_config
        manifest["actual_kv"] = kv_storage_inventory(
            config, observer.runner.kv_caches, torch.device("cuda", 0)
        )
        actual_kv_bytes = manifest["actual_kv"]["bytes"]
        artifacts.event("kv-storage-verified", **manifest["actual_kv"])
        manifest["startup_graph_estimator"] = "skipped: explicit kv_cache_memory_bytes"
        observer.replay()
        observer.retire()
        unknown = ByteMeasurement(
            None, None, "Raw allocator segments require ownership/access attribution"
        )
        manifest["status"] = "success"
        manifest["correctness"] = "pending: diagnostic dummy inputs only"
        record = RunRecord(
            Header(
                1,
                job["run_id"],
                job["pair_id"],
                job["variant"],
                "diagnostic",
                "success",
            ),
            memory.digest(
                {
                    "engine": job["engine_options"],
                    "capture_policy": job["capture_sizes"],
                    "revision": manifest["revision"],
                    "dirty_diff_digest": manifest["dirty_diff_digest"],
                    "environment": manifest["environment"],
                    "packages": manifest["packages"],
                    "cuda_runtime": manifest["cuda_runtime"],
                    "gpu": manifest["gpu"],
                    "sample_interval_ms": job["sample_interval_ms"],
                    "stress_steps": job["stress_steps"],
                }
            ),
            memory.digest(manifest),
            "v1",
            tuple(decode(memory.Descriptor, d) for d in observer.selected),
            len(observer.entries),
            actual_kv_bytes,
            (config.num_blocks,) * len(config.kv_cache_groups),
            memory.digest(observer.dispatches),
            (),
            MemoryCategories(unknown, unknown, unknown, unknown),
            (),
            "pending",
        )
        record.validate()
        artifacts.write(
            "allocations.json",
            {
                "status": "unsupported",
                "reason": "allocator snapshots do not prove kernel access lifetimes",
            },
        )
    finally:
        try:
            sampler.close()
        finally:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
    if sampler.foreign_processes or sampler.errors:
        raise Unsupported(
            "gpu_contention_or_counter_failure",
            str(sampler.foreign_processes or sampler.errors),
        )
    return record
