# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collector controls must preserve native coverage and explicit failure artifacts.

CPU tests exercise config -> selected descriptors/events without loading weights;
actual capture coverage is checked by the adapter on the real manager.
"""

import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from benchmarks.cudagraphs.binding_probe import probe_trace
from benchmarks.cudagraphs.collect import (
    Unsupported,
    request_sequence,
    run_worker,
    select_descriptors,
    validate_engine_options,
)
from benchmarks.cudagraphs.production import kv_storage_inventory


@pytest.fixture
def kv_storage():
    """Real CPU views catch duplicate backing counts without CUDA or model loading."""
    import torch

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        create_kv_cache_views,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout

    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=40,
    )
    blocks = 3
    layer_bytes = blocks * spec.page_size_bytes
    size = 2 * layer_bytes
    names = ["model.layers.0.attn", "model.layers.1.attn"]
    placements = [
        KVCacheTensor(size, [name], layer_bytes, spec.page_size_bytes, i * layer_bytes)
        for i, name in enumerate(names)
    ]
    config = KVCacheConfig(
        blocks,
        placements,
        [KVCacheGroupSpec(names, spec)],
        kv_cache_layout="LBHNC",
    )
    backing = torch.empty(size, dtype=torch.uint8)
    caches = [
        create_kv_cache_views(backing, spec, blocks, KVCacheLayout.LBHNC, p)[0]
        for p in placements
    ]
    return config, caches


def test_kv_bytes_count_shared_backing_once_and_preserve_view_layout(kv_storage):
    """Config + live views -> bytes/layout; aliases must not inflate KV controls."""
    config, caches = kv_storage
    # Padded pages create strided views at different offsets in the same backing.
    assert not caches[1].is_contiguous()
    assert caches[0].data_ptr() != caches[1].data_ptr()
    result = kv_storage_inventory(config, caches, "cpu")
    assert result["bytes"] == config.kv_cache_tensors[0].size
    assert result["configured_backing_bytes"] == result["bytes"]
    assert sum(p.size for p in config.kv_cache_tensors) == 2 * result["bytes"]
    assert len(result["storages"]) == 1
    assert [v["storage_id"] for v in result["views"]] == [0, 0]
    view = result["views"][1]
    assert view["offset_bytes"] == caches[1].storage_offset() * caches[1].element_size()
    assert view["stride_bytes"] == [
        s * caches[1].element_size() for s in caches[1].stride()
    ]
    assert result["group_details"] == [
        dict(
            group_id=0,
            layers=config.kv_cache_groups[0].layer_names,
            num_blocks=config.num_blocks,
            block_size_tokens=2,
            page_size_bytes=config.kv_cache_groups[0].kv_cache_spec.page_size_bytes,
            layout="LBHNC",
        )
    ]


@pytest.mark.parametrize("failure", ["extra_backing", "wrong_size"])
def test_kv_storage_discrepancies_reject_comparison(kv_storage, failure):
    config, caches = kv_storage
    if failure == "extra_backing":
        caches[1] = caches[1].clone()
    else:
        for placement in config.kv_cache_tensors:
            placement.size += 16
    with pytest.raises(Unsupported, match="KV backing") as error:
        kv_storage_inventory(config, caches, "cpu")
    assert error.value.reason == "kv_storage_mismatch"


@pytest.mark.parametrize("failure", ["host", "layout", "blocks", "coverage", "device"])
def test_unsupported_kv_layouts_are_explicit(kv_storage, failure):
    config, caches = kv_storage
    device = "cpu"
    if failure == "host":
        config.kv_cache_tensors[0].host_resident = True
    elif failure == "layout":
        config.kv_cache_layout = "unknown"
    elif failure == "blocks":
        caches[0] = caches[0][1:]
    elif failure == "coverage":
        caches.pop()
    else:
        device = "cuda:0"
    with pytest.raises(Unsupported) as error:
        kv_storage_inventory(config, caches, device)
    assert error.value.reason == "unsupported_kv_layout"


def resolved():
    return json.loads(json.dumps([asdict(m.descriptor) for m in probe_trace().members]))


def test_control_selection_preserves_full_native_descriptors():
    descriptors = resolved()
    assert select_descriptors(descriptors, "shared_baseline", None) == descriptors
    assert select_descriptors(descriptors, "separate_pools", None) == descriptors
    assert (
        select_descriptors(descriptors, "standalone", descriptors[1])
        == descriptors[1:2]
    )
    assert select_descriptors(descriptors, "max_tokens_only", None) == descriptors[-1:]
    changed = json.loads(json.dumps(descriptors[1]))
    fields = dict(changed["fields"])
    fields["num_reqs"] = 2
    changed["fields"] = sorted(fields.items())
    with pytest.raises(Unsupported, match="exact resolved descriptor"):
        select_descriptors(descriptors, "standalone", changed)


@pytest.mark.parametrize(
    "options",
    [
        {"tensor_parallel_size": 2},
        {"enable_sleep_mode": True},
        {"speculative_config": {}},
        {"compilation_config": {"cudagraph_mode": "FULL"}},
        {"max_num_seqs": 1},
    ],
)
def test_unsupported_execution_controls_are_rejected(options):
    with pytest.raises(ValueError):
        validate_engine_options({"model": "test", **options}, [1, 2, 4], 4096)


def test_fixed_kv_and_capture_policy_are_explicit():
    options = validate_engine_options({"model": "test"}, [1, 2, 4], 4096)
    assert options["kv_cache_memory_bytes"] == 4096
    assert options["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4]
    assert options["distributed_executor_backend"] == "uni"


def test_request_sequence_contains_padded_and_alternating_shapes():
    sequence = request_sequence([1, 4, 8], 100, 42)
    assert sequence == request_sequence([1, 4, 8], 100, 42)
    assert len(sequence) == 100
    assert {1, 3, 4, 7, 8}.issubset(sequence)
    assert sequence[:6] == [1, 4, 8, 8, 4, 1]


@pytest.mark.parametrize("reason", ["cuda_unavailable", "kv_storage_mismatch"])
def test_unsupported_worker_writes_failure_manifest_and_raw_event(
    tmp_path, monkeypatch, reason
):
    def reject(*args):
        raise Unsupported(reason, "injected collector rejection")

    monkeypatch.setitem(
        sys.modules, "benchmarks.cudagraphs.production", SimpleNamespace(collect=reject)
    )
    monkeypatch.setattr(
        "benchmarks.cudagraphs.collect.environment_manifest", lambda: {}
    )
    # run_worker deliberately sets process environment before importing the adapter.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    directory = tmp_path / "run"
    job = dict(
        directory=str(directory),
        run_id="run",
        pair_id="pair",
        variant="shared_baseline",
        engine_options={},
        capture_sizes=[1],
        seed=0,
        sample_interval_ms=100,
    )
    assert run_worker(job) == 2
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "unsupported"
    assert manifest["reason"] == reason
    assert not (directory / "record.json").exists()
    event = json.loads((directory / "events.jsonl").read_text())
    assert event["status"] == "unsupported" and event["run_id"] == "run"
    with pytest.raises(FileExistsError):
        run_worker(job)


def test_production_adapter_preserves_capture_keywords_and_restores_hooks(monkeypatch):
    from benchmarks.cudagraphs.production import CaptureObserver, native_descriptor
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    descriptor = BatchDescriptor(4, 4, True, False, 0)
    calls = []

    def warmup(desc, cudagraph_runtime_mode, **kwargs):
        calls.append((desc, cudagraph_runtime_mode, kwargs))

    dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [(CUDAGraphMode.FULL, [descriptor])]
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False, is_moe=False, is_hybrid=False
        ),
        cudagraph_dispatcher=dispatcher,
        _warmup_and_capture=warmup,
    )
    observer = CaptureObserver(
        dict(variant="shared_baseline", selected_descriptor=None), None
    )
    snapshots = []
    monkeypatch.setattr(
        observer, "snapshot", lambda phase, **kwargs: snapshots.append(phase)
    )
    value = native_descriptor(CUDAGraphMode.FULL, descriptor)
    from benchmarks.cudagraphs.cpu import memory
    from benchmarks.cudagraphs.schema import decode

    monkeypatch.setattr(
        observer,
        "inventory",
        lambda: [{"member_id": decode(memory.Descriptor, value).member_id}],
    )

    def capture(target):
        target._warmup_and_capture(
            descriptor,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            allow_microbatching=False,
        )
        return 123

    original_dispatch = dispatcher.get_capture_descs
    assert observer.capture(runner, capture) == 123
    assert calls == [(descriptor, CUDAGraphMode.FULL, {"allow_microbatching": False})]
    assert runner._warmup_and_capture is warmup
    assert dispatcher.get_capture_descs is original_dispatch
    assert snapshots == [
        "capture-before",
        "member-0-before",
        "member-0-after",
        "capture-after",
    ]
