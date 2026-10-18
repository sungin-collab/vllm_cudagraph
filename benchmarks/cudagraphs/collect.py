# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fresh-process diagnostic collection through the production V1 capture manager."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

from benchmarks.cudagraphs.cpu import memory
from benchmarks.cudagraphs.schema import decode

VARIANTS = ("shared_baseline", "max_tokens_only", "standalone", "separate_pools")
ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_ENABLE_CUDA_COMPATIBILITY",
    "VLLM_CUDA_COMPATIBILITY_PATH",
    "LD_LIBRARY_PATH",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "VLLM_USE_FLASHINFER_SAMPLER",
)


class Unsupported(RuntimeError):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(detail)


class Artifacts:
    """Append-only raw events; every artifact records its measurement identity."""

    def __init__(self, directory: Path, job: dict):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.header = {
            "schema_version": 1,
            "run_id": job["run_id"],
            "pair_id": job["pair_id"],
            "variant": job["variant"],
            "pass": "diagnostic",
        }
        self._lock = threading.Lock()
        self._events = (directory / "events.jsonl").open("x")

    def write(self, name: str, data: dict) -> None:
        with (self.directory / name).open("x") as target:
            json.dump({**self.header, **data}, target, indent=2, allow_nan=False)
            target.write("\n")

    def event(self, phase: str, *, status: str = "success", **data) -> None:
        event = {
            **self.header,
            "status": status,
            "phase": phase,
            "monotonic_ns": time.monotonic_ns(),
            "clock_domain": "monotonic_ns",
            "pid": os.getpid(),
            "rank": 0,
            **data,
        }
        with self._lock:
            self._events.write(json.dumps(event, allow_nan=False) + "\n")
            self._events.flush()

    def close(self) -> None:
        self._events.close()


def environment_manifest() -> dict:
    versions: dict[str, str | None] = {}
    for package in ("vllm", "torch", "transformers", "triton", "flashinfer-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
    )
    # Include source files not yet staged; a plain git diff misses them.
    source_hashes = {}
    for name in untracked.decode().split("\0"):
        if name and (root / name).is_file():
            import hashlib

            source_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {
        "command": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "packages": versions,
        "revision": revision,
        "dirty_diff_digest": memory.digest([diff.hex(), source_hashes]),
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "clock_mapping": {
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
            "sampler_clock": "same-process monotonic_ns",
        },
    }


def validate_engine_options(
    options: dict, capture_sizes: list[int], kv_bytes: int
) -> dict:
    if not isinstance(options, dict) or not options.get("model"):
        raise ValueError("Engine config must contain a model")
    if any(key in options for key in ("hf_token", "api_key")):
        raise ValueError("Use the model cache or environment for credentials")
    if not capture_sizes or capture_sizes != sorted(set(capture_sizes)):
        raise ValueError("Capture sizes must be unique and increasing")
    if (
        any(type(size) is not int or size <= 0 for size in capture_sizes)
        or kv_bytes <= 0
    ):
        raise ValueError("Capture sizes and fixed KV bytes must be positive")
    forbidden = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_lora": False,
        "enable_sleep_mode": False,
        "enable_dbo": False,
        "cpu_offload_gb": 0,
        "speculative_config": None,
        "kv_transfer_config": None,
        "weight_transfer_config": None,
        "quantization": None,
        "enforce_eager": False,
        "worker_cls": "auto",
    }
    for key, allowed in forbidden.items():
        if key in options and options[key] != allowed:
            raise ValueError(f"Unsupported initial collector feature: {key}")
    result = dict(options)
    compilation = result.get("compilation_config", {})
    if not isinstance(compilation, dict):
        raise ValueError("compilation_config must be an object")
    compilation = dict(compilation)
    if compilation.get("cudagraph_mode", "FULL_DECODE_ONLY") != "FULL_DECODE_ONLY":
        raise ValueError("Collector requires FULL_DECODE_ONLY")
    if compilation.get("cudagraph_capture_sizes", capture_sizes) != capture_sizes:
        raise ValueError("Conflicting capture size controls")
    compilation.update(
        cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=capture_sizes
    )
    result.update(
        compilation_config=compilation,
        kv_cache_memory_bytes=kv_bytes,
        distributed_executor_backend="uni",
        enable_prefix_caching=False,
    )
    if result.get("max_num_seqs", max(capture_sizes)) < max(capture_sizes):
        raise ValueError("max_num_seqs must cover the full decode capture set")
    result.setdefault("max_num_seqs", max(capture_sizes))
    return result


def select_descriptors(
    resolved: list[dict], variant: str, selected: dict | None
) -> list[dict]:
    if not resolved:
        raise Unsupported("empty_capture_set", "No FULL decode descriptors resolved")
    for value in resolved:
        decode(memory.Descriptor, value).validate()
    if variant in ("shared_baseline", "separate_pools"):
        return resolved
    if variant == "max_tokens_only":
        largest = max(dict(value["fields"])["num_tokens"] for value in resolved)
        candidates = [
            value
            for value in resolved
            if dict(value["fields"])["num_tokens"] == largest
        ]
        if len(candidates) != 1:
            raise Unsupported(
                "ambiguous_descriptor",
                "Maximum tokens does not identify one descriptor",
            )
        return candidates
    if variant == "standalone" and selected in resolved:
        return [selected]
    raise Unsupported(
        "descriptor_mismatch", "Standalone requires one exact resolved descriptor"
    )


def request_sequence(tokens: list[int], steps: int, seed: int) -> list[int]:
    import random

    rng = random.Random(seed)
    # Include unaligned requests as well as exact descriptors.
    candidates = sorted(set(tokens + [max(1, size - 1) for size in tokens]))
    prefix = tokens + list(reversed(tokens)) + [tokens[0]] * 4
    prefix += [value for _ in range(4) for value in (tokens[0], tokens[-1])]
    return (
        prefix + [rng.choice(candidates) for _ in range(max(0, steps - len(prefix)))]
    )[:steps]


def run_worker(job: dict) -> int:
    artifacts = Artifacts(Path(job["directory"]), job)
    manifest = environment_manifest()
    manifest.update(
        engine_options=job["engine_options"],
        capture_policy=job["capture_sizes"],
        seed=job["seed"],
        peak_sample_interval_ms=job["sample_interval_ms"],
        peak_kind="sampled lower bound",
        instrumentation="diagnostic only",
        trimming_policy="unchanged production capture_model schedule",
        gc_policy="unchanged production capture_model schedule",
        output_retention=(
            "production wrapper contract; dummy outputs consumed before next replay"
        ),
        runner="v1",
        requested_memory_mode="default",
        effective_memory_mode="default",
    )
    code = 0
    try:
        if job["variant"] not in VARIANTS:
            raise Unsupported(
                "unsupported_runtime", "Production union binding is not validated"
            )
        # Initialize this before importing vLLM or creating an engine.
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        manifest["environment"] = {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}
        from benchmarks.cudagraphs.production import collect

        record = collect(job, artifacts, manifest)
        # Publish a successful measurement only after adapter cleanup succeeds.
        with (artifacts.directory / "record.json").open("x") as target:
            json.dump(asdict(record), target, indent=2)
        manifest["status"] = "success"
    except Unsupported as error:
        manifest.update(status="unsupported", reason=error.reason, detail=str(error))
        artifacts.event(
            "rejection", status="unsupported", reason=error.reason, detail=str(error)
        )
        code = 2
    except Exception as error:
        manifest.update(
            status="failure", reason=type(error).__name__, detail=str(error)
        )
        artifacts.event("error", status="failure", traceback=traceback.format_exc())
        code = 1
    finally:
        artifacts.write("manifest.json", manifest)
        artifacts.close()
    return code


def launch_job(job: dict, root: Path) -> dict:
    job_path = root / (job["run_id"] + "-job.json")
    with job_path.open("x") as output:
        json.dump(job, output, indent=2)
    with (root / (job["run_id"] + ".log")).open("x") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.cudagraphs.collect",
                "--worker",
                str(job_path),
            ],
            cwd=Path(__file__).resolve().parents[2],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    manifest_path = Path(job["directory"]) / "manifest.json"
    if not manifest_path.exists():
        return {
            "status": "failure",
            "reason": "worker_exited_without_manifest",
            "exit_code": result.returncode,
        }
    manifest = json.loads(manifest_path.read_text())
    if result.returncode != 0 and manifest.get("status") == "success":
        raise RuntimeError("Worker exit code disagrees with manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--engine-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--capture-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--stress-steps", type=int, default=10000)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(run_worker(json.loads(args.worker.read_text())))
    if (
        not args.engine_config
        or not args.output_dir
        or args.kv_cache_memory_bytes is None
    ):
        parser.error(
            "--engine-config, --output-dir and --kv-cache-memory-bytes are required"
        )
    if min(args.repetitions, args.stress_steps, args.sample_interval_ms) < 1:
        parser.error("Repetitions, stress steps and sampling interval must be positive")
    sizes = [int(value) for value in args.capture_sizes.split(",")]
    options = validate_engine_options(
        json.loads(args.engine_config.read_text()), sizes, args.kv_cache_memory_bytes
    )
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for repetition in range(args.repetitions):
        pair_id = uuid.uuid4().hex

        def execute(
            variant: str,
            selected: dict | None = None,
            pair_id=pair_id,
            repetition=repetition,
        ) -> dict:
            run_id = uuid.uuid4().hex
            job = {
                "run_id": run_id,
                "pair_id": pair_id,
                "variant": variant,
                "directory": str(root / run_id),
                "engine_options": options,
                "capture_sizes": sizes,
                "selected_descriptor": selected,
                "seed": args.seed + repetition,
                "stress_steps": args.stress_steps,
                "sample_interval_ms": args.sample_interval_ms,
            }
            manifest = launch_job(job, root)
            outcomes.append(
                {
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "variant": variant,
                    "status": manifest["status"],
                    "reason": manifest.get("reason"),
                }
            )
            print(json.dumps(outcomes[-1]), flush=True)
            return manifest

        # Resolve full native descriptors using a production baseline first.
        baseline = execute("shared_baseline")
        if baseline["status"] != "success":
            break
        variants = [v for v in args.variants if v != "shared_baseline"]
        if repetition % 2:
            variants.reverse()
        for variant in variants:
            if variant == "standalone":
                for descriptor in baseline["resolved_descriptors"]:
                    execute(variant, descriptor)
            else:
                execute(variant)
    with (root / "collection.json").open("x") as output:
        json.dump(
            {
                "schema_version": 1,
                "status": "success"
                if all(value["status"] == "success" for value in outcomes)
                else "incomplete",
                "runs": outcomes,
            },
            output,
            indent=2,
        )
    if any(value["status"] != "success" for value in outcomes):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
