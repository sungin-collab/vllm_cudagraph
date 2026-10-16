# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize saved paired measurements on CPU, without making acceptance claims."""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from benchmarks.cudagraphs.schema import RunRecord, load, validate_pair


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    index = int(position)
    fraction = position - index
    return (
        ordered[index] * (1 - fraction)
        + ordered[min(index + 1, len(ordered) - 1)] * fraction
    )


def _estimate(values: list[float], seed: int) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "reason": "no comparable measurements"}
    result: dict = {"mean": mean(values), "ci95": None, "reason": None}
    if len(values) < 5:
        result["reason"] = "fewer than five independent process pairs"
        return result
    rng = random.Random(seed)
    draws = [mean(rng.choices(values, k=len(values))) for _ in range(2000)]
    result["ci95"] = [_quantile(draws, 0.025), _quantile(draws, 0.975)]
    return result


def _latency_groups(record: RunRecord) -> dict[tuple[str, str, str], list[float]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample in record.latency:
        if sample.phase == "steady":
            groups[(sample.workload, sample.member_id, sample.method)].append(
                float(sample.duration_ns)
            )
    return dict(groups)


def summarize(records: list[RunRecord], *, seed: int = 0) -> dict:
    """Bootstrap independent paired runs, never individual correlated replays.

    Input collectors are responsible for fresh processes and complete manifests.
    Results retain signed memory deltas; uncertainty and unknowns remain visible.
    """
    pairs: dict[str, dict[str, RunRecord]] = defaultdict(dict)
    run_ids = set()
    for record in records:
        record.validate()
        if record.header.run_id in run_ids:
            raise ValueError("Duplicate run ID")
        run_ids.add(record.header.run_id)
        pair = pairs[record.header.pair_id]
        if record.header.variant in pair:
            raise ValueError("Duplicate treatment in pair")
        pair[record.header.variant] = record
    if not pairs:
        raise ValueError("No measurements")
    saved_bytes = []
    paired_results = []
    latency_deltas = defaultdict(list)
    control = None
    group_keys = None
    for pair_id, pair in sorted(pairs.items()):
        if set(pair) != {"shared_baseline", "union"}:
            raise ValueError(
                "Primary report requires one shared baseline and union per pair"
            )
        baseline, union = pair["shared_baseline"], pair["union"]
        validate_pair(baseline, union)
        if control is None:
            control = baseline
        else:
            # Seed/dispatch can vary across pairs; workload/configuration cannot.
            for field in (
                "controls_digest",
                "runner",
                "descriptors",
                "actual_kv_bytes",
                "actual_kv_blocks",
                "executable_count",
            ):
                if getattr(baseline, field) != getattr(control, field):
                    raise ValueError(f"Mixed experiment controls: {field}")
            if baseline.header.measurement_pass != control.header.measurement_pass:
                raise ValueError("Mixed measurement passes")
        base_memory, union_memory = baseline.memory.total(), union.memory.total()
        saving = None
        uncertainty = None
        if base_memory.value is not None and union_memory.value is not None:
            saving = base_memory.value - union_memory.value
            assert base_memory.uncertainty is not None
            assert union_memory.uncertainty is not None
            uncertainty = base_memory.uncertainty + union_memory.uncertainty
            saved_bytes.append(float(saving))
        per_run = []
        if baseline.header.measurement_pass == "timing":
            base_groups, union_groups = (
                _latency_groups(baseline),
                _latency_groups(union),
            )
            if not base_groups or base_groups.keys() != union_groups.keys():
                raise ValueError(
                    "Mismatched or absent steady latency workload coverage"
                )
            if group_keys is not None and group_keys != base_groups.keys():
                raise ValueError("Mismatched latency coverage across pairs")
            group_keys = base_groups.keys()
            for key in sorted(base_groups):
                a, b = base_groups[key], union_groups[key]
                if len(a) < 100 or len(b) < 100:
                    raise ValueError(
                        "At least 100 steady samples per workload required"
                    )
                med_a, med_b = median(a), median(b)
                tail_a, tail_b = _quantile(a, 0.95), _quantile(b, 0.95)
                if med_a <= 0 or tail_a <= 0:
                    raise ValueError("Nonpositive baseline latency")
                ratios = (med_b / med_a - 1, tail_b / tail_a - 1)
                latency_deltas[key].append(ratios)
                per_run.append(
                    {
                        "workload": key[0],
                        "member_id": key[1],
                        "method": key[2],
                        "baseline_samples": len(a),
                        "union_samples": len(b),
                        "baseline_median_ns": med_a,
                        "union_median_ns": med_b,
                        "baseline_p95_ns": tail_a,
                        "union_p95_ns": tail_b,
                    }
                )
        paired_results.append(
            {
                "pair_id": pair_id,
                "saved_bytes": saving,
                "measurement_uncertainty_bytes": uncertainty,
                "baseline_memory": asdict(baseline.memory),
                "union_memory": asdict(union.memory),
                "latency": per_run,
            }
        )
    estimate = _estimate(saved_bytes, seed)
    if len(saved_bytes) != len(pairs):
        estimate = {
            "mean": None,
            "ci95": None,
            "reason": "incomplete memory attribution",
        }
    return {
        "schema_version": 1,
        "status": "reported",
        "independent_pairs": len(pairs),
        "bootstrap_seed": seed,
        "bootstrap_replicates": 2000,
        "saved_bytes": estimate,
        "pairs": paired_results,
        "latency_regression": [
            {
                "workload": key[0],
                "member_id": key[1],
                "method": key[2],
                "median_fraction": _estimate([value[0] for value in values], seed),
                "p95_fraction": _estimate([value[1] for value in values], seed),
            }
            for key, values in sorted(latency_deltas.items())
        ],
        "correctness": "passed"
        if all(r.correctness == "passed" for r in records)
        else "pending_or_failed",
        "acceptance": (
            "pending: frozen targets, GPU binding, model and sanitizer "
            "evidence required"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = summarize([load(path, RunRecord) for path in args.records], seed=args.seed)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
