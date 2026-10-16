# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the CPU planner on a saved contract trace; never a GPU benchmark."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.cudagraphs.cpu import memory, planner
from benchmarks.cudagraphs.schema import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arena-limit", type=int, default=memory.MAX_BYTES)
    args = parser.parse_args()
    trace = load(args.trace, memory.FamilyTrace)
    try:
        plan = planner.plan(trace, arena_limit=args.arena_limit)
        result = {
            "schema_version": memory.SCHEMA_VERSION,
            "artifact_kind": "cpu_simulation",
            "status": "success",
            "trace_digest": trace.fingerprint,
            "plan": asdict(plan),
            "verification": "passed",
            "gpu_binding_validation": "pending",
        }
    except memory.UnionMemoryError as error:
        result = {
            "schema_version": memory.SCHEMA_VERSION,
            "artifact_kind": "cpu_simulation",
            "status": "rejected",
            "reason": error.reason,
            "detail": str(error),
        }
    # Exclusive creation prevents accidentally overwriting experimental evidence.
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
