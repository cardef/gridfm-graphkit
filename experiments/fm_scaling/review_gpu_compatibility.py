# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt the detailed authoritative I010 report to the strict gate schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import file_sha256


EXPECTED_CHECKS = {
    "upstream_identity",
    "fresh_cpu_compatibility",
    "upstream_flat_checkpoint_load",
    "cuda_compile_exercised",
    "compile_policy_fail_closed",
    "profiler_flop_crosscheck",
    "largest_grid_host_and_accelerator_peaks",
}


def review_gpu_compatibility(input_path: Path, output_path: Path) -> dict:
    payload = json.loads(input_path.read_text())
    checks = {
        str(item.get("name")): item.get("passed")
        for item in payload.get("checks", [])
        if isinstance(item, dict)
    }
    results = payload.get("results", {})
    device = results.get("cuda", {}).get("device")
    if (
        payload.get("schema_version") != "fm-scaling-evidence-v1"
        or payload.get("gate_id") != "I010"
        or payload.get("status") != "PASS"
        or set(checks) != EXPECTED_CHECKS
        or any(value is not True for value in checks.values())
        or device != "NVIDIA A40"
    ):
        raise ContractError("authoritative I010 report is incomplete or not an A40 PASS")
    fork_commit = str(results.get("fork_commit", ""))
    if len(fork_commit) != 40:
        raise ContractError("authoritative I010 report lacks a full fork commit")
    reviewed = {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": "I010",
        "status": "PASS",
        "checks": [
            {"name": "i010_criteria", "passed": True},
            {"name": "immutable_inputs", "passed": True},
        ],
        "inputs": [
            {
                "path": str(input_path.resolve()),
                "sha256": file_sha256(input_path.resolve()),
            },
        ],
        "results": {
            "fork_commit": fork_commit,
            "device": device,
            "largest_network": results.get("largest_network"),
            "largest_bus_count": results.get("largest_bus_count"),
            "compile_mode": results.get("compile_policy", {}).get("selected_mode"),
            "profiler_relative_gap": results.get("profiler", {}).get("relative_gap"),
            "reviewed_checks": sorted(EXPECTED_CHECKS),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        json.dump(reviewed, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return reviewed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review_gpu_compatibility(args.input.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
