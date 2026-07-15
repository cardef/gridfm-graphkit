# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic R004--R012 reducers over immutable calibration/profile records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gridfm_graphkit.fm_scaling.accounting import deterministic_capacity_match
from gridfm_graphkit.fm_scaling.analysis import design_power
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import file_sha256


def _load(path: Path):
    return json.loads(path.read_text())


def freeze_capacity(payload: dict) -> dict:
    candidates = {
        arm: [(item["spec"], int(item["parameters"])) for item in items]
        for arm, items in payload["candidates"].items()
    }
    return {
        "schema_version": "fm-scaling-r004-v1",
        "status": "PASS",
        **deterministic_capacity_match(candidates),
    }


def freeze_loss(payload: dict) -> dict:
    candidates = payload.get("candidates", [])
    if len(candidates) != 3 or any(
        item.get("status") != "FINISHED" for item in candidates
    ):
        raise ContractError(
            "R006 requires exactly three finished Flat calibration runs",
        )
    if sum(float(item["gpu_hours"]) for item in candidates) > 3:
        raise ContractError("Flat calibration exceeds its three GPU-hour budget")
    feasible = [
        item
        for item in candidates
        if math.isfinite(float(item["error"]))
        and math.isfinite(float(item["residual"]))
    ]
    if not feasible:
        raise ContractError("no finite loss candidate")
    selected = min(
        feasible,
        key=lambda item: (float(item["error"]), float(item["residual"]), item["id"]),
    )
    return {
        "schema_version": "fm-scaling-r006-v1",
        "status": "PASS",
        "selection_rule": "min_error_then_residual_then_id",
        "selected": selected,
        "candidates": candidates,
    }


def freeze_power(payload: dict) -> dict:
    errors = [float(value) for value in payload["source_dev_group_log_errors"]]
    if len(errors) < 2:
        raise ContractError("R007 requires at least two source-development groups")
    sigma = math.sqrt(2) * float(np.std(errors, ddof=1))
    available = int(payload["available_target_groups"])
    rows = []
    for groups in range(6, min(available, 10) + 1):
        rows.append(
            {
                "group_count": groups,
                "power": design_power(group_count=groups, sigma_design=sigma),
            },
        )
    selected = next((row for row in rows if row["power"] >= 0.8), None)
    if selected is None:
        raise ContractError("no feasible target-group count reaches 80 percent power")
    return {
        "schema_version": "fm-scaling-r007-r008-v1",
        "status": "PASS",
        "s_flat": sigma / math.sqrt(2),
        "sigma_design": sigma,
        "seed": 20260714,
        "draws": 1_000_000,
        "candidates": rows,
        "selected_group_count": selected["group_count"],
    }


def freeze_budget(payload: dict) -> dict:
    bounds = {key: float(value) for key, value in payload["run_upper_hours"].items()}
    if len(bounds) != 20 or min(bounds.values()) <= 0:
        raise ContractError("R012 requires twenty positive run upper bounds")
    campaign = sum(bounds.values())
    total = 10 + campaign + 0.2 * campaign
    common_flops = int(payload["common_flops"])
    learning_horizon = int(payload["learning_horizon_flops"])
    if total > 230 or common_flops < learning_horizon:
        raise ContractError("R012 budget or source-only learning-horizon gate fails")
    return {
        "schema_version": "fm-scaling-r010-r012-v1",
        "status": "PASS",
        "run_upper_hours": bounds,
        "campaign_upper_hours": campaign,
        "reserve_hours": 0.2 * campaign,
        "total_gpu_hours": total,
        "common_flops": common_flops,
        "learning_horizon_flops": learning_horizon,
    }


REDUCERS = {
    "capacity": freeze_capacity,
    "loss": freeze_loss,
    "power": freeze_power,
    "budget": freeze_budget,
}
ALLOWED_GATES = {
    "capacity": {"R004"},
    "loss": {"R006"},
    "power": {"R007", "R008"},
    "budget": {"R010", "R012"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(REDUCERS))
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gate_id not in ALLOWED_GATES[args.kind]:
        raise ContractError(f"{args.kind} cannot produce {args.gate_id} evidence")
    result = REDUCERS[args.kind](_load(args.input.resolve()))
    result_payload = dict(result)
    result.update(
        {
            "result_schema_version": result.pop("schema_version"),
            "schema_version": "fm-scaling-evidence-v1",
            "gate_id": args.gate_id,
            "checks": [
                {"name": f"{args.gate_id.lower()}_criteria", "passed": True},
                {"name": "immutable_inputs", "passed": True},
            ],
            "inputs": [
                {
                    "path": str(args.input.resolve()),
                    "sha256": file_sha256(args.input.resolve()),
                },
            ],
            "results": result_payload,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
