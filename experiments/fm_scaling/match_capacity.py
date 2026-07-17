# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Enumerate the frozen model-capacity domain and record the R004 match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from gridfm_graphkit.fm_scaling.accounting import (
    deterministic_capacity_match,
    trainable_parameter_count,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import file_sha256
from gridfm_graphkit.fm_scaling.model import FMScalingPF
from gridfm_graphkit.fm_scaling.registry import GeometryRegistry


CAPACITY_DOMAIN_SCHEMA = "fm-scaling-capacity-domain-v1"
_ARMS = ("flat", "global", "kron", "quotient")
_DOMAIN_FIELDS = {
    "widths",
    "flat_blocks",
    "tolerance",
    "l_pre",
    "l_post",
    "edge_dim",
    "input_bus_dim",
    "input_gen_dim",
}


def _model_args(domain: dict, arm: str, width: int, flat_blocks: int):
    return SimpleNamespace(
        task=SimpleNamespace(task_name="FMScalingPowerFlow"),
        data=SimpleNamespace(
            normalization="CaseDeclaredMVANormalizer",
            confirmatory=True,
            hierarchy=SimpleNamespace(enable=False),
        ),
        model=SimpleNamespace(
            communication_core=arm,
            hidden_size=width,
            edge_dim=int(domain["edge_dim"]),
            input_bus_dim=int(domain["input_bus_dim"]),
            input_gen_dim=int(domain["input_gen_dim"]),
            l_pre=int(domain["l_pre"]),
            l_post=int(domain["l_post"]),
            flat_blocks=flat_blocks,
            geometry_bundle="capacity-only",
        ),
    )


def match_capacity(domain_path: Path) -> dict:
    payload = yaml.safe_load(domain_path.read_text())
    if payload.get("schema_version") != CAPACITY_DOMAIN_SCHEMA:
        raise ContractError("R004 capacity domain has the wrong schema")
    domain = payload.get("domain")
    if not isinstance(domain, dict) or set(domain) != _DOMAIN_FIELDS:
        raise ContractError("R004 capacity domain has the wrong fields")
    widths = [int(value) for value in domain["widths"]]
    flat_blocks = [int(value) for value in domain["flat_blocks"]]
    if (
        widths != sorted(set(widths))
        or flat_blocks != sorted(set(flat_blocks))
        or min(widths + flat_blocks) < 1
    ):
        raise ContractError("R004 widths and Flat depths must be positive and unique")
    tolerance = float(domain["tolerance"])
    candidates = {arm: [] for arm in _ARMS}
    with patch.object(
        GeometryRegistry,
        "from_bundle",
        return_value=(object(), "capacity-only"),
    ):
        for arm in _ARMS:
            depths = flat_blocks if arm == "flat" else [1]
            for width in widths:
                for depth in depths:
                    model = FMScalingPF(_model_args(domain, arm, width, depth))
                    spec = {"width": width}
                    if arm == "flat":
                        spec["flat_blocks"] = depth
                    candidates[arm].append(
                        (spec, trainable_parameter_count(model)),
                    )
    selected = deterministic_capacity_match(candidates, tolerance=tolerance)
    return {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": "R004",
        "status": "PASS",
        "checks": [
            {"name": "r004_criteria", "passed": True},
            {"name": "immutable_inputs", "passed": True},
        ],
        "inputs": [
            {
                "path": str(domain_path.resolve()),
                "sha256": file_sha256(domain_path.resolve()),
            },
        ],
        "results": {
            "outcomes_read": False,
            "tolerance": tolerance,
            "candidate_counts": {
                arm: len(rows) for arm, rows in candidates.items()
            },
            **selected,
        },
        "candidates": {
            arm: [
                {"spec": spec, "parameters": count}
                for spec, count in rows
            ]
            for arm, rows in candidates.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = match_capacity(args.domain.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["results"]["selection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
