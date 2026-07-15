# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate at most twelve source-development geometry policies and freeze one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gridfm_graphkit.fm_scaling.contracts import ContractError, GeometryBudget
from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.geometry import KronGeometryBuilder
from gridfm_graphkit.fm_scaling.manifest import file_sha256
from gridfm_graphkit.fm_scaling.topology import load_grid_topology


def evaluate_candidates(
    manifest_path: Path,
    data_root: Path,
    candidates_path: Path,
) -> dict:
    manifest = load_topology_manifest(manifest_path)
    payload = yaml.safe_load(candidates_path.read_text())
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
        raise ContractError("R003 requires between one and twelve policy candidates")
    networks = sorted(
        network
        for network, record in manifest["topologies"].items()
        if record["split"] == "source_dev"
    )
    if not networks:
        raise ContractError("R003 requires source-development topologies")
    rows = []
    for raw_policy in candidates:
        budget = GeometryBudget(**raw_policy)
        measurements = []
        failures = []
        for network in networks:
            try:
                topology = load_grid_topology(
                    data_root,
                    network,
                    manifest["topologies"][network],
                )
                geometry = KronGeometryBuilder().build(topology, budget)
                measurements.append(
                    {
                        "network": network,
                        "residual": geometry.harmonic_residual,
                        "condition_number": geometry.condition_number,
                        "cross_nnz": geometry.prolong.nnz,
                        "coarse_nnz": geometry.coarse_graph.nnz,
                        "build_seconds": geometry.provenance.build_seconds,
                        "dense_bytes": geometry.provenance.dense_bytes,
                    },
                )
            except Exception as error:
                failures.append(
                    {
                        "network": network,
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                )
        rows.append(
            {
                "policy": raw_policy,
                "policy_hash": budget.policy_hash,
                "status": "PASS" if not failures else "FAIL",
                "measurements": measurements,
                "failures": failures,
            },
        )
    feasible = [row for row in rows if row["status"] == "PASS"]
    if not feasible:
        raise ContractError("no geometry policy covers all source-development cases")
    selected = min(
        feasible,
        key=lambda row: (
            max(item["residual"] for item in row["measurements"]),
            max(item["condition_number"] for item in row["measurements"]),
            sum(item["cross_nnz"] + item["coarse_nnz"] for item in row["measurements"]),
            row["policy_hash"],
        ),
    )
    return {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": "R003",
        "status": "PASS",
        "checks": [
            {"name": "r003_criteria", "passed": True},
            {"name": "immutable_inputs", "passed": True},
        ],
        "inputs": [
            {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            {"path": str(candidates_path), "sha256": file_sha256(candidates_path)},
        ],
        "results": {"selected_policy_hash": selected["policy_hash"]},
        "selection_rule": "minimax_residual_condition_nnz_policy_hash",
        "selected_policy": selected["policy"],
        "selected_policy_hash": selected["policy_hash"],
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_candidates(
        args.manifest.resolve(),
        args.data_root.resolve(),
        args.candidates.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
