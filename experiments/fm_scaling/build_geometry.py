# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Build explicit Kron and Quotient bundles from topology-only parquet columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gridfm_graphkit.fm_scaling.contracts import ContractError, GeometryBudget
from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.geometry import (
    KronGeometryBuilder,
    QuotientGeometryBuilder,
)
from gridfm_graphkit.fm_scaling.registry import save_geometry_bundle
from gridfm_graphkit.fm_scaling.topology import load_grid_topology


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    topology_payload = load_topology_manifest(args.topology_manifest.resolve())
    policy_payload = yaml.safe_load(args.policy.read_text())
    budget = GeometryBudget(**policy_payload)
    geometries = []
    report = {
        "schema_version": "fm-scaling-geometry-build-v1",
        "status": "FAIL",
        "policy_hash": budget.policy_hash,
        "topologies": {},
    }
    failures = []
    for network, record in sorted(topology_payload["topologies"].items()):
        try:
            topology = load_grid_topology(args.data_root.resolve(), network, record)
            kron = KronGeometryBuilder().build(topology, budget)
            quotient = QuotientGeometryBuilder().build(topology, budget)
            if kron.partition != quotient.partition:
                raise ContractError("Kron and Quotient partitions differ")
            geometries.extend([kron, quotient])
            report["topologies"][network] = {
                "status": "PASS",
                "topology_hash": topology.topology_hash,
                "kron_geometry_hash": kron.geometry_hash,
                "quotient_geometry_hash": quotient.geometry_hash,
                "coarse_nodes": kron.partition.num_cells,
                "kron_cross_nnz": kron.prolong.nnz,
                "kron_coarse_nnz": kron.coarse_graph.nnz,
                "quotient_cross_nnz": quotient.prolong.nnz,
                "quotient_coarse_nnz": quotient.coarse_graph.nnz,
                "harmonic_residual": kron.harmonic_residual,
                "condition_number": kron.condition_number,
                "kron_build_seconds": kron.provenance.build_seconds,
                "quotient_build_seconds": quotient.provenance.build_seconds,
                "dense_bytes": kron.provenance.dense_bytes,
            }
        except Exception as error:
            failures.append(network)
            report["topologies"][network] = {
                "status": "FAIL",
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
    report["geometry_bundle_sha256"] = save_geometry_bundle(
        args.output.resolve(),
        geometries,
    )
    if not failures:
        report["status"] = "PASS"
    else:
        report["failed_topologies"] = failures
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
