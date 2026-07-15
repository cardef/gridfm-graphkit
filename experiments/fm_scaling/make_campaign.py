# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Materialize the explicit 20 configs and hashed campaign manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.manifest import (
    CAMPAIGN_SCHEMA,
    REQUIRED_GATES,
    expected_run_matrix,
    file_sha256,
)


def _file_record(path: Path, repo_root: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(repo_root)),
        "sha256": file_sha256(path.resolve()),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    return parser.parse_args()


def _config(freeze: dict, run_id: str, core: str, g_level: str, seed: int) -> dict:
    train_networks = freeze["source_sets"][g_level]
    target_networks = freeze["target_networks"]
    metadata = freeze["topology_payload"]["topologies"]
    width = int(freeze["model"]["widths"][core])
    geometry_bundle = freeze["geometry_bundle"]
    return {
        "callbacks": freeze["callbacks"],
        "task": {"task_name": "FMScalingPowerFlow"},
        "data": {
            "confirmatory": True,
            "hierarchy": {"enable": False},
            "normalization": "CaseDeclaredMVANormalizer",
            "topology_manifest": freeze["topology_manifest"],
            "mask_value": 0.0,
            "networks": train_networks,
            "train_networks": train_networks,
            "target_networks": target_networks,
            "failed_target_networks": freeze.get("failed_target_networks", []),
            "provenance_groups": [
                metadata[network]["provenance_group"] for network in train_networks
            ],
            "scenarios": [
                int(freeze["available_scenarios"][network])
                for network in train_networks
            ],
            "target_scenarios": [
                int(freeze["available_scenarios"][network])
                for network in target_networks
            ],
            "split_from_existing_files": freeze["split_directory"],
            "test_ratio": 0.0,
            "val_ratio": 0.0,
            "workers": int(freeze["data_workers"]),
            "same_grid_batches": True,
            "samples_total": int(freeze["samples_total"]),
            "consolidated": bool(freeze.get("consolidated", True)),
        },
        "model": {
            "type": "FMScalingPF",
            "communication_core": core,
            "geometry_bundle": geometry_bundle,
            "hidden_size": width,
            "edge_dim": 10,
            "input_bus_dim": 15,
            "input_gen_dim": 6,
            "l_pre": int(freeze["model"]["l_pre"]),
            "l_post": int(freeze["model"]["l_post"]),
            "flat_blocks": int(freeze["model"]["flat_blocks"]),
        },
        "optimizer": freeze["optimizer"],
        "seed": seed,
        "training": {
            "batch_size": int(freeze["training"]["batch_size"]),
            "epochs": int(freeze["training"]["epochs"]),
            "losses": ["GraphBalancedMaskedVMVA", "GraphBalancedPBE"],
            "loss_args": [{}, {}],
            "loss_weights": [float(value) for value in freeze["loss_weights"]],
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "flop_checkpoints": [int(value) for value in freeze["flop_checkpoints"]],
            "flop_checkpoint_dir": (
                f"mlruns/fm-scaling/result-summaries/{run_id}-checkpoints"
            ),
            "runtime_output_path": (
                f"mlruns/fm-scaling/result-summaries/{run_id}-runtime.json"
            ),
        },
        "evaluation": {
            "run_id": run_id,
            "g_level": g_level,
            "checkpoint": "C",
            "vm_scale": float(freeze["metric_scales"]["vm_pu"]),
            "va_scale": float(freeze["metric_scales"]["va_rad"]),
            "output_path": f"mlruns/fm-scaling/result-summaries/{run_id}-metrics.json",
        },
        "verbose": False,
    }


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    freeze = yaml.safe_load(args.freeze.read_text())
    topology_manifest = (repo_root / freeze["topology_manifest"]).resolve()
    freeze["topology_payload"] = load_topology_manifest(topology_manifest)
    geometry_bundle = (repo_root / freeze["geometry_bundle"]).resolve()
    geometry_report = (repo_root / freeze["geometry_report"]).resolve()
    geometry_payload = json.loads(geometry_report.read_text())
    failed_targets = {
        network
        for network, item in geometry_payload["topologies"].items()
        if item["status"] == "FAIL"
        and freeze["topology_payload"]["topologies"][network]["split"] == "target"
    }
    freeze["target_networks"] = [
        network
        for network in freeze["target_networks"]
        if network not in failed_targets
    ]
    freeze["failed_target_networks"] = sorted(failed_targets)
    split_manifest = (repo_root / freeze["split_manifest"]).resolve()
    source_sets = freeze["source_sets"]
    if not (
        set(source_sets["G8"]).issubset(source_sets["G16"])
        and set(source_sets["G16"]).issubset(source_sets["G28"])
        and [len(source_sets[key]) for key in ("G8", "G16", "G28")] == [8, 16, 28]
    ):
        raise ValueError("source sets must be nested with sizes 8, 16, and 28")
    target_networks = freeze.get("target_networks")
    if not isinstance(target_networks, list) or not target_networks:
        raise ValueError("target_networks must be a nonempty explicit list")
    if set(target_networks) & set(source_sets["G28"]):
        raise ValueError("source and target networks must be disjoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_records = []
    for run_id, (core, g_level, seed) in expected_run_matrix().items():
        config_path = args.output_dir / f"{run_id}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                _config(freeze, run_id, core, g_level, seed),
                sort_keys=False,
            ),
        )
        run_records.append(
            {
                "run_id": run_id,
                "core": core,
                "g_level": g_level,
                "seed": seed,
                "config": _file_record(config_path, repo_root),
            },
        )

    gate_dir = (repo_root / freeze["gate_directory"]).resolve()
    gates = {
        gate_id: _file_record(gate_dir / f"{gate_id}.json", repo_root)
        for gate_id in REQUIRED_GATES
    }
    campaign = {
        "schema_version": CAMPAIGN_SCHEMA,
        "fork_commit": freeze["fork_commit"],
        "upstream_commit": freeze["upstream_commit"],
        "merge_base": freeze["merge_base"],
        "topology_manifest": _file_record(topology_manifest, repo_root),
        "geometry_bundle": _file_record(geometry_bundle, repo_root),
        "geometry_report": _file_record(geometry_report, repo_root),
        "split_manifest": _file_record(split_manifest, repo_root),
        "split_root": freeze["split_directory"],
        "data_root": freeze["data_root"],
        "mlflow_store": "mlruns/fm-scaling/mlflow-store",
        "result_root": "mlruns/fm-scaling/result-summaries",
        "analysis_files": [
            _file_record(
                repo_root / "experiments/fm_scaling/analyze_campaign.py",
                repo_root,
            ),
            _file_record(
                repo_root / "gridfm_graphkit/fm_scaling/analysis.py",
                repo_root,
            ),
        ],
        "gates": gates,
        "runs": run_records,
    }
    args.campaign.parent.mkdir(parents=True, exist_ok=True)
    args.campaign.write_text(yaml.safe_dump(campaign, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
