# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Run one frozen Flat loss calibration and emit typed C001--C003 evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.manifest import (
    file_sha256,
    validate_gate_evidence,
)


CANDIDATE_IDS = ("C001", "C002", "C003")
SAMPLES_TOTAL = 11_655


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path.resolve())}


def _candidate(loss_candidates_path: Path, candidate_id: str) -> dict:
    payload = yaml.safe_load(loss_candidates_path.read_text())
    rows = payload.get("candidates", [])
    if (
        payload.get("schema_version") != "fm-scaling-loss-candidates-v1"
        or int(payload.get("candidate_count", 0)) != 3
        or [row.get("id") for row in rows] != list(CANDIDATE_IDS)
    ):
        raise ContractError("loss candidate table differs from the frozen matrix")
    try:
        candidate = next(row for row in rows if row["id"] == candidate_id)
    except StopIteration as error:
        raise ContractError(f"unknown calibration candidate {candidate_id}") from error
    weights = [float(value) for value in candidate.get("loss_weights", [])]
    if len(weights) != 2 or min(weights) <= 0:
        raise ContractError("calibration loss weights must contain two positives")
    return {"id": candidate_id, "loss_weights": weights}


def _evaluation_policy(policy_path: Path) -> dict:
    payload = yaml.safe_load(policy_path.read_text())
    scales = payload.get("metric_scales", {})
    expected = {
        "schema_version": "fm-scaling-calibration-evaluation-policy-v1",
        "communication_core": "flat",
        "compile_mode": "default",
        "candidate_count": 3,
    }
    if (
        any(payload.get(key) != value for key, value in expected.items())
        or float(scales.get("vm_pu", 0)) != 0.01
        or not math.isclose(
            float(scales.get("va_rad", 0)),
            math.pi / 180.0,
            rel_tol=0,
            abs_tol=1e-15,
        )
    ):
        raise ContractError("calibration evaluation policy is not frozen")
    return payload

def _r005_budget(r005_evidence_path: Path) -> int:
    payload = validate_gate_evidence(r005_evidence_path, "R005")
    budget = payload.get("results", {}).get("budget", {})
    if (
        payload.get("status") != "PASS"
        or int(budget.get("candidate_count", 0)) != 3
        or float(budget.get("aggregate_upper_seconds", math.inf)) > 10_800
    ):
        raise ContractError("R005 does not authorize the calibration matrix")
    c_cal = int(budget.get("c_cal_flops", 0))
    if c_cal <= 0:
        raise ContractError("R005 lacks a positive C_cal")
    return c_cal


def build_calibration_config(
    *,
    candidate: dict,
    c_cal: int,
    metric_scales: dict,
    topology_payload: dict,
    topology_manifest: Path,
    geometry_bundle: Path,
    split_root: Path,
    output_root: Path,
) -> dict:
    metadata = topology_payload["topologies"]
    source = sorted(
        network for network, record in metadata.items() if record["split"] == "source"
    )
    source_dev = sorted(
        network
        for network, record in metadata.items()
        if record["split"] == "source_dev"
    )
    if len(source) != 26 or len(source_dev) != 2:
        raise ContractError("calibration requires exactly 26 source and 2 source-dev grids")
    if any(metadata[name].get("integrity_status") != "PASS" for name in source + source_dev):
        raise ContractError("calibration topology integrity is incomplete")
    if any(int(metadata[name]["scenario_count"]) != 2_331 for name in source):
        raise ContractError("source pools differ from the frozen 2,331 scenarios")
    if any(int(metadata[name]["scenario_count"]) != 512 for name in source_dev):
        raise ContractError("source-development pools differ from 512 scenarios")

    run_id = str(candidate["id"])
    return {
        "callbacks": {"patience": 100, "tol": 0.0},
        "task": {"task_name": "FMScalingPowerFlow"},
        "data": {
            "confirmatory": True,
            "hierarchy": {"enable": False},
            "normalization": "CaseDeclaredMVANormalizer",
            "topology_manifest": str(topology_manifest),
            "mask_value": 0.0,
            "networks": source,
            "train_networks": source,
            "target_networks": source_dev,
            "provenance_groups": [
                metadata[network]["provenance_group"] for network in source
            ],
            "scenarios": [int(metadata[network]["scenario_count"]) for network in source],
            "target_scenarios": [
                int(metadata[network]["scenario_count"]) for network in source_dev
            ],
            "split_from_existing_files": str(split_root),
            "test_ratio": 0.0,
            "val_ratio": 0.0,
            "workers": 4,
            "same_grid_batches": True,
            "samples_total": SAMPLES_TOTAL,
            "consolidated": True,
        },
        "model": {
            "type": "FMScalingPF",
            "communication_core": "flat",
            "geometry_bundle": str(geometry_bundle),
            "hidden_size": 122,
            "edge_dim": 10,
            "input_bus_dim": 15,
            "input_gen_dim": 6,
            "l_pre": 2,
            "l_post": 2,
            "flat_blocks": 1,
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "learning_rate": 0.0005,
            "lr_decay": 0.7,
            "lr_patience": 5,
        },
        "seed": 0,
        "training": {
            "batch_size": 1,
            "epochs": 10_000,
            "losses": ["GraphBalancedMaskedVMVA", "GraphBalancedPBE"],
            "loss_args": [{}, {}],
            "loss_weights": candidate["loss_weights"],
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "flop_checkpoints": [c_cal],
            "flop_checkpoint_dir": str(output_root / f"{run_id}-checkpoints"),
            "runtime_output_path": str(output_root / f"{run_id}-runtime.json"),
        },
        "evaluation": {
            "run_id": run_id,
            "g_level": "G26",
            "checkpoint": "C_cal",
            "vm_scale": float(metric_scales["vm_pu"]),
            "va_scale": float(metric_scales["va_rad"]),
            "output_path": str(output_root / f"{run_id}-metrics.json"),
        },
        "verbose": False,
    }


def aggregate_calibration_metrics(records: list[dict], topology_payload: dict) -> dict:
    metadata = topology_payload["topologies"]
    expected = {
        record["topology_key"]: (name, record["provenance_group"])
        for name, record in metadata.items()
        if record["split"] == "source_dev"
    }
    if not records or {row.get("topology_key") for row in records} != set(expected):
        raise ContractError("calibration metrics do not cover exactly source-development")
    topology_rows: dict[str, list[dict]] = {}
    for row in records:
        topology_key = str(row["topology_key"])
        error = float(row["family_balanced_error"])
        residual = float(row["dimensionless_residual"])
        if not all(math.isfinite(value) and value >= 0 for value in (error, residual)):
            raise ContractError("calibration metrics must be finite and nonnegative")
        topology_rows.setdefault(topology_key, []).append(row)

    topologies = []
    groups: dict[str, list[dict]] = {}
    for topology_key in sorted(topology_rows):
        name, group = expected[topology_key]
        rows = topology_rows[topology_key]
        summary = {
            "network": name,
            "topology_key": topology_key,
            "provenance_group": group,
            "scenario_count": len(rows),
            "error": sum(float(row["family_balanced_error"]) for row in rows) / len(rows),
            "residual": sum(float(row["dimensionless_residual"]) for row in rows)
            / len(rows),
        }
        topologies.append(summary)
        groups.setdefault(group, []).append(summary)

    group_rows = []
    for group in sorted(groups):
        rows = groups[group]
        group_rows.append(
            {
                "provenance_group": group,
                "topology_count": len(rows),
                "error": sum(row["error"] for row in rows) / len(rows),
                "residual": sum(row["residual"] for row in rows) / len(rows),
            },
        )
    return {
        "error": sum(row["error"] for row in group_rows) / len(group_rows),
        "residual": sum(row["residual"] for row in group_rows) / len(group_rows),
        "topologies": topologies,
        "groups": group_rows,
    }


def _run(command: list[str], repo_root: Path) -> None:
    result = subprocess.run(command, cwd=repo_root, check=False)
    if result.returncode != 0:
        raise ContractError(
            f"calibration subprocess failed with return code {result.returncode}",
        )


def _subprocess_python(path: Path) -> Path:
    """Return an absolute interpreter path without dereferencing venv symlinks."""
    return Path(os.path.abspath(path.expanduser()))


def run_calibration(args) -> dict:
    repo_root = args.repo_root.resolve()
    if _git(repo_root, "status", "--short"):
        raise ContractError("calibration launch requires a clean worktree")
    fork_commit = _git(repo_root, "rev-parse", "HEAD")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError("calibration launch requires exactly one visible GPU")
    device = torch.cuda.get_device_name(0)
    if device != "NVIDIA A40":
        raise ContractError("authoritative calibration requires NVIDIA A40")

    candidate = _candidate(args.loss_candidates.resolve(), args.candidate_id)
    c_cal = _r005_budget(args.r005_evidence.resolve())
    policy = _evaluation_policy(args.evaluation_policy.resolve())
    topology_manifest = args.topology_manifest.resolve()
    topology_payload = load_topology_manifest(topology_manifest)
    geometry_bundle = args.geometry_bundle.resolve()
    split_root = args.split_root.resolve()
    split_manifest = args.split_manifest.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = candidate["id"]
    config_path = output_root / f"{run_id}-config.yaml"
    evidence_path = args.evidence.resolve()
    if config_path.exists() or evidence_path.exists():
        raise ContractError(f"refusing to overwrite prior {run_id} artifacts")

    config = build_calibration_config(
        candidate=candidate,
        c_cal=c_cal,
        topology_payload=topology_payload,
        topology_manifest=topology_manifest,
        geometry_bundle=geometry_bundle,
        split_root=split_root,
        output_root=output_root,
        metric_scales=policy["metric_scales"],
    )
    with config_path.open("x") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    python = _subprocess_python(args.python)
    common = [
        "--config",
        str(config_path),
        "--plugins",
        "gridfm_graphkit.fm_scaling",
        "--log_dir",
        str(args.mlflow_store.resolve()),
        "--data_path",
        str(data_root),
        "--compile",
        str(policy["compile_mode"]),
    ]
    started = time.monotonic()
    _run(
        [
            str(python),
            "-m",
            "gridfm_graphkit",
            "train",
            *common,
            "--exp_name",
            "fm-scaling-calibration",
            "--run_name",
            run_id,
            "--deterministic",
            "true",
            "--train_only",
        ],
        repo_root,
    )
    checkpoint_dir = Path(config["training"]["flop_checkpoint_dir"])
    checkpoint = checkpoint_dir / f"flops_{c_cal}.pt"
    ledger_path = checkpoint_dir / "flop_checkpoints.json"
    runtime_path = Path(config["training"]["runtime_output_path"])
    if not all(path.is_file() for path in (checkpoint, ledger_path, runtime_path)):
        raise ContractError(f"{run_id} lacks its first-crossing training artifacts")
    ledger = json.loads(ledger_path.read_text())
    crossed = ledger.get("crossed", [])
    if (
        ledger.get("thresholds") != [c_cal]
        or len(crossed) != 1
        or int(crossed[0].get("threshold", 0)) != c_cal
        or not (
            int(crossed[0].get("previous_flops", c_cal))
            < c_cal
            <= int(crossed[0].get("crossing_flops", -1))
        )
    ):
        raise ContractError(f"{run_id} did not preserve first-crossing C_cal")

    metrics_path = Path(config["evaluation"]["output_path"])
    _run(
        [
            str(python),
            "-m",
            "gridfm_graphkit",
            "evaluate",
            *common,
            "--model_path",
            str(checkpoint),
            "--exp_name",
            "fm-scaling-calibration-evaluation",
            "--run_name",
            f"{run_id}-C-cal",
            "--evaluation_checkpoint",
            "C_cal",
            "--evaluation_output",
            str(metrics_path),
            "--evaluation_targets",
        ],
        repo_root,
    )
    elapsed = time.monotonic() - started
    metrics = aggregate_calibration_metrics(
        json.loads(metrics_path.read_text()),
        topology_payload,
    )
    checks = [
        {"name": f"{run_id.lower()}_criteria", "passed": True},
        {"name": "immutable_inputs", "passed": True},
        {"name": "flat_only", "passed": True},
        {"name": "source_development_evaluation_only", "passed": True},
        {"name": "first_crossing_c_cal", "passed": True},
        {"name": "one_gpu_hour_bound", "passed": elapsed <= 3_600},
    ]
    payload = {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": run_id,
        "status": "PASS" if all(check["passed"] for check in checks) else "BLOCKED",
        "checks": checks,
        "inputs": [
            _file_record(args.r005_evidence),
            _file_record(args.loss_candidates),
            _file_record(topology_manifest),
            _file_record(args.evaluation_policy),
            _file_record(geometry_bundle),
            _file_record(split_manifest),
            _file_record(config_path),
        ],
        "results": {
            "run_status": "FINISHED",
            "candidate": candidate,
            "c_cal_flops": c_cal,
            "gpu_hours": elapsed / 3_600.0,
            "wall_seconds": elapsed,
            "error": metrics["error"],
            "residual": metrics["residual"],
            "group_metrics": metrics["groups"],
            "topology_metrics": metrics["topologies"],
            "fork_commit": fork_commit,
            "upstream_commit": _git(repo_root, "rev-parse", "upstream/main"),
            "merge_base": _git(repo_root, "merge-base", "HEAD", "upstream/main"),
            "device": device,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "artifacts": {
                "checkpoint": _file_record(checkpoint),
                "ledger": _file_record(ledger_path),
                "runtime": _file_record(runtime_path),
                "metrics": _file_record(metrics_path),
            },
        },
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({run_id: payload["status"]}))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", choices=CANDIDATE_IDS, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--r005-evidence", type=Path, required=True)
    parser.add_argument("--loss-candidates", type=Path, required=True)
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-policy", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mlflow-store", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    payload = run_calibration(args)
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
