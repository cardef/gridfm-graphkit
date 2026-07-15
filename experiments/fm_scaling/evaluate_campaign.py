# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one run only after the complete 20-run training matrix is sealed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.fm_scaling.launch import (
    _training_artifacts,
    build_evaluation_commands,
    validate_launch,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import file_sha256, load_campaign_manifest


def _sealed_training_matrix(manifest, repo_root: Path) -> dict[str, dict]:
    records = {}
    campaign_hash = file_sha256(manifest.path)
    for run in manifest.runs:
        path = manifest.result_root / f"{run.run_id}-launch.json"
        if not path.is_file():
            raise ContractError(f"training matrix is incomplete: missing {path}")
        record = json.loads(path.read_text())
        if record.get("status") != "TRAINED":
            raise ContractError(f"training matrix is not sealed: {run.run_id}")
        if record.get("provenance", {}).get("campaign_sha256") != campaign_hash:
            raise ContractError(f"{run.run_id} belongs to another campaign")
        observed = _training_artifacts(manifest, run.run_id, repo_root)
        if record.get("artifacts") != observed:
            raise ContractError(
                f"{run.run_id} training artifacts changed after sealing",
            )
        records[run.run_id] = record
    return records


def _validate_metrics(
    path: Path,
    run,
    checkpoint: str,
    expected_scenarios: set[tuple[str, int]],
) -> dict:
    if not path.is_file():
        raise ContractError(f"missing evaluation metrics {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ContractError(f"empty or invalid evaluation metrics {path}")
    required = {
        "run_id",
        "system",
        "g_level",
        "seed",
        "checkpoint",
        "topology_key",
        "scenario_id",
        "rmse_vm_pu",
        "rmse_va_rad",
        "family_balanced_error",
        "dimensionless_residual",
    }
    seen = set()
    for item in payload:
        if not required.issubset(item):
            raise ContractError(f"{path} has an incomplete metric record")
        identity = (
            item["run_id"],
            item["system"],
            item["g_level"],
            int(item["seed"]),
            item["checkpoint"],
        )
        expected = (run.run_id, run.core, run.g_level, run.seed, checkpoint)
        if identity != expected:
            raise ContractError(f"{path} identity mismatch: {identity} != {expected}")
        key = (str(item["topology_key"]), int(item["scenario_id"]))
        if key in seen:
            raise ContractError(f"{path} duplicates {key}")
        seen.add(key)
    if seen != expected_scenarios:
        raise ContractError(f"{path} does not exactly match frozen target scenarios")
    return {"path": str(path), "sha256": file_sha256(path), "records": len(payload)}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--record", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    default_name = (
        f"{args.run_id}-evaluation.json"
        if args.execute
        else f"{args.run_id}-evaluation-ready-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
    )
    record_path = args.record or (
        repo_root / "mlruns/fm-scaling/result-summaries" / default_name
    )
    record_path = record_path if record_path.is_absolute() else repo_root / record_path
    if record_path.exists():
        print(json.dumps({"status": "BLOCKED", "failure": "record already exists"}))
        return 2
    lock_path = record_path.with_suffix(record_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("x") as handle:
            handle.write(f"pid={os.getpid()}\n")
    except FileExistsError:
        print(json.dumps({"status": "BLOCKED", "failure": "run is already active"}))
        return 2
    record = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED",
    }
    try:
        manifest = load_campaign_manifest(args.manifest.resolve(), repo_root)
        validate_launch(repo_root, manifest)
        training = _sealed_training_matrix(manifest, repo_root)
        run = next(item for item in manifest.runs if item.run_id == args.run_id)
        commands = build_evaluation_commands(
            manifest,
            args.run_id,
            args.python.resolve(),
            repo_root,
        )
        record.update(
            {
                "status": "READY",
                "campaign_sha256": file_sha256(manifest.path),
                "sealed_training_records": len(training),
                "commands": [command for _, _, command in commands],
            },
        )
        if args.execute:
            environment = os.environ.copy()
            environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            started = time.monotonic()
            outputs = []
            expected_scenarios = {
                (item["topology_key"], scenario)
                for network, item in manifest.topology_payload["topologies"].items()
                if item["split"] == "target"
                and network not in manifest.geometry_failures
                for scenario in range(int(item["scenario_count"]))
            }
            expected_outputs = [
                manifest.result_root / f"{args.run_id}-{suffix}-metrics.json"
                for suffix in ("C4", "C2", "C")
            ]
            if any(path.exists() for path in expected_outputs):
                raise ContractError("refusing to overwrite prior evaluation metrics")
            for checkpoint, _, command in commands:
                result = subprocess.run(
                    command,
                    cwd=repo_root,
                    env=environment,
                    check=False,
                )
                if result.returncode:
                    raise ContractError(
                        f"evaluation {checkpoint} failed with {result.returncode}",
                    )
                suffix = {"C/4": "C4", "C/2": "C2", "C": "C"}[checkpoint]
                outputs.append(
                    _validate_metrics(
                        manifest.result_root / f"{args.run_id}-{suffix}-metrics.json",
                        run,
                        checkpoint,
                        expected_scenarios,
                    ),
                )
            record["wall_seconds"] = time.monotonic() - started
            record["metrics"] = outputs
            record["training_artifacts"] = training[args.run_id]["artifacts"]
            record["status"] = "FINISHED"
    except Exception as error:
        record["status"] = "FAILED" if args.execute else "BLOCKED"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    lock_path.unlink()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] in {"READY", "FINISHED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
