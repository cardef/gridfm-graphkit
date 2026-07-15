# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed launcher for one explicit run from the frozen 20-run matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from experiments.fm_scaling.preflight import smoke_mlflow_store, validate_store_layout
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import load_campaign_manifest
from gridfm_graphkit.fm_scaling.manifest import file_sha256


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_command(
    manifest,
    run_id: str,
    python: Path,
) -> list[str]:
    try:
        spec = next(run for run in manifest.runs if run.run_id == run_id)
    except StopIteration as error:
        raise ContractError(f"run {run_id} is absent from campaign manifest") from error
    return [
        str(python),
        "-m",
        "gridfm_graphkit",
        "train",
        "--config",
        str(spec.config_path),
        "--plugins",
        "gridfm_graphkit.fm_scaling",
        "--exp_name",
        "fm-scaling-confirmatory",
        "--run_name",
        run_id,
        "--log_dir",
        str(manifest.mlflow_store),
        "--data_path",
        str(manifest.data_root),
        "--deterministic",
        "true",
        "--train_only",
    ]


def build_evaluation_commands(
    manifest,
    run_id: str,
    python: Path,
    repo_root: Path,
) -> list[tuple[str, Path, list[str]]]:
    spec = next(run for run in manifest.runs if run.run_id == run_id)
    config = yaml.safe_load(spec.config_path.read_text())
    thresholds = [int(value) for value in config["training"]["flop_checkpoints"]]
    if len(thresholds) != 3 or thresholds != sorted(set(thresholds)):
        raise ContractError("each run requires exactly three sorted FLOP checkpoints")
    checkpoint_dir = Path(config["training"]["flop_checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    labels = ("C/4", "C/2", "C")
    suffixes = ("C4", "C2", "C")
    commands = []
    for threshold, label, suffix in zip(thresholds, labels, suffixes):
        model_path = checkpoint_dir / f"flops_{threshold}.pt"
        output_path = manifest.result_root / f"{run_id}-{suffix}-metrics.json"
        command = [
            str(python),
            "-m",
            "gridfm_graphkit",
            "evaluate",
            "--config",
            str(spec.config_path),
            "--model_path",
            str(model_path),
            "--plugins",
            "gridfm_graphkit.fm_scaling",
            "--exp_name",
            "fm-scaling-confirmatory-evaluation",
            "--run_name",
            f"{run_id}-{suffix}",
            "--log_dir",
            str(manifest.mlflow_store),
            "--data_path",
            str(manifest.data_root),
            "--evaluation_checkpoint",
            label,
            "--evaluation_output",
            str(output_path),
            "--evaluation_targets",
        ]
        commands.append((label, model_path, command))
    return commands


def _training_artifacts(manifest, run_id: str, repo_root: Path) -> dict:
    """Validate and hash the complete first-crossing training evidence."""
    spec = next(run for run in manifest.runs if run.run_id == run_id)
    config = yaml.safe_load(spec.config_path.read_text())
    thresholds = [int(value) for value in config["training"]["flop_checkpoints"]]
    output_dir = Path(config["training"]["flop_checkpoint_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    ledger_path = output_dir / "flop_checkpoints.json"
    if not ledger_path.is_file():
        raise ContractError(f"missing FLOP ledger {ledger_path}")
    ledger = json.loads(ledger_path.read_text())
    crossed = ledger.get("crossed")
    if ledger.get("thresholds") != thresholds or not isinstance(crossed, list):
        raise ContractError("FLOP ledger does not match frozen thresholds")
    if [item.get("threshold") for item in crossed] != thresholds:
        raise ContractError("FLOP ledger lacks one first crossing per threshold")
    checkpoints = []
    for threshold, item in zip(thresholds, crossed):
        checkpoint = output_dir / f"flops_{threshold}.pt"
        if Path(str(item.get("checkpoint"))).resolve() != checkpoint.resolve():
            raise ContractError(f"FLOP ledger checkpoint mismatch for {threshold}")
        if not (
            int(item.get("previous_flops", threshold))
            < threshold
            <= int(item.get("crossing_flops", -1))
        ):
            raise ContractError(f"invalid first crossing for threshold {threshold}")
        if not checkpoint.is_file():
            raise ContractError(f"missing first-crossing checkpoint {checkpoint}")
        checkpoints.append(
            {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        )
    runtime_path = manifest.result_root / f"{run_id}-runtime.json"
    if not runtime_path.is_file():
        raise ContractError(f"missing runtime summary {runtime_path}")
    runtime = json.loads(runtime_path.read_text())
    if runtime.get("run_id") != run_id or runtime.get("status") != "TRAINED":
        raise ContractError(f"invalid runtime summary for {run_id}")
    return {
        "flop_ledger": {"path": str(ledger_path), "sha256": file_sha256(ledger_path)},
        "checkpoints": checkpoints,
        "runtime": {"path": str(runtime_path), "sha256": file_sha256(runtime_path)},
    }


def validate_launch(repo_root: Path, manifest) -> None:
    head = _git(repo_root, "rev-parse", "HEAD")
    if head != manifest.fork_commit:
        raise ContractError(f"fork commit mismatch: {head} != {manifest.fork_commit}")
    if _git(repo_root, "status", "--porcelain"):
        raise ContractError("confirmatory launch requires a clean worktree")
    upstream = _git(repo_root, "rev-parse", "upstream/main")
    merge_base = _git(repo_root, "merge-base", "HEAD", "upstream/main")
    if upstream != manifest.upstream_commit or merge_base != manifest.merge_base:
        raise ContractError("upstream reference or merge base differs from manifest")
    layout = validate_store_layout(repo_root / "mlruns", manifest.mlflow_store)
    if not all(check["passed"] for check in layout):
        raise ContractError(f"MLflow store layout failed: {layout}")
    smoke = smoke_mlflow_store(manifest.mlflow_store)
    if not smoke["passed"]:
        raise ContractError(f"MLflow store smoke failed: {smoke['detail']}")
    if not torch.cuda.is_available():
        raise ContractError(
            "confirmatory launch requires an available CUDA accelerator",
        )


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
        f"{args.run_id}-launch.json"
        if args.execute
        else f"{args.run_id}-launch-ready-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
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
        spec = next(run for run in manifest.runs if run.run_id == args.run_id)
        record["provenance"] = {
            "campaign_manifest": str(manifest.path),
            "campaign_sha256": file_sha256(manifest.path),
            "fork_commit": manifest.fork_commit,
            "upstream_commit": manifest.upstream_commit,
            "merge_base": manifest.merge_base,
            "config_sha256": spec.config_sha256,
            "topology_manifest_sha256": file_sha256(manifest.topology_manifest),
            "geometry_bundle_sha256": file_sha256(manifest.geometry_bundle),
            "split_manifest_sha256": file_sha256(manifest.split_manifest),
        }
        validate_launch(repo_root, manifest)
        command = build_command(manifest, args.run_id, args.python.resolve())
        record.update(
            {
                "status": "READY",
                "command": command,
            },
        )
        if args.execute:
            config = yaml.safe_load(spec.config_path.read_text())
            checkpoint_dir = repo_root / config["training"]["flop_checkpoint_dir"]
            runtime_path = repo_root / config["training"]["runtime_output_path"]
            if runtime_path.exists():
                raise ContractError("refusing to overwrite prior training artifacts")
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            environment = os.environ.copy()
            environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            started = time.monotonic()
            result = subprocess.run(
                command,
                cwd=repo_root,
                check=False,
                env=environment,
            )
            record["wall_seconds"] = time.monotonic() - started
            record["train_return_code"] = result.returncode
            if result.returncode != 0:
                raise ContractError(f"training process failed with {result.returncode}")
            record["artifacts"] = _training_artifacts(
                manifest,
                args.run_id,
                repo_root,
            )
            record["environment"] = {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "cublas_workspace_config": environment["CUBLAS_WORKSPACE_CONFIG"],
            }
            record["status"] = "TRAINED"
    except Exception as error:
        record["status"] = "FAILED" if args.execute else "BLOCKED"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    lock_path.unlink()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] in {"READY", "TRAINED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
