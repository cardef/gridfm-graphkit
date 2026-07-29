# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Run the treatment-blind Flat throughput probe and freeze R005 C_cal."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from experiments.fm_scaling.check_gpu_compatibility import _load_sample, _model_args
from gridfm_graphkit.fm_scaling.calibration import freeze_calibration_budget
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.loss import (
    GraphBalancedMaskedVMVA,
    GraphBalancedPBE,
)
from gridfm_graphkit.fm_scaling.manifest import file_sha256
from gridfm_graphkit.fm_scaling.model import FMScalingPF


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _objective(model, batch, data_loss, physics_loss):
    output = model(batch)
    masks = dict(batch.mask_dict)
    masks["_bus_batch"] = batch.batch_dict["bus"]
    masks["_num_graphs"] = int(batch.num_graphs)
    positional = (
        output,
        batch.y_dict,
        batch.edge_index_dict,
        batch.edge_attr_dict,
        masks,
    )
    data = data_loss(*positional, model=model, x_dict=batch.x_dict)["loss"]
    physics = physics_loss(*positional, model=model, x_dict=batch.x_dict)["loss"]
    return 0.5 * data + 0.5 * physics


def _step(model, batch, optimizer, data_loss, physics_loss) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = _objective(model, batch, data_loss, physics_loss)
    loss.backward()
    optimizer.step()


def _count_training_step(model, batch, optimizer, data_loss, physics_loss) -> int:
    with FlopCounterMode(display=False) as counter:
        _step(model, batch, optimizer, data_loss, physics_loss)
        torch.cuda.synchronize()
    flops = int(counter.get_total_flops())
    if flops <= 0:
        raise ContractError("R005 training step produced no counted FLOPs")
    return flops


def _load_policy(policy_path: Path, loss_candidates_path: Path) -> tuple[dict, list]:
    policy = yaml.safe_load(policy_path.read_text())
    required_policy = {
        "schema_version": "fm-scaling-r005-policy-v1",
        "gate_id": "R005",
        "communication_core": "flat",
        "allowed_split": "source_dev",
        "seed": 20260714,
        "compile_mode": "default",
        "timing_quantile": 0.95,
        "quantile_method": "higher",
        "budget_rounding": "floor_two_significant_digits",
    }
    if any(policy.get(key) != value for key, value in required_policy.items()):
        raise ContractError("R005 policy differs from the preregistered constants")
    candidates = yaml.safe_load(loss_candidates_path.read_text())
    candidate_rows = candidates.get("candidates", [])
    if (
        candidates.get("schema_version") != "fm-scaling-loss-candidates-v1"
        or int(candidates.get("candidate_count", 0)) != len(candidate_rows)
        or [row.get("id") for row in candidate_rows] != ["C001", "C002", "C003"]
        or any(
            len(row.get("loss_weights", [])) != 2
            or min(float(value) for value in row["loss_weights"]) <= 0
            for row in candidate_rows
        )
    ):
        raise ContractError("R005 requires the three preregistered loss candidates")
    return policy, candidate_rows


def run_probe(
    repo_root: Path,
    topology_manifest: Path,
    geometry_bundle: Path,
    data_root: Path,
    policy_path: Path,
    loss_candidates_path: Path,
    output_path: Path,
) -> dict:
    if _git(repo_root, "status", "--short"):
        raise ContractError("R005 requires a clean worktree")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError("R005 requires exactly one visible CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA A40":
        raise ContractError("R005 authoritative throughput probe requires NVIDIA A40")
    policy, candidate_rows = _load_policy(policy_path, loss_candidates_path)
    warmup_steps = int(policy["warmup_steps"])
    measured_steps = int(policy["measured_steps"])
    if warmup_steps < 1 or measured_steps < 3:
        raise ContractError("R005 requires warmup and at least three measured steps")

    topology = load_topology_manifest(topology_manifest)
    networks = sorted(
        network
        for network, record in topology["topologies"].items()
        if record["split"] == "source_dev"
    )
    if not networks or any(
        topology["topologies"][network].get("integrity_status") != "PASS"
        for network in networks
    ):
        raise ContractError("R005 requires every audited source-development topology")

    args = _model_args("flat", topology_manifest, geometry_bundle)
    torch.manual_seed(int(policy["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = FMScalingPF(args).cuda().train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        betas=(0.9, 0.999),
    )
    data_loss = GraphBalancedMaskedVMVA(SimpleNamespace(), args).cuda()
    physics_loss = GraphBalancedPBE(SimpleNamespace(), args).cuda()
    batches = {
        network: _load_sample(
            data_root,
            network,
            topology_manifest,
            geometry_bundle,
        ).cuda()
        for network in networks
    }
    counted_flops = {
        network: _count_training_step(
            model,
            batch,
            optimizer,
            data_loss,
            physics_loss,
        )
        for network, batch in batches.items()
    }

    compiled = torch.compile(model, mode=str(policy["compile_mode"]), dynamic=False)
    measurements = []
    for network, batch in batches.items():
        started = time.perf_counter()
        for _ in range(warmup_steps):
            _step(compiled, batch, optimizer, data_loss, physics_loss)
        torch.cuda.synchronize()
        compile_and_warmup_seconds = time.perf_counter() - started
        timings = []
        for _ in range(measured_steps):
            started = time.perf_counter()
            _step(compiled, batch, optimizer, data_loss, physics_loss)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
        measurements.append(
            {
                "network": network,
                "split": "source_dev",
                "communication_core": "flat",
                "bus_count": int(topology["topologies"][network]["bus_count"]),
                "training_step_flops": counted_flops[network],
                "step_seconds": timings,
                "compile_and_warmup_seconds": compile_and_warmup_seconds,
            },
        )

    budget = freeze_calibration_budget(
        measurements,
        candidate_count=int(policy["candidate_count"]),
        aggregate_hours=float(policy["aggregate_gpu_hours"]),
        nontraining_reserve_seconds=float(
            policy["nontraining_reserve_seconds_per_candidate"],
        ),
        throughput_guard=float(policy["throughput_guard"]),
    )
    fork_commit = _git(repo_root, "rev-parse", "HEAD")
    payload = {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": "R005",
        "status": "PASS",
        "checks": [
            {"name": "r005_criteria", "passed": True},
            {"name": "immutable_inputs", "passed": True},
            {"name": "flat_only", "passed": True},
            {"name": "source_development_only", "passed": True},
            {"name": "three_hour_aggregate_upper_bound", "passed": True},
        ],
        "inputs": [
            {
                "path": str(topology_manifest),
                "sha256": file_sha256(topology_manifest),
            },
            {
                "path": str(geometry_bundle),
                "sha256": file_sha256(geometry_bundle),
            },
            {
                "path": str(policy_path),
                "sha256": file_sha256(policy_path),
            },
            {
                "path": str(loss_candidates_path),
                "sha256": file_sha256(loss_candidates_path),
            },
            {
                "path": str(repo_root / "experiments/fm_scaling/probe_calibration.py"),
                "sha256": file_sha256(
                    repo_root / "experiments/fm_scaling/probe_calibration.py",
                ),
            },
        ],
        "results": {
            "fork_commit": fork_commit,
            "upstream_commit": _git(repo_root, "rev-parse", "upstream/main"),
            "merge_base": _git(repo_root, "merge-base", "HEAD", "upstream/main"),
            "communication_core": "flat",
            "compile_mode": str(policy["compile_mode"]),
            "seed": int(policy["seed"]),
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "networks": networks,
            "loss_candidates": candidate_rows,
            "target_outputs_read": False,
            "budget": budget,
            "device": device_name,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--loss-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_probe(
        args.repo_root.resolve(),
        args.topology_manifest.resolve(),
        args.geometry_bundle.resolve(),
        args.data_root.resolve(),
        args.policy.resolve(),
        args.loss_candidates.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({"R005": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
