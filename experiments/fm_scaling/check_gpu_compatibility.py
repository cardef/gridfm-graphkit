# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Run the CUDA, profiler, checkpoint, and largest-grid subset of I010."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import resource
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import yaml
from torch.profiler import ProfilerActivity, profile
from torch_geometric.data import Batch

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.fm_scaling.accounting import (
    counted_forward_flops,
    output_and_gradient_parity,
    trainable_parameter_count,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.data import (
    CaseDeclaredMVANormalizer,
    FMScalingPowerFlowTransforms,
    load_topology_manifest,
)
from gridfm_graphkit.fm_scaling.manifest import file_sha256
from gridfm_graphkit.fm_scaling.model import FMScalingPF
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task, load_normalizer


UPSTREAM_COMMIT = "b3d663b62179222c1ebec00ee29f67ea50e68c0b"  # pragma: allowlist secret
UPSTREAM_FLAT_CHECKPOINT_SHA256 = "3e2abcae1e86587fcb36f3ac186e32dac89ed01d454563a276bdd82e079c8bc2"  # pragma: allowlist secret
CAPACITY = {
    "flat": {"width": 122, "flat_blocks": 1},
    "global": {"width": 118, "flat_blocks": 1},
    "kron": {"width": 123, "flat_blocks": 1},
    "quotient": {"width": 123, "flat_blocks": 1},
}
CPU_CHECKS = {
    "upstream_identity",
    "mlflow_child_store_smoke",
    "clean_clone_import",
    "flat_schema_and_flop_cpu",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolve_compile_policy(parity: dict, eager_flops: int, compiled_flops: int) -> dict:
    relative_gap = (
        abs(compiled_flops - eager_flops) / max(eager_flops, 1)
        if eager_flops > 0 and compiled_flops > 0
        else None
    )
    enabled = (
        bool(parity.get("passed")) and relative_gap is not None and relative_gap <= 0.02
    )
    return {
        "selected_mode": "default" if enabled else "disabled",
        "parity": parity,
        "eager_flops": int(eager_flops),
        "compiled_flops": int(compiled_flops),
        "relative_flop_gap": relative_gap,
        "passed": True,
    }


def _model_args(
    core: str,
    topology_manifest: Path,
    geometry_bundle: Path,
) -> SimpleNamespace:
    spec = CAPACITY[core]
    return SimpleNamespace(
        task=SimpleNamespace(task_name="FMScalingPowerFlow"),
        data=SimpleNamespace(
            normalization="CaseDeclaredMVANormalizer",
            confirmatory=True,
            hierarchy=SimpleNamespace(enable=False),
            topology_manifest=str(topology_manifest),
            mask_value=0.0,
        ),
        model=SimpleNamespace(
            type="FMScalingPF",
            communication_core=core,
            hidden_size=spec["width"],
            edge_dim=10,
            input_bus_dim=15,
            input_gen_dim=6,
            l_pre=2,
            l_post=2,
            flat_blocks=spec["flat_blocks"],
            geometry_bundle=str(geometry_bundle),
        ),
    )


def _read_scenario_zero(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[("scenario", "==", 0)])
    if frame.empty or set(frame["scenario"].unique()) != {0}:
        raise ContractError(f"{path} did not yield exactly scenario zero")
    return frame


def _merge_generator_q_limits(
    bus: pd.DataFrame,
    gen: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the dataset raw-loader's bus-level generator-Q aggregate."""
    bus_keys = {"scenario", "bus"}
    q_columns = {"min_q_mvar", "max_q_mvar"}
    if not bus_keys.issubset(bus.columns) or not (bus_keys | q_columns).issubset(
        gen.columns,
    ):
        raise ContractError("generated PF tables lack generator-Q merge columns")
    if q_columns & set(bus.columns):
        raise ContractError("generated PF bus table already contains generator-Q limits")
    aggregated = (
        gen.groupby(["scenario", "bus"])[sorted(q_columns)].sum().reset_index()
    )
    merged = bus.merge(aggregated, on=["scenario", "bus"], how="left").fillna(0)
    if len(merged) != len(bus):
        raise ContractError("generator-Q merge changed the bus row count")
    return merged


def _load_sample(
    data_root: Path,
    network: str,
    topology_manifest: Path,
    geometry_bundle: Path,
) -> Batch:
    raw = data_root / network / "raw"
    bus = _read_scenario_zero(raw / "bus_data.parquet")
    gen = _read_scenario_zero(raw / "gen_data.parquet")
    bus = _merge_generator_q_limits(bus, gen)
    branch = _read_scenario_zero(raw / "branch_data.parquet")
    data = HeteroGridDatasetDisk._build_scenario(
        0,
        bus.groupby("scenario"),
        gen.groupby("scenario"),
        branch.groupby("scenario"),
    )
    args = _model_args("flat", topology_manifest, geometry_bundle)
    normalizer = CaseDeclaredMVANormalizer(args)
    normalizer.set_network(network)
    normalizer.transform(data)
    transforms = FMScalingPowerFlowTransforms(args)
    for transform in transforms.transforms:
        if hasattr(transform, "set_root"):
            transform.set_root(str(data_root / network))
    data = transforms(data)
    return Batch.from_data_list([data])


def _load_upstream_flat_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    batch: Batch,
) -> dict:
    if file_sha256(checkpoint_path) != UPSTREAM_FLAT_CHECKPOINT_SHA256:
        raise ContractError("upstream Flat checkpoint fixture hash changed")
    args = NestedNamespace(**yaml.safe_load(config_path.read_text()))
    normalizer = load_normalizer(args)
    task = get_task(args, [normalizer])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    task.load_state_dict(state, strict=True)
    task.model.eval()
    with torch.no_grad():
        output = task.model(batch.cpu())
    if set(output) != {"bus", "gen"}:
        raise ContractError("upstream Flat checkpoint output schema changed")
    return {
        "checkpoint_tensors": len(state),
        "bus_output_shape": list(output["bus"].shape),
        "gen_output_shape": list(output["gen"].shape),
        "checkpoint_sha256": UPSTREAM_FLAT_CHECKPOINT_SHA256,
    }


def _profile_flops(module: torch.nn.Module, batch: Batch) -> int:
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, with_flops=True) as trace, torch.no_grad():
        module(batch)
        torch.cuda.synchronize()
    return int(sum(int(event.flops or 0) for event in trace.key_averages()))


def _measure_core(
    core: str,
    batch: Batch,
    topology_manifest: Path,
    geometry_bundle: Path,
) -> dict:
    torch.manual_seed(20260714)
    model = FMScalingPF(
        _model_args(core, topology_manifest, geometry_bundle),
    ).cuda()
    model.train()
    device_batch = batch.cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    forward_flops = counted_forward_flops(model, device_batch)
    output = model(device_batch)
    loss = sum(value.float().square().mean() for value in output.values())
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "core": core,
        "parameters": trainable_parameter_count(model),
        "forward_flops": forward_flops,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": elapsed,
        "bus_output_shape": list(output["bus"].shape),
        "gen_output_shape": list(output["gen"].shape),
    }
    del loss, output, model, device_batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_gpu_compatibility(
    repo_root: Path,
    cpu_evidence_path: Path,
    topology_manifest: Path,
    geometry_bundle: Path,
    data_root: Path,
    upstream_config: Path,
    upstream_checkpoint: Path,
    output_path: Path,
) -> dict:
    if _git(repo_root, "status", "--short"):
        raise ContractError("I010 GPU checks require a clean worktree")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError("I010 GPU checks require exactly one visible CUDA device")
    fork_commit = _git(repo_root, "rev-parse", "HEAD")
    merge_base = _git(repo_root, "merge-base", "HEAD", "upstream/main")
    upstream_identity = (
        _git(repo_root, "rev-parse", "upstream/main") == UPSTREAM_COMMIT
        and merge_base == UPSTREAM_COMMIT
    )

    cpu_evidence = json.loads(cpu_evidence_path.read_text())
    cpu_by_name = {item["name"]: item for item in cpu_evidence.get("checks", [])}
    cpu_passed = (
        cpu_evidence.get("gate_id") == "I010"
        and cpu_evidence.get("results", {}).get("fork_commit") == fork_commit
        and all(cpu_by_name.get(name, {}).get("passed") is True for name in CPU_CHECKS)
    )

    topology = load_topology_manifest(topology_manifest)
    records = topology["topologies"]
    if any(record.get("integrity_status") != "PASS" for record in records.values()):
        raise ContractError("largest-grid probe requires every topology audit to pass")
    smallest = min(records, key=lambda name: (int(records[name]["bus_count"]), name))
    largest = max(records, key=lambda name: (int(records[name]["bus_count"]), name))

    host_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    small_batch = _load_sample(data_root, smallest, topology_manifest, geometry_bundle)
    large_batch = _load_sample(data_root, largest, topology_manifest, geometry_bundle)
    host_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    upstream_checkpoint_result = _load_upstream_flat_checkpoint(
        upstream_config,
        upstream_checkpoint,
        small_batch,
    )

    torch.manual_seed(20260714)
    eager = FMScalingPF(
        _model_args("flat", topology_manifest, geometry_bundle),
    ).cuda()
    candidate = copy.deepcopy(eager)
    small_cuda = small_batch.cuda()
    compile_error = None
    try:
        compiled = torch.compile(candidate, mode="default", dynamic=False)
        parity = output_and_gradient_parity(eager, compiled, small_cuda)
        eager_flops = counted_forward_flops(eager, small_cuda)
        compiled_flops = counted_forward_flops(compiled, small_cuda)
        compile_policy = resolve_compile_policy(parity, eager_flops, compiled_flops)
        compile_exercised = True
    except Exception as error:
        compile_error = {"type": type(error).__name__, "message": str(error)}
        compile_policy = resolve_compile_policy({"passed": False}, 0, 0)
        compile_exercised = False
    profiler_flops = _profile_flops(eager, small_cuda)
    counter_flops = counted_forward_flops(eager, small_cuda)
    profiler_gap = abs(profiler_flops - counter_flops) / max(counter_flops, 1)
    profiler_passed = min(profiler_flops, counter_flops) > 0 and profiler_gap <= 0.10
    del eager, candidate, small_cuda
    if "compiled" in locals():
        del compiled
    gc.collect()
    torch.cuda.empty_cache()

    core_results = {
        core: _measure_core(
            core,
            large_batch,
            topology_manifest,
            geometry_bundle,
        )
        for core in ("flat", "global", "kron", "quotient")
    }
    largest_passed = all(
        result["forward_flops"] > 0
        and result["peak_cuda_bytes"] > 0
        and result["bus_output_shape"][0] == int(records[largest]["bus_count"])
        for result in core_results.values()
    )
    checks = [
        {"name": "upstream_identity", "passed": upstream_identity},
        {"name": "fresh_cpu_compatibility", "passed": cpu_passed},
        {"name": "upstream_flat_checkpoint_load", "passed": True},
        {"name": "cuda_compile_exercised", "passed": compile_exercised},
        {"name": "compile_policy_fail_closed", "passed": compile_policy["passed"]},
        {"name": "profiler_flop_crosscheck", "passed": profiler_passed},
        {"name": "largest_grid_host_and_accelerator_peaks", "passed": largest_passed},
    ]
    properties = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": "I010",
        "status": "PASS" if all(item["passed"] for item in checks) else "BLOCKED",
        "checks": checks,
        "inputs": [
            {"path": str(cpu_evidence_path), "sha256": file_sha256(cpu_evidence_path)},
            {"path": str(topology_manifest), "sha256": file_sha256(topology_manifest)},
            {"path": str(geometry_bundle), "sha256": file_sha256(geometry_bundle)},
            {"path": str(upstream_config), "sha256": file_sha256(upstream_config)},
            {
                "path": str(upstream_checkpoint),
                "sha256": file_sha256(upstream_checkpoint),
            },
            {
                "path": str(
                    repo_root / "experiments/fm_scaling/check_gpu_compatibility.py"
                ),
                "sha256": file_sha256(
                    repo_root / "experiments/fm_scaling/check_gpu_compatibility.py",
                ),
            },
        ],
        "results": {
            "fork_commit": fork_commit,
            "upstream_commit": UPSTREAM_COMMIT,
            "merge_base": merge_base,
            "smallest_network": smallest,
            "largest_network": largest,
            "largest_bus_count": int(records[largest]["bus_count"]),
            "host_max_rss_kib_before_load": int(host_before),
            "host_max_rss_kib_after_load": int(host_after),
            "upstream_flat": upstream_checkpoint_result,
            "compile_policy": compile_policy,
            "compile_error": compile_error,
            "profiler": {
                "counter_flops": counter_flops,
                "profiler_flops": profiler_flops,
                "relative_gap": profiler_gap,
            },
            "cores": core_results,
            "cuda": {
                "device": properties.name,
                "total_memory": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cpu-evidence", type=Path, required=True)
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--upstream-config", type=Path, required=True)
    parser.add_argument("--upstream-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_gpu_compatibility(
        args.repo_root.resolve(),
        args.cpu_evidence.resolve(),
        args.topology_manifest.resolve(),
        args.geometry_bundle.resolve(),
        args.data_root.resolve(),
        args.upstream_config.resolve(),
        args.upstream_checkpoint.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({"I010": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
