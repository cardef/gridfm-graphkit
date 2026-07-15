# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the locked C1/C2 decision from explicit campaign artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.fm_scaling.evaluate_campaign import _sealed_training_matrix
from gridfm_graphkit.fm_scaling.analysis import (
    aggregate_scenarios,
    average_seeds,
    exact_sign_flip_pvalue,
    exact_upper_bound,
    provenance_group_contrasts,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.manifest import file_sha256, load_campaign_manifest


CHECKPOINT_FILES = {"C/4": "C4", "C/2": "C2", "C": "C"}


def _load_records(manifest, topology_payload: dict, repo_root: Path) -> list[dict]:
    expected_scenarios = {
        record["topology_key"]: set(range(int(record["scenario_count"])))
        for network, record in topology_payload["topologies"].items()
        if record["split"] == "target" and network not in manifest.geometry_failures
    }
    records = []
    training_records = _sealed_training_matrix(manifest, repo_root)
    campaign_hash = file_sha256(manifest.path)
    for run in manifest.runs:
        launch_path = manifest.result_root / f"{run.run_id}-launch.json"
        launch = json.loads(launch_path.read_text())
        if launch.get("status") != "TRAINED":
            raise ContractError(f"{run.run_id} training is not sealed")
        evaluation_path = manifest.result_root / f"{run.run_id}-evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        if evaluation.get("status") != "FINISHED":
            raise ContractError(f"{run.run_id} evaluation is not FINISHED")
        if evaluation.get("campaign_sha256") != campaign_hash:
            raise ContractError(f"{run.run_id} evaluation belongs to another campaign")
        if (
            evaluation.get("training_artifacts")
            != training_records[run.run_id]["artifacts"]
        ):
            raise ContractError(
                f"{run.run_id} evaluation used changed training artifacts",
            )
        metric_hashes = {
            Path(item["path"]).resolve(): item["sha256"]
            for item in evaluation.get("metrics", [])
        }
        for checkpoint, suffix in CHECKPOINT_FILES.items():
            path = manifest.result_root / f"{run.run_id}-{suffix}-metrics.json"
            if metric_hashes.get(path.resolve()) != file_sha256(path):
                raise ContractError(f"{path} changed after evaluation sealing")
            payload = json.loads(path.read_text())
            if not isinstance(payload, list):
                raise ContractError(f"{path} is not a record list")
            selected: dict[tuple[str, int], dict] = {}
            for record in payload:
                identity = (
                    record.get("run_id"),
                    record.get("system"),
                    record.get("g_level"),
                    int(record.get("seed", -1)),
                    record.get("checkpoint"),
                )
                expected_identity = (
                    run.run_id,
                    run.core,
                    run.g_level,
                    run.seed,
                    checkpoint,
                )
                if identity != expected_identity:
                    raise ContractError(
                        f"{path} contains an identity mismatch {identity}",
                    )
                topology_key = str(record["topology_key"])
                if topology_key not in expected_scenarios:
                    raise ContractError(
                        f"{path} contains non-target topology {topology_key}",
                    )
                key = (topology_key, int(record["scenario_id"]))
                if key in selected:
                    raise ContractError(f"{path} duplicates {key}")
                selected[key] = record
            expected = {
                (topology_key, scenario)
                for topology_key, scenarios in expected_scenarios.items()
                for scenario in scenarios
            }
            if set(selected) != expected:
                missing = sorted(expected - set(selected))
                raise ContractError(
                    f"{path} omits target scenarios; first missing={missing[:10]}",
                )
            records.extend(selected.values())
    return records


def _decision(groups: dict[str, dict[str, float]]) -> dict:
    errors = [record["error"] for record in groups.values()]
    residuals = [record["residual"] for record in groups.values()]
    return {
        "group_count": len(groups),
        "groups": groups,
        "error_point": float(np.mean(errors)),
        "error_pvalue": exact_sign_flip_pvalue(errors),
        "error_upper95": exact_upper_bound(errors),
        "residual_point": float(np.mean(residuals)),
        "residual_pvalue_at_log_1p05": exact_sign_flip_pvalue(
            residuals,
            math.log(1.05),
        ),
        "residual_upper95": exact_upper_bound(residuals),
        "wild_cluster_bootstrap_t": _wild_cluster_bootstrap_t(errors),
    }


def _wild_cluster_bootstrap_t(
    values: list[float],
    draws: int = 9_999,
    seed: int = 20260714,
) -> dict:
    """Fixed-seed Rademacher wild-cluster bootstrap-t sensitivity."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ContractError("wild bootstrap requires at least two groups")
    standard_error = array.std(ddof=1) / math.sqrt(array.size)
    observed = array.mean() / max(standard_error, 1e-15)
    centered = array - array.mean()
    rng = np.random.Generator(np.random.PCG64(seed))
    signs = rng.choice((-1.0, 1.0), size=(draws, array.size))
    bootstrap = signs * centered
    denominators = bootstrap.std(axis=1, ddof=1) / math.sqrt(array.size)
    statistics = bootstrap.mean(axis=1) / np.maximum(denominators, 1e-15)
    return {
        "draws": draws,
        "seed": seed,
        "t": float(observed),
        "one_sided_pvalue": float((1 + np.sum(statistics <= observed)) / (draws + 1)),
    }


def _diagnostics(scenarios: list[dict], topology_records: list[dict], manifest) -> dict:
    component = defaultdict(list)
    tails = defaultdict(list)
    for record in scenarios:
        key = f"{record['system']}:{record['g_level']}:{record['checkpoint']}"
        component[key].append((record["rmse_vm_pu"], record["rmse_va_rad"]))
        tails[key].append(record["family_balanced_error"])
    seed_pairs = defaultdict(dict)
    for record in topology_records:
        key = (
            record["system"],
            record["g_level"],
            record["checkpoint"],
            record["topology_key"],
        )
        seed_pairs[key][int(record["seed"])] = record["family_balanced_error"]
    runtime = []
    for run in manifest.runs:
        launch = json.loads(
            (manifest.result_root / f"{run.run_id}-launch.json").read_text(),
        )
        runtime_record = launch["artifacts"]["runtime"]
        runtime_path = Path(runtime_record["path"])
        if file_sha256(runtime_path) != runtime_record["sha256"]:
            raise ContractError(f"{run.run_id} runtime artifact changed")
        summary = json.loads(runtime_path.read_text())
        runtime.append(
            {
                "run_id": run.run_id,
                "system": run.core,
                "wall_seconds": summary["wall_seconds"],
                "gpu_hours": summary["gpu_hours"],
                "peak_cuda_bytes": summary["peak_cuda_bytes"],
            },
        )
    return {
        "component_means": {
            key: {
                "rmse_vm_pu": float(np.mean([item[0] for item in values])),
                "rmse_va_rad": float(np.mean([item[1] for item in values])),
            }
            for key, values in sorted(component.items())
        },
        "scenario_p95": {
            key: float(np.quantile(values, 0.95))
            for key, values in sorted(tails.items())
        },
        "seed_dispersion": {
            "mean_absolute_difference": float(
                np.mean([abs(values[0] - values[1]) for values in seed_pairs.values()]),
            ),
            "pair_count": len(seed_pairs),
        },
        "systems": runtime,
        "failure_accounting": {
            "construction": len(manifest.geometry_failures),
            "training": 0,
            "evaluation": 0,
            "construction_records": manifest.geometry_failures,
            "note": "missing training/evaluation terminal artifacts make analysis INVALID",
        },
    }


def _point_estimate(
    averaged: list[dict],
    metadata: dict[str, dict],
    treatment: str,
    baseline: str,
    g_level: str,
    checkpoint: str,
    topology_keys: set[str] | None = None,
) -> float:
    selected = {
        (record["system"], record["topology_key"]): record
        for record in averaged
        if record["g_level"] == g_level and record["checkpoint"] == checkpoint
    }
    by_group: dict[str, list[float]] = defaultdict(list)
    keys = topology_keys or {
        key for key, record in metadata.items() if record["split"] == "target"
    }
    for key in sorted(keys):
        treatment_value = selected[(treatment, key)]["family_balanced_error"]
        baseline_value = selected[(baseline, key)]["family_balanced_error"]
        by_group[metadata[key]["provenance_group"]].append(
            math.log(max(treatment_value, 1e-12))
            - math.log(max(baseline_value, 1e-12)),
        )
    return float(np.mean([np.mean(values) for values in by_group.values()]))


def analyze(manifest_path: Path, repo_root: Path) -> dict:
    manifest = load_campaign_manifest(manifest_path, repo_root)
    topology_payload = load_topology_manifest(manifest.topology_manifest)
    metadata = {
        record["topology_key"]: {**record, "network": network}
        for network, record in topology_payload["topologies"].items()
    }
    for network in manifest.geometry_failures:
        key = topology_payload["topologies"][network]["topology_key"]
        metadata[key]["split"] = "failed_target"
    scenario_records = _load_records(manifest, topology_payload, repo_root)
    topology_records = aggregate_scenarios(scenario_records)
    averaged = average_seeds(topology_records)

    comparisons = {}
    for baseline in ("flat", "global"):
        groups = provenance_group_contrasts(
            averaged,
            metadata,
            treatment="kron",
            baseline=baseline,
            g_level="G28",
            checkpoint="C",
        )
        comparisons[f"kron_vs_{baseline}"] = _decision(groups)
    quotient_groups = provenance_group_contrasts(
        averaged,
        metadata,
        treatment="kron",
        baseline="quotient",
        g_level="G28",
        checkpoint="C",
    )
    comparisons["kron_vs_quotient"] = _decision(quotient_groups)

    point_estimates = {}
    for baseline in ("flat", "global"):
        key = f"kron_vs_{baseline}"
        point_estimates[key] = {
            "checkpoints": {
                checkpoint: _point_estimate(
                    averaged,
                    metadata,
                    "kron",
                    baseline,
                    "G28",
                    checkpoint,
                )
                for checkpoint in CHECKPOINT_FILES
            },
            "diversity": {
                level: _point_estimate(averaged, metadata, "kron", baseline, level, "C")
                for level in ("G8", "G16", "G28")
            },
            "size_terciles": {
                tercile: _point_estimate(
                    averaged,
                    metadata,
                    "kron",
                    baseline,
                    "G28",
                    "C",
                    {
                        topology_key
                        for topology_key, record in metadata.items()
                        if record.get("size_tercile") == tercile
                        and record.get("split") == "target"
                    },
                )
                for tercile in ("smallest", "middle", "largest")
            },
        }

    c1_comparisons = [comparisons["kron_vs_flat"], comparisons["kron_vs_global"]]
    c1 = (
        not manifest.geometry_failures
        and all(item["error_upper95"] < 0 for item in c1_comparisons)
        and all(item["residual_upper95"] <= math.log(1.05) for item in c1_comparisons)
        and all(
            values["checkpoints"][checkpoint] < 0
            for values in point_estimates.values()
            for checkpoint in ("C/2", "C")
        )
        and all(
            values["diversity"]["G8"] < 0
            and values["diversity"]["G28"] <= values["diversity"]["G8"]
            for values in point_estimates.values()
        )
        and all(
            max(values["size_terciles"].values()) < 0
            and values["size_terciles"]["largest"]
            <= values["size_terciles"]["smallest"]
            for values in point_estimates.values()
        )
    )
    c2_record = comparisons["kron_vs_quotient"]
    c2 = (
        not manifest.geometry_failures
        and c2_record["error_upper95"] < 0
        and c2_record["residual_upper95"] <= math.log(1.05)
    )
    return {
        "schema_version": "fm-scaling-analysis-v1",
        "status": "PASS" if c1 and c2 else "FAIL",
        "claims": {"C1": "PASS" if c1 else "FAIL", "C2": "PASS" if c2 else "FAIL"},
        "comparisons": comparisons,
        "point_estimates": point_estimates,
        "diagnostics": _diagnostics(scenario_records, topology_records, manifest),
        "coverage": {
            "scenario_records": len(scenario_records),
            "topology_seed_records": len(topology_records),
            "topology_records": len(averaged),
            "runs": len(manifest.runs),
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = analyze(args.manifest.resolve(), args.repo_root.resolve())
    except Exception as error:
        result = {
            "schema_version": "fm-scaling-analysis-v1",
            "status": "INVALID",
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"PASS", "FAIL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
