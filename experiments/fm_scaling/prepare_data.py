# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare immutable datakit configs and a draft topology manifest.

The command deliberately separates preparation from generation.  Its default
mode only writes reviewed configs.  ``--execute`` invokes the shared virtual
environment after proving that ``gridfm-datakit`` is an editable install of the
exact sibling research fork and clean pinned commit.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

from experiments.fm_scaling.datakit_topology import (
    TOPOLOGY_PREPROCESSING_POLICY,
    normalize_datakit_network,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError, SCHEMA_VERSION
from gridfm_graphkit.fm_scaling.manifest import file_sha256


INVENTORY_SCHEMA = "fm-scaling-data-inventory-v1"
_CASE_FIELDS = {
    "network",
    "topology_key",
    "source",
    "network_dir",
    "provenance_group",
    "split",
    "scenarios",
    "seed",
}


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _editable_root() -> tuple[Path | None, bool]:
    distribution = importlib.metadata.distribution("gridfm-datakit")
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return None, False
    direct_url = json.loads(raw)
    parsed = urlparse(str(direct_url.get("url", "")))
    if parsed.scheme != "file":
        return None, False
    return (
        Path(unquote(parsed.path)).resolve(),
        direct_url.get("dir_info", {}).get("editable") is True,
    )


def assert_datakit_checkout(repo_root: Path, expected_commit: str) -> Path:
    """Fail unless the shared env resolves to the exact clean sibling fork."""
    expected_root = (repo_root.parent / "gridfm-datakit").resolve()
    expected_env = (repo_root.parent / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_env:
        raise ContractError(
            f"run with shared environment {expected_env}; active prefix={sys.prefix}",
        )
    editable_root, editable = _editable_root()
    if not editable or editable_root != expected_root:
        raise ContractError(
            "gridfm-datakit must be editable from the exact sibling checkout; "
            f"expected={expected_root}; observed={editable_root}; editable={editable}",
        )
    git_root = Path(_run(["git", "rev-parse", "--show-toplevel"], expected_root))
    commit = _run(["git", "rev-parse", "HEAD"], expected_root)
    dirty = _run(["git", "status", "--short"], expected_root)
    origin = _run(["git", "remote", "get-url", "origin"], expected_root)
    if git_root.resolve() != expected_root:
        raise ContractError(f"datakit git root is {git_root}, expected {expected_root}")
    if origin != "git@github.com:cardef/gridfm-datakit.git":
        raise ContractError(f"datakit origin is not the research fork: {origin}")
    if commit != expected_commit:
        raise ContractError(f"datakit commit mismatch: {commit} != {expected_commit}")
    if dirty:
        raise ContractError(f"datakit checkout is dirty:\n{dirty}")
    return expected_root


def validate_inventory(payload: dict) -> tuple[str, list[dict]]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != INVENTORY_SCHEMA
    ):
        raise ContractError("data inventory has the wrong schema")
    expected_commit = str(payload.get("datakit_commit", ""))
    if len(expected_commit) != 40:
        raise ContractError("data inventory requires a full datakit_commit")
    design = payload.get("design")
    design_fields = {
        "source_scenarios_per_topology",
        "source_dev_scenarios_per_topology",
        "target_scenarios_per_topology",
        "seed_rule",
        "r002_sha256",
    }
    if not isinstance(design, dict) or set(design) != design_fields:
        raise ContractError("data inventory requires the exact frozen design fields")
    if design["seed_rule"] != "20260714_plus_frozen_case_index":
        raise ContractError("data inventory has the wrong seed rule")
    if len(str(design["r002_sha256"])) != 64:
        raise ContractError("data inventory requires an R002 SHA-256")
    expected_scenarios = {
        "source": int(design["source_scenarios_per_topology"]),
        "source_dev": int(design["source_dev_scenarios_per_topology"]),
        "target": int(design["target_scenarios_per_topology"]),
    }
    if min(expected_scenarios.values()) < 1:
        raise ContractError("frozen scenario pool sizes must be positive")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("data inventory requires a nonempty cases list")
    networks: set[str] = set()
    topology_keys: set[str] = set()
    split_counts = {"source": 0, "source_dev": 0, "target": 0}
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) - _CASE_FIELDS:
            raise ContractError("data inventory case has unknown fields")
        required = {
            "network",
            "topology_key",
            "source",
            "provenance_group",
            "split",
            "scenarios",
            "seed",
        }
        if required - set(case):
            raise ContractError(
                f"data inventory case misses {sorted(required - set(case))}",
            )
        network = str(case["network"])
        topology_key = str(case["topology_key"])
        if network in networks or topology_key in topology_keys:
            raise ContractError("network and topology_key values must be unique")
        networks.add(network)
        topology_keys.add(topology_key)
        if case["source"] not in {"pglib", "file"}:
            raise ContractError(f"{network} has invalid source")
        if case["source"] == "file" and not case.get("network_dir"):
            raise ContractError(f"{network} requires network_dir")
        split = str(case["split"])
        if split not in split_counts:
            raise ContractError(f"{network} has invalid split")
        split_counts[split] += 1
        if int(case["scenarios"]) != expected_scenarios[split]:
            raise ContractError(f"{network} differs from the frozen scenario pool")
        if int(case["seed"]) != 20260714 + case_index:
            raise ContractError(f"{network} differs from the frozen seed rule")
    if split_counts != {"source": 26, "source_dev": 2, "target": 27}:
        raise ContractError(f"data inventory has wrong split counts: {split_counts}")
    return expected_commit, cases


def render_datakit_config(case: dict, data_root: Path, workers: int) -> dict:
    """Render the only data-generating treatment allowed by the contract."""
    load = {
        "generator": "agg_load_profile",
        "agg_profile": "default",
        "scenarios": int(case["scenarios"]),
        "sigma": 0.2,
        "change_reactive_power": True,
        "global_range": 0.4,
        "max_scaling_factor": 4.0,
        "step_size": 0.1,
        "start_scaling_factor": 1.0,
    }
    network = {
        "name": str(case["network"]),
        "source": str(case["source"]),
        "reader": "native",
    }
    if case["source"] == "file":
        network["network_dir"] = str(Path(case["network_dir"]).resolve())
    return {
        "network": network,
        "load": load,
        "topology_perturbation": {"type": "none"},
        "generation_perturbation": {"type": "none"},
        "admittance_perturbation": {"type": "none"},
        "settings": {
            "num_processes": int(workers),
            "data_dir": str(data_root.resolve()),
            "large_chunk_size": 1000,
            "overwrite": True,
            "mode": "pf",
            "include_dc_res": False,
            "enable_solver_logs": False,
            "pf_fast": True,
            "dcpf_fast": True,
            "pf_solver": "powermodel",
            "opf_formulation": "polar",
            "max_iter": 200,
            "seed": int(case["seed"]),
        },
    }


def _network_metadata(case: dict) -> tuple[float, int]:
    from gridfm_datakit.network import load_net_from_file, load_net_from_pglib

    if case["source"] == "pglib":
        network = load_net_from_pglib(str(case["network"]))
    else:
        path = Path(case["network_dir"]).resolve() / f"{case['network']}.m"
        network = load_net_from_file(str(path))
    network, _ = normalize_datakit_network(network)
    return float(network.baseMVA), int(network.buses.shape[0])


def prepare(
    inventory_path: Path,
    repo_root: Path,
    config_dir: Path,
    data_root: Path,
    manifest_path: Path,
    workers: int,
    execute: bool,
) -> dict:
    payload = yaml.safe_load(inventory_path.read_text())
    expected_commit, cases = validate_inventory(payload)
    assert_datakit_checkout(repo_root, expected_commit)
    config_dir.mkdir(parents=True, exist_ok=True)
    topologies = {}
    for case in cases:
        network = str(case["network"])
        base_mva, bus_count = _network_metadata(case)
        config = render_datakit_config(case, data_root, workers)
        config_path = config_dir / f"{network}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        topologies[network] = {
            "topology_key": str(case["topology_key"]),
            "baseMVA": base_mva,
            "provenance_group": str(case["provenance_group"]),
            "split": str(case["split"]),
            "bus_count": bus_count,
            "scenario_count": int(case["scenarios"]),
            "datakit_commit": expected_commit,
            "config_sha256": file_sha256(config_path),
            "config_path": (
                str(config_path.resolve().relative_to(repo_root))
                if repo_root in config_path.resolve().parents
                else str(config_path.resolve())
            ),
            "integrity_status": "PENDING",
            "generation_provenance_path": str(
                (config_dir / f"{network}.provenance.json").resolve(),
            ),
        }
        if execute:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.fm_scaling.datakit_generate",
                    "--config",
                    str(config_path),
                    "--repo-root",
                    str(repo_root),
                    "--expected-commit",
                    expected_commit,
                    "--inventory-sha256",
                    file_sha256(inventory_path),
                    "--output",
                    str(config_dir / f"{network}.provenance.json"),
                ],
                cwd=repo_root,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gridfm_datakit.cli",
                    "validate",
                    str(data_root / network / "raw"),
                    "--n-partitions",
                    "0",
                    "--mode",
                    "pf",
                    "--sn-mva",
                    str(base_mva),
                ],
                cwd=repo_root,
                check=True,
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_freeze": {
            "rule": "inventory_split_before_generation",
            "inventory_path": str(inventory_path.resolve()),
            "inventory_sha256": file_sha256(inventory_path),
            "topology_preprocessing": TOPOLOGY_PREPROCESSING_POLICY,
        },
        "topologies": topologies,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.workers <= 0:
        raise ContractError("workers must be positive")
    prepare(
        args.inventory.resolve(),
        args.repo_root.resolve(),
        args.config_dir.resolve(),
        args.data_root.resolve(),
        args.manifest.resolve(),
        args.workers,
        args.execute,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
