# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Run datakit generation and emit provenance from that exact process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml


# Datakit's feasibility-bound worker uses multiprocessing.Pool. Force spawn
# before Julia starts: forking the initialized multithreaded Julia runtime
# deadlocks both parent and child on Abacus.
multiprocessing.set_start_method("spawn", force=True)

# JuliaCall must initialize before GraphKit imports Torch; the reverse order is a
# documented segfault risk in mixed Julia/PyTorch processes.
import juliacall  # noqa: F401
from experiments.fm_scaling.datakit_retry import (
    RETRY_CANDIDATE_POLICY,
    RETRY_POLICY,
    RETRY_STATE_FILE,
    complete_pf_pool,
    retry_candidate_indices,
    successful_scenario_count,
)

from experiments.fm_scaling.datakit_topology import (
    TOPOLOGY_PREPROCESSING_POLICY,
    normalize_datakit_network,
)
from experiments.fm_scaling.prepare_data import assert_datakit_checkout
from gridfm_graphkit.fm_scaling.manifest import file_sha256


_CHUNK_SEED_SHIM = "GRIDFM_DATAKIT_UINT32_CHUNK_SEED"
_TOPOLOGY_PREPROCESSING: dict[str, list[int]] = {}


def _install_uint32_chunk_seed_shim() -> None:
    """Bound Datakit's derived chunk seed without changing frozen base seeds."""
    from gridfm_datakit.process import process_network

    current = process_network.custom_seed
    if getattr(current, "_gridfm_uint32_chunk_seed", False):
        return

    class Uint32ChunkSeed(current):
        _gridfm_uint32_chunk_seed = True

        def __init__(self, seed=None):
            bounded = None if seed is None else int(seed) % (2**32)
            super().__init__(bounded)

    process_network.custom_seed = Uint32ChunkSeed


def _install_topology_preprocessing_shim() -> None:
    """Normalize declared inert type-4 buses before scenario generation."""
    from gridfm_datakit import generate

    current = generate.load_net_from_pglib
    if getattr(current, "_gridfm_topology_preprocessing", False):
        return

    def load_net_from_pglib(grid_name: str):
        network, dropped = normalize_datakit_network(current(grid_name))
        _TOPOLOGY_PREPROCESSING[grid_name] = dropped
        return network

    load_net_from_pglib._gridfm_topology_preprocessing = True
    generate.load_net_from_pglib = load_net_from_pglib


def _install_retry_candidate_shim(target_scenarios: int) -> None:
    """Keep retries on the frozen full aggregate-profile horizon."""
    from gridfm_datakit import generate

    current = generate.get_load_scenario_generator
    if getattr(current, "_gridfm_retry_candidate_policy", None) is not None:
        return

    def get_load_scenario_generator(args):
        base_generator = current(args)

        def generate_candidates(net, n_scenarios, scenarios_log, max_iter, seed):
            if n_scenarios == target_scenarios:
                return base_generator(
                    net,
                    n_scenarios,
                    scenarios_log,
                    max_iter,
                    seed,
                )
            full_horizon = base_generator(
                net,
                target_scenarios,
                scenarios_log,
                max_iter,
                seed,
            )
            indices = retry_candidate_indices(
                target_scenarios,
                n_scenarios,
                seed,
            )
            return full_horizon[:, indices, :]

        return generate_candidates

    get_load_scenario_generator._gridfm_retry_candidate_policy = RETRY_CANDIDATE_POLICY
    generate.get_load_scenario_generator = get_load_scenario_generator


def _recorded_dropped_bus_ids() -> list[int]:
    if len(_TOPOLOGY_PREPROCESSING) != 1:
        raise RuntimeError(
            "generation must load exactly one normalized PGLib topology",
        )
    return next(iter(_TOPOLOGY_PREPROCESSING.values()))


# Spawned workers import this module without calling main(). The parent sets the
# marker only after verifying the exact editable Datakit checkout.
if os.environ.get(_CHUNK_SEED_SHIM) == "1":
    _install_uint32_chunk_seed_shim()
    _install_topology_preprocessing_shim()


def _environment_hash() -> str:
    rows = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    datakit_root = assert_datakit_checkout(
        args.repo_root.resolve(),
        args.expected_commit,
    )
    os.environ[_CHUNK_SEED_SHIM] = "1"
    _install_uint32_chunk_seed_shim()
    _install_topology_preprocessing_shim()
    config_path = args.config.resolve()
    config_sha256 = file_sha256(config_path)
    prior = None
    if args.resume_incomplete:
        if not args.output.is_file():
            raise RuntimeError("resume requires the original generation provenance")
        prior = json.loads(args.output.read_text())
        if (
            prior.get("status") != "GENERATED"
            or prior.get("config_sha256") != config_sha256
            or prior.get("datakit_commit") != args.expected_commit
            or prior.get("inventory_sha256") != args.inventory_sha256
        ):
            raise RuntimeError("resume provenance does not match frozen inputs")

    config = yaml.safe_load(config_path.read_text())
    _install_retry_candidate_shim(int(config["load"]["scenarios"]))
    import gridfm_datakit
    from gridfm_datakit.generate import generate_power_flow_data_distributed

    raw = (
        Path(config["settings"]["data_dir"]).resolve()
        / config["network"]["name"]
        / "raw"
    )
    retry_state = complete_pf_pool(
        config,
        raw,
        generate_power_flow_data_distributed,
        resume=args.resume_incomplete,
        candidate_policy=RETRY_CANDIDATE_POLICY,
    )
    topology_preprocessing = (
        {
            "policy": TOPOLOGY_PREPROCESSING_POLICY,
            "dropped_bus_ids": _recorded_dropped_bus_ids(),
        }
        if _TOPOLOGY_PREPROCESSING
        else prior["topology_preprocessing"]
    )
    payload = {
        "schema_version": "fm-scaling-datakit-provenance-v1",
        "status": "GENERATED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "datakit_commit": args.expected_commit,
        "datakit_root": str(datakit_root),
        "module_path": str(Path(gridfm_datakit.__file__).resolve()),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment_sha256": _environment_hash(),
        "inventory_sha256": args.inventory_sha256,
        "topology_preprocessing": topology_preprocessing,
        "retry_policy": RETRY_POLICY,
        "retry_candidate_policy": RETRY_CANDIDATE_POLICY,
        "requested_scenario_count": int(config["load"]["scenarios"]),
        "successful_scenario_count": successful_scenario_count(raw),
        "generation_attempts": retry_state["attempts"],
        "retry_state_sha256": file_sha256(raw / RETRY_STATE_FILE),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
