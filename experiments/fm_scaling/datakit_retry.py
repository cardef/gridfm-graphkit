# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministically complete Datakit PF pools after solver attrition."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


RETRY_POLICY = "deterministic_retry_to_fixed_success_count_v1"
RETRY_CANDIDATE_POLICY = "full_horizon_pcg64_subsample_v1"
RETRY_SEED_STRIDE = 1000
RETRY_STATE_FILE = "fm_scaling_retry_state.json"
MAX_RETRY_ROUNDS = 128


def retry_seed(base_seed: int, retry_round: int) -> int:
    """Derive a deterministic uint32 seed for a global retry round."""
    if retry_round < 1:
        raise ValueError("retry_round must be positive")
    return int(base_seed) + RETRY_SEED_STRIDE * retry_round


def retry_candidate_indices(target: int, requested: int, seed: int) -> np.ndarray:
    """Select retry candidates from the frozen full scenario horizon."""
    if target <= 0:
        raise ValueError("target must be positive")
    if requested <= 0 or requested > target:
        raise ValueError("requested must be in [1, target]")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return np.sort(generator.choice(target, size=requested, replace=False))


def successful_scenario_count(raw: Path) -> int:
    path = raw / "n_scenarios.txt"
    if not path.is_file():
        raise RuntimeError(f"missing Datakit scenario counter {path}")
    count = int(path.read_text().strip())
    if count < 0:
        raise RuntimeError(f"negative Datakit scenario counter {path}")
    return count


def build_retry_config(base_config: dict, retry_round: int, deficit: int) -> dict:
    if deficit <= 0:
        raise ValueError("retry deficit must be positive")
    config = copy.deepcopy(base_config)
    base_seed = int(config["settings"]["seed"])
    config["settings"]["seed"] = retry_seed(base_seed, retry_round)
    config["settings"]["overwrite"] = False
    config["load"]["scenarios"] = int(deficit)
    return config


def _write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def complete_pf_pool(
    base_config: dict,
    raw: Path,
    generate: Callable[[dict], Any],
    *,
    resume: bool = False,
    max_retry_rounds: int = MAX_RETRY_ROUNDS,
    candidate_policy: str | None = None,
) -> dict:
    """Run or resume a topology pool until the frozen success count is met."""
    if base_config.get("settings", {}).get("mode") != "pf":
        raise ValueError("retry-to-count policy is defined only for PF mode")
    target = int(base_config["load"]["scenarios"])
    base_seed = int(base_config["settings"]["seed"])
    state_path = raw / RETRY_STATE_FILE

    if resume:
        observed = successful_scenario_count(raw)
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if (
                state.get("retry_policy") != RETRY_POLICY
                or int(state.get("target_scenarios", -1)) != target
                or int(state.get("base_seed", -1)) != base_seed
            ):
                raise RuntimeError("retry state does not match the frozen config")
            attempts = state.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise RuntimeError("retry state has no generation attempts")
            last = attempts[-1]
            if last.get("successful_after") is None:
                last["successful_after"] = observed
                _write_state(state_path, state)
            elif int(last["successful_after"]) != observed:
                raise RuntimeError("retry state disagrees with Datakit scenario count")
        else:
            state = {
                "schema_version": "fm-scaling-retry-state-v1",
                "retry_policy": RETRY_POLICY,
                "target_scenarios": target,
                "base_seed": base_seed,
                "attempts": [
                    {
                        "kind": "initial",
                        "retry_round": 0,
                        "seed": base_seed,
                        "requested": target,
                        "successful_before": 0,
                        "successful_after": observed,
                    },
                ],
            }
            _write_state(state_path, state)
    else:
        generate(copy.deepcopy(base_config))
        observed = successful_scenario_count(raw)
        state = {
            "schema_version": "fm-scaling-retry-state-v1",
            "retry_policy": RETRY_POLICY,
            "target_scenarios": target,
            "base_seed": base_seed,
            "attempts": [
                {
                    "kind": "initial",
                    "retry_round": 0,
                    "seed": base_seed,
                    "requested": target,
                    "successful_before": 0,
                    "successful_after": observed,
                },
            ],
        }
        _write_state(state_path, state)

    if observed > target:
        raise RuntimeError(
            f"Datakit produced {observed} scenarios, above frozen target {target}",
        )

    attempts = state["attempts"]
    next_round = int(attempts[-1]["retry_round"]) + 1
    while observed < target:
        if next_round > max_retry_rounds:
            raise RuntimeError(
                f"PF pool remains incomplete after {max_retry_rounds} retries: "
                f"{observed}/{target}",
            )
        deficit = target - observed
        retry_config = build_retry_config(base_config, next_round, deficit)
        attempt = {
            "kind": "retry",
            "retry_round": next_round,
            "seed": int(retry_config["settings"]["seed"]),
            "requested": deficit,
            "successful_before": observed,
            "successful_after": None,
        }
        if candidate_policy is not None:
            attempt["candidate_policy"] = candidate_policy
        attempts.append(attempt)
        _write_state(state_path, state)
        generate(retry_config)
        updated = successful_scenario_count(raw)
        if updated < observed or updated > target:
            raise RuntimeError(
                f"invalid PF retry progress {observed} -> {updated} for target {target}",
            )
        attempt["successful_after"] = updated
        observed = updated
        _write_state(state_path, state)
        next_round += 1

    return state
