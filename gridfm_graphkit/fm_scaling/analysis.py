# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Locked scenario-to-topology-to-group aggregation and exact inference."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np

from gridfm_graphkit.fm_scaling.contracts import ContractError


def aggregate_scenarios(records: list[dict]) -> list[dict]:
    """Average scenarios first, preserving arm, seed, checkpoint, and topology."""
    required = {
        "system",
        "g_level",
        "seed",
        "checkpoint",
        "topology_key",
        "family_balanced_error",
        "dimensionless_residual",
        "rmse_vm_pu",
        "rmse_va_rad",
    }
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        missing = required - set(record)
        if missing:
            raise ContractError(f"evaluation record misses {sorted(missing)}")
        key = tuple(
            record[field]
            for field in sorted(
                required
                - {
                    "family_balanced_error",
                    "dimensionless_residual",
                    "rmse_vm_pu",
                    "rmse_va_rad",
                },
            )
        )
        grouped[key].append(record)
    output = []
    metric_fields = (
        "family_balanced_error",
        "dimensionless_residual",
        "rmse_vm_pu",
        "rmse_va_rad",
    )
    key_fields = sorted(required - set(metric_fields))
    for key, items in sorted(grouped.items()):
        errors = [float(item["family_balanced_error"]) for item in items]
        residuals = [float(item["dimensionless_residual"]) for item in items]
        if min(errors + residuals) < 0 or not all(
            math.isfinite(value) for value in errors + residuals
        ):
            raise ContractError("metrics must be finite and nonnegative")
        record = dict(zip(key_fields, key))
        record.update(
            {
                "scenario_count": len(items),
                "family_balanced_error": float(np.mean(errors)),
                "dimensionless_residual": float(np.mean(residuals)),
                "rmse_vm_pu": float(np.mean([item["rmse_vm_pu"] for item in items])),
                "rmse_va_rad": float(np.mean([item["rmse_va_rad"] for item in items])),
            },
        )
        output.append(record)
    return output


def average_seeds(topology_records: list[dict]) -> list[dict]:
    """Average the two preregistered seeds; seeds are not replicates."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    fields = ("system", "g_level", "checkpoint", "topology_key")
    for record in topology_records:
        grouped[tuple(record[field] for field in fields)].append(record)
    output = []
    for key, items in sorted(grouped.items()):
        seeds = sorted(int(item["seed"]) for item in items)
        if seeds != [0, 1]:
            raise ContractError(f"expected seeds [0, 1], found {seeds} for {key}")
        output.append(
            {
                **dict(zip(fields, key)),
                "family_balanced_error": float(
                    np.mean([item["family_balanced_error"] for item in items]),
                ),
                "dimensionless_residual": float(
                    np.mean([item["dimensionless_residual"] for item in items]),
                ),
                "rmse_vm_pu": float(np.mean([item["rmse_vm_pu"] for item in items])),
                "rmse_va_rad": float(np.mean([item["rmse_va_rad"] for item in items])),
            },
        )
    return output


def provenance_group_contrasts(
    averaged: list[dict],
    topology_metadata: dict[str, dict],
    *,
    treatment: str,
    baseline: str,
    g_level: str,
    checkpoint: str,
) -> dict[str, dict[str, float]]:
    """Pair topologies, take log differences, then equal-weight within groups."""
    selected = {
        (record["system"], record["topology_key"]): record
        for record in averaged
        if record["g_level"] == g_level and record["checkpoint"] == checkpoint
    }
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    target_keys = sorted(
        key for key, meta in topology_metadata.items() if meta["split"] == "target"
    )
    for topology_key in target_keys:
        try:
            treatment_record = selected[(treatment, topology_key)]
            baseline_record = selected[(baseline, topology_key)]
        except KeyError as error:
            raise ContractError(
                f"missing paired topology result {error.args[0]}",
            ) from error
        group = topology_metadata[topology_key]["provenance_group"]
        groups[group].append(
            (
                math.log(max(treatment_record["family_balanced_error"], 1e-12))
                - math.log(max(baseline_record["family_balanced_error"], 1e-12)),
                math.log(max(treatment_record["dimensionless_residual"], 1e-12))
                - math.log(max(baseline_record["dimensionless_residual"], 1e-12)),
            ),
        )
    if not groups:
        raise ContractError("no paired provenance groups remain")
    return {
        group: {
            "error": float(np.mean([value[0] for value in values])),
            "residual": float(np.mean([value[1] for value in values])),
            "topology_count": len(values),
        }
        for group, values in sorted(groups.items())
    }


def exact_sign_flip_pvalue(contrasts: list[float], null_location: float = 0.0) -> float:
    """Exact one-sided p-value for the alternative mean contrast < location."""
    values = np.asarray(contrasts, dtype=float) - null_location
    if values.size < 1 or not np.isfinite(values).all():
        raise ContractError("sign-flip contrasts must be finite and nonempty")
    observed = float(values.mean())
    count = 0
    total = 2**values.size
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        statistic = float(np.mean(values * np.asarray(signs)))
        count += statistic <= observed + 1e-15
    return count / total


def exact_upper_bound(contrasts: list[float], alpha: float = 0.05) -> float:
    """Invert the exact one-sided sign-flip test for an upper location bound."""
    values = np.asarray(contrasts, dtype=float)
    if not 0 < alpha < 0.5:
        raise ContractError("alpha must lie between zero and one half")
    scale = max(float(np.max(np.abs(values))), 1.0)
    low = float(np.min(values) - 10 * scale)
    high = float(np.max(values) + 10 * scale)
    # p(mu) decreases as the tested null location moves above the data.
    for _ in range(100):
        middle = (low + high) / 2
        if exact_sign_flip_pvalue(values.tolist(), middle) < alpha:
            high = middle
        else:
            low = middle
    return high


def design_power(
    *,
    group_count: int,
    sigma_design: float,
    delta_min: float = -math.log(0.95),
    draws: int = 1_000_000,
    seed: int = 20260714,
) -> float:
    """Fixed-PCG64 Monte Carlo power for the preregistered design model."""
    if group_count < 1 or group_count > 10 or sigma_design <= 0 or draws < 1:
        raise ContractError("invalid design-power arguments")
    rng = np.random.Generator(np.random.PCG64(seed))
    rejected = 0
    permutations = 2**group_count
    integers = np.arange(permutations, dtype=np.uint16)[:, None]
    bits = (integers >> np.arange(group_count, dtype=np.uint16)) & 1
    signs = bits.astype(float) * 2 - 1
    chunk = max(1, min(10_000, 2_000_000 // permutations))
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        samples = rng.normal(-delta_min, sigma_design, size=(size, group_count))
        observed = samples.mean(axis=1, keepdims=True)
        randomized = samples @ signs.T / group_count
        pvalues = (randomized <= observed + 1e-15).mean(axis=1)
        rejected += int((pvalues < 0.05).sum())
    return rejected / draws
