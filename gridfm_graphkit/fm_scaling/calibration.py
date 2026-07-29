# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Treatment-blind calibration-budget helpers."""

from __future__ import annotations

import math

import numpy as np

from gridfm_graphkit.fm_scaling.contracts import ContractError


def _round_down_significant(value: float, digits: int = 2) -> int:
    if not math.isfinite(value) or value <= 0 or digits <= 0:
        raise ContractError("calibration budget must be finite and positive")
    quantum = 10 ** max(math.floor(math.log10(value)) - digits + 1, 0)
    return int(math.floor(value / quantum) * quantum)


def freeze_calibration_budget(
    measurements: list[dict],
    *,
    candidate_count: int = 3,
    aggregate_hours: float = 3.0,
    nontraining_reserve_seconds: float = 600.0,
    throughput_guard: float = 1.25,
) -> dict:
    """Freeze the largest guarded common FLOP budget under three GPU-hours."""
    if candidate_count != 3:
        raise ContractError("R005 requires exactly three loss candidates")
    if aggregate_hours <= 0 or throughput_guard < 1:
        raise ContractError("invalid R005 budget constants")
    if not measurements:
        raise ContractError("R005 requires source-development measurements")

    rows = []
    for measurement in measurements:
        if measurement.get("split") != "source_dev":
            raise ContractError("R005 may read only source-development batches")
        if measurement.get("communication_core") != "flat":
            raise ContractError("R005 throughput probe must use Flat only")
        flops = int(measurement.get("training_step_flops", 0))
        seconds = [float(value) for value in measurement.get("step_seconds", [])]
        if flops <= 0 or len(seconds) < 3:
            raise ContractError("R005 requires positive FLOPs and at least three timings")
        if any(not math.isfinite(value) or value <= 0 for value in seconds):
            raise ContractError("R005 timings must be finite and positive")
        p95_seconds = float(np.quantile(seconds, 0.95, method="higher"))
        rows.append(
            {
                "network": str(measurement["network"]),
                "training_step_flops": flops,
                "measured_steps": len(seconds),
                "p95_step_seconds": p95_seconds,
                "p95_seconds_per_flop": p95_seconds / flops,
                "compile_and_warmup_seconds": float(
                    measurement.get("compile_and_warmup_seconds", 0.0),
                ),
            },
        )

    slowest_seconds_per_flop = max(row["p95_seconds_per_flop"] for row in rows)
    aggregate_seconds = float(aggregate_hours) * 3600.0
    per_candidate_seconds = aggregate_seconds / candidate_count
    train_seconds = per_candidate_seconds - float(nontraining_reserve_seconds)
    if train_seconds <= 0:
        raise ContractError("R005 non-training reserve exhausts the candidate budget")
    raw_c_cal = train_seconds / (throughput_guard * slowest_seconds_per_flop)
    c_cal_flops = _round_down_significant(raw_c_cal)
    per_candidate_upper_seconds = (
        nontraining_reserve_seconds
        + throughput_guard * slowest_seconds_per_flop * c_cal_flops
    )
    aggregate_upper_seconds = candidate_count * per_candidate_upper_seconds
    if aggregate_upper_seconds > aggregate_seconds:
        raise ContractError("R005 rounded budget exceeds three GPU-hours")

    return {
        "selection_rule": (
            "two-significant-digit floor of the largest common counted-training-"
            "FLOP budget under three guarded one-hour candidate bounds"
        ),
        "candidate_count": candidate_count,
        "aggregate_budget_seconds": aggregate_seconds,
        "per_candidate_budget_seconds": per_candidate_seconds,
        "nontraining_reserve_seconds_per_candidate": nontraining_reserve_seconds,
        "throughput_guard": throughput_guard,
        "slowest_p95_seconds_per_flop": slowest_seconds_per_flop,
        "raw_c_cal_flops": raw_c_cal,
        "c_cal_flops": c_cal_flops,
        "per_candidate_upper_seconds": per_candidate_upper_seconds,
        "aggregate_upper_seconds": aggregate_upper_seconds,
        "measurements": rows,
    }
