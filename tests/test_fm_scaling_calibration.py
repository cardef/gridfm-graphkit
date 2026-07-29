# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.fm_scaling.review_gpu_compatibility import (
    EXPECTED_CHECKS,
    review_gpu_compatibility,
)
from gridfm_graphkit.fm_scaling.calibration import freeze_calibration_budget
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import validate_gate_evidence


def _measurement(network, flops, seconds, split="source_dev", core="flat"):
    return {
        "network": network,
        "split": split,
        "communication_core": core,
        "training_step_flops": flops,
        "step_seconds": seconds,
        "compile_and_warmup_seconds": 12.0,
    }


def test_freeze_calibration_budget_is_guarded_and_rounded():
    result = freeze_calibration_budget(
        [
            _measurement("case200", 1_000, [0.8, 1.0, 0.9, 0.7]),
            _measurement("case240", 2_000, [1.0, 1.2, 1.1, 0.9]),
        ],
    )
    assert result["c_cal_flops"] == 2_400_000
    assert result["per_candidate_upper_seconds"] <= 3600
    assert result["aggregate_upper_seconds"] <= 3 * 3600
    assert result["slowest_p95_seconds_per_flop"] == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("split", "core"),
    [("target", "flat"), ("source_dev", "kron")],
)
def test_freeze_calibration_budget_rejects_treatment_information(split, core):
    with pytest.raises(ContractError):
        freeze_calibration_budget(
            [_measurement("forbidden", 1_000, [1.0, 1.0, 1.0], split, core)],
        )


def test_review_gpu_compatibility_preserves_authoritative_hash(tmp_path):
    source = tmp_path / "I010.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "fm-scaling-evidence-v1",
                "gate_id": "I010",
                "status": "PASS",
                "checks": [
                    {"name": name, "passed": True} for name in sorted(EXPECTED_CHECKS)
                ],
                "results": {
                    "fork_commit": "a" * 40,
                    "largest_network": "case13659_pegase",
                    "largest_bus_count": 13659,
                    "compile_policy": {"selected_mode": "default"},
                    "profiler": {"relative_gap": 0.001},
                    "cuda": {"device": "NVIDIA A40"},
                },
            },
        ),
    )
    output = tmp_path / "I010-reviewed.json"
    result = review_gpu_compatibility(source, output)
    assert result["results"]["device"] == "NVIDIA A40"
    assert validate_gate_evidence(output, "I010")["status"] == "PASS"


def test_review_gpu_compatibility_rejects_non_authoritative_device(tmp_path):
    source = tmp_path / "I010.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "fm-scaling-evidence-v1",
                "gate_id": "I010",
                "status": "PASS",
                "checks": [
                    {"name": name, "passed": True} for name in sorted(EXPECTED_CHECKS)
                ],
                "results": {
                    "fork_commit": "a" * 40,
                    "cuda": {"device": "NVIDIA RTX A4000"},
                },
            },
        ),
    )
    with pytest.raises(ContractError):
        review_gpu_compatibility(source, tmp_path / "reviewed.json")
