# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.fm_scaling.freeze_calibration import freeze_loss
from experiments.fm_scaling.run_calibration import (
    _subprocess_python,
    aggregate_calibration_metrics,
    build_calibration_config,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError


def _topology_payload():
    topologies = {}
    for index in range(26):
        topologies[f"source{index:02d}"] = {
            "topology_key": f"source:{index}",
            "baseMVA": 100.0,
            "provenance_group": f"group{index % 5}",
            "split": "source",
            "bus_count": index + 3,
            "scenario_count": 2_331,
            "integrity_status": "PASS",
        }
    for index, group in enumerate(("activ", "pserc")):
        topologies[f"dev{index}"] = {
            "topology_key": f"dev:{index}",
            "baseMVA": 100.0,
            "provenance_group": group,
            "split": "source_dev",
            "bus_count": 200 + 40 * index,
            "scenario_count": 512,
            "integrity_status": "PASS",
        }
    return {"topologies": topologies}


def test_build_calibration_config_is_flat_source_only(tmp_path):
    config = build_calibration_config(
        candidate={"id": "C002", "loss_weights": [1.0, 0.1]},
        c_cal=62_000_000_000_000,
        topology_payload=_topology_payload(),
        topology_manifest=tmp_path / "topology.yaml",
        metric_scales={"vm_pu": 0.01, "va_rad": math.pi / 180.0},
        geometry_bundle=tmp_path / "geometry.pt",
        split_root=tmp_path / "splits",
        output_root=tmp_path / "output",
    )
    assert config["model"]["communication_core"] == "flat"
    assert len(config["data"]["networks"]) == 26
    assert len(config["data"]["target_networks"]) == 2
    assert config["training"]["loss_weights"] == [1.0, 0.1]
    assert config["training"]["flop_checkpoints"] == [62_000_000_000_000]
    assert config["evaluation"]["vm_scale"] == 0.01
    assert config["evaluation"]["va_scale"] == pytest.approx(math.pi / 180)


def test_subprocess_python_preserves_virtualenv_symlink(tmp_path):
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base_python)

    assert _subprocess_python(venv_python) == venv_python
    assert _subprocess_python(venv_python) != venv_python.resolve()


def test_aggregate_calibration_metrics_is_group_balanced():
    payload = _topology_payload()
    records = [
        {
            "topology_key": "dev:0",
            "family_balanced_error": 2.0,
            "dimensionless_residual": 4.0,
        },
        {
            "topology_key": "dev:0",
            "family_balanced_error": 4.0,
            "dimensionless_residual": 6.0,
        },
        {
            "topology_key": "dev:1",
            "family_balanced_error": 8.0,
            "dimensionless_residual": 10.0,
        },
    ]
    result = aggregate_calibration_metrics(records, payload)
    assert result["error"] == 5.5
    assert result["residual"] == 7.5
    assert [row["provenance_group"] for row in result["groups"]] == ["activ", "pserc"]


def test_aggregate_calibration_metrics_rejects_target_subset():
    with pytest.raises(ContractError):
        aggregate_calibration_metrics(
            [
                {
                    "topology_key": "dev:0",
                    "family_balanced_error": 1.0,
                    "dimensionless_residual": 1.0,
                },
            ],
            _topology_payload(),
        )


def test_freeze_loss_uses_residual_for_candidates_within_one_percent():
    result = freeze_loss(
        {
            "candidates": [
                {
                    "id": "C001",
                    "status": "FINISHED",
                    "gpu_hours": 0.8,
                    "error": 1.0,
                    "residual": 3.0,
                },
                {
                    "id": "C002",
                    "status": "FINISHED",
                    "gpu_hours": 0.8,
                    "error": 1.009,
                    "residual": 1.0,
                },
                {
                    "id": "C003",
                    "status": "FINISHED",
                    "gpu_hours": 0.8,
                    "error": 1.02,
                    "residual": 0.1,
                },
            ],
        },
    )
    assert result["selected"]["id"] == "C002"
