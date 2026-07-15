# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from gridfm_graphkit.fm_scaling.accounting import (
    CumulativeFlopCheckpoint,
    counted_forward_flops,
    deterministic_capacity_match,
    output_and_gradient_parity,
    parameter_match_report,
)
from gridfm_graphkit.fm_scaling.analysis import (
    aggregate_scenarios,
    average_seeds,
    exact_sign_flip_pvalue,
    exact_upper_bound,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import expected_run_matrix
from gridfm_graphkit.fm_scaling.loss import GraphBalancedMaskedVMVA
from experiments.fm_scaling.launch import build_evaluation_commands
from experiments.fm_scaling.evaluate_campaign import _validate_metrics
from experiments.fm_scaling.make_campaign import _config


def _records():
    records = []
    for system, error in (("kron", 0.9), ("flat", 1.0)):
        for seed in (0, 1):
            for scenario in (0, 1):
                records.append(
                    {
                        "system": system,
                        "g_level": "G28",
                        "seed": seed,
                        "checkpoint": "C",
                        "topology_key": "case-a",
                        "scenario_id": scenario,
                        "family_balanced_error": error + 0.01 * scenario,
                        "dimensionless_residual": 0.1,
                        "rmse_vm_pu": error,
                        "rmse_va_rad": error / 10,
                    },
                )
    return records


def test_aggregation_orders_scenarios_before_seeds():
    topology = aggregate_scenarios(_records())
    averaged = average_seeds(topology)
    by_system = {record["system"]: record for record in averaged}
    assert by_system["kron"]["family_balanced_error"] == pytest.approx(0.905)
    assert by_system["flat"]["family_balanced_error"] == pytest.approx(1.005)


def test_seed_aggregation_rejects_missing_failures():
    topology = aggregate_scenarios(_records())
    with pytest.raises(ContractError, match="expected seeds"):
        average_seeds([record for record in topology if record["seed"] == 0])


def test_exact_sign_flip_and_inverted_bound():
    contrasts = [-0.2] * 6
    assert exact_sign_flip_pvalue(contrasts) == pytest.approx(1 / 64)
    assert exact_upper_bound(contrasts) < 0
    mixed = [-0.2, -0.1, -0.1, 0.1, 0.1, 0.2]
    assert exact_sign_flip_pvalue(mixed) >= 0.05


def test_flop_counter_and_parameter_match_are_machine_checkable():
    first = torch.nn.Linear(4, 3)
    second = copy.deepcopy(first)
    assert counted_forward_flops(first, torch.randn(2, 4)) == 48
    report = parameter_match_report({"first": first, "second": second})
    assert report["passed"] is True
    assert report["relative_gap"] == 0

    selected = deterministic_capacity_match(
        {
            "flat": [({"width": 8, "q": 2}, 100), ({"width": 9, "q": 1}, 120)],
            "kron": [({"width": 7}, 101), ({"width": 8}, 130)],
        },
    )
    assert selected["selection"]["flat"]["parameters"] == 100
    assert selected["selection"]["kron"]["parameters"] == 101


def test_output_gradient_parity_and_first_crossing_checkpoint(tmp_path):
    class DictLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)

        def forward(self, value):
            return {"output": self.linear(value)}

    reference = DictLinear()
    candidate = copy.deepcopy(reference)
    assert output_and_gradient_parity(
        reference,
        candidate,
        torch.randn(2, 4),
    )["passed"]

    class Strategy:
        root_device = torch.device("cpu")

    class Trainer:
        strategy = Strategy()
        global_step = 1
        is_global_zero = True
        should_stop = False

    checkpoint = CumulativeFlopCheckpoint([1], str(tmp_path))
    trainer = Trainer()
    checkpoint.on_train_batch_start(trainer, reference, None, 0)
    reference(torch.randn(2, 4))["output"].sum().backward()
    checkpoint.on_train_batch_end(trainer, reference, None, None, 0)
    assert checkpoint.cumulative_flops > 0
    assert trainer.should_stop is True
    assert (tmp_path / "flops_1.pt").is_file()
    assert (tmp_path / "flop_checkpoints.json").is_file()


def test_run_matrix_is_exact_and_has_only_two_quotient_runs():
    matrix = expected_run_matrix()
    assert list(matrix) == [f"E{index:03d}" for index in range(1, 21)]
    quotient = [value for value in matrix.values() if value[0] == "quotient"]
    assert quotient == [("quotient", "G28", 0), ("quotient", "G28", 1)]


def test_generated_config_trains_on_sources_and_evaluates_frozen_targets(tmp_path):
    sources = [f"source-{index}" for index in range(8)]
    targets = ["target-a", "target-b"]
    metadata = {
        network: {"provenance_group": f"group-{index}"}
        for index, network in enumerate(sources + targets)
    }
    freeze = {
        "source_sets": {"G8": sources},
        "target_networks": targets,
        "topology_payload": {"topologies": metadata},
        "model": {
            "widths": {"flat": 8},
            "l_pre": 1,
            "l_post": 1,
            "flat_blocks": 1,
        },
        "geometry_bundle": "geometry.pt",
        "callbacks": {"patience": 1, "tol": 0},
        "topology_manifest": "topology.yaml",
        "available_scenarios": {network: 4 for network in sources + targets},
        "split_directory": "splits",
        "data_workers": 0,
        "samples_total": 8,
        "optimizer": {},
        "training": {"batch_size": 1, "epochs": 2},
        "loss_weights": [0.5, 0.5],
        "flop_checkpoints": [10, 20, 40],
        "metric_scales": {"vm_pu": 0.01, "va_rad": 0.01},
    }
    config = _config(freeze, "E001", "flat", "G8", 0)
    assert config["data"]["train_networks"] == sources
    assert config["data"]["networks"] == sources
    assert config["data"]["target_networks"] == targets
    assert len(config["data"]["provenance_groups"]) == 8

    config_path = tmp_path / "E001.yaml"
    config_path.write_text(yaml.safe_dump(config))
    manifest = SimpleNamespace(
        runs=(SimpleNamespace(run_id="E001", config_path=config_path),),
        result_root=tmp_path / "results",
        mlflow_store=tmp_path / "mlflow",
        data_root=tmp_path / "data",
    )
    commands = build_evaluation_commands(
        manifest,
        "E001",
        Path("/python"),
        tmp_path,
    )
    assert [label for label, _, _ in commands] == ["C/4", "C/2", "C"]
    assert (
        commands[-1][1]
        == tmp_path / "mlruns/fm-scaling/result-summaries/E001-checkpoints/flops_40.pt"
    )


def test_evaluation_seal_requires_every_frozen_target_scenario(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("[]")
    run = SimpleNamespace(run_id="E001", core="flat", g_level="G8", seed=0)
    with pytest.raises(ContractError, match="empty or invalid"):
        _validate_metrics(path, run, "C", {("target", 0), ("target", 1)})


def test_vmva_objective_weights_graphs_not_nodes():
    # Graph 0 has one bus with squared error 4; graph 1 has three buses with
    # squared error 1. Graph-balanced mean is (4 + 1) / 2 = 2.5, not 1.75.
    pred = torch.zeros((4, 4))
    target = torch.zeros((4, 5))
    pred[:, 0] = torch.tensor([2.0, 1.0, 1.0, 1.0])
    mask = torch.ones((4, 15), dtype=torch.bool)
    batch = torch.tensor([0, 1, 1, 1])
    loss = GraphBalancedMaskedVMVA(None, None)(
        {"bus": pred},
        {"bus": target},
        None,
        None,
        {"bus": mask, "_bus_batch": batch, "_num_graphs": 2},
    )
    # VA is exactly zero, so the final equal-component average halves 2.5.
    assert loss["loss"].item() == pytest.approx(1.25)
