# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.fm_scaling.datakit_topology import prune_declared_inert_buses
from experiments.fm_scaling.finalize_data import RAW_FILES, finalize
from experiments.fm_scaling.freeze_targets import freeze_targets
from experiments.fm_scaling.make_splits import materialize
from experiments.fm_scaling.prepare_data import (
    INVENTORY_SCHEMA,
    render_datakit_config,
    validate_inventory,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError, SCHEMA_VERSION
from gridfm_graphkit.fm_scaling.manifest import file_sha256
from gridfm_graphkit.fm_scaling.splits import validate_materialized_splits


def test_datakit_generation_forces_spawn_before_julia_initialization():
    entrypoint = (
        Path(__file__).parents[1]
        / "experiments"
        / "fm_scaling"
        / "datakit_generate.py"
    )
    source = entrypoint.read_text()
    spawn = 'multiprocessing.set_start_method("spawn", force=True)'
    assert 0 <= source.index(spawn) < source.index("import juliacall")


def test_datakit_chunk_seed_shim_preserves_frozen_seed_separation():
    modulus = 2**32
    base_seeds = range(20260714, 20260714 + 55)
    derived = {
        (seed * 20_000 + scenario + 1) % modulus
        for seed in base_seeds
        for scenario in range(2331)
    }
    assert len(derived) == 55 * 2331


def test_declared_inert_type4_bus_is_dropped_without_reading_outcomes():
    mpc = {
        "bus": np.array(
            [
                [1, 3, 0, 0, 0, 0],
                [2, 1, 1, 0.2, 0, 0],
                [24082, 4, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
        "gen": np.array([[1, 0, 0, 0, 0, 0, 0, 1]], dtype=float),
        "branch": np.array(
            [
                [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [2, 24082, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
        "gencost": np.array([[2, 0, 0, 3, 1, 1, 1]], dtype=float),
    }
    normalized, dropped = prune_declared_inert_buses(mpc)
    assert dropped == [24082]
    assert normalized["bus"][:, 0].tolist() == [1, 2]
    assert normalized["branch"].shape[0] == 1
    assert normalized["gen"].shape[0] == normalized["gencost"].shape[0] == 1


def test_declared_noninert_type4_bus_fails_closed():
    mpc = {
        "bus": np.array([[1, 3, 0, 0, 0, 0], [2, 4, 1, 0, 0, 0]], dtype=float),
        "gen": np.empty((0, 8)),
        "branch": np.empty((0, 11)),
        "gencost": np.empty((0, 7)),
    }
    with pytest.raises(ContractError, match="nonzero load or shunt"):
        prune_declared_inert_buses(mpc)


def _case():
    return {
        "network": "case2_test",
        "topology_key": "case2-test-v1",
        "source": "pglib",
        "provenance_group": "synthetic",
        "split": "source",
        "scenarios": 2,
        "seed": 7,
    }


def test_inventory_and_rendered_config_disable_all_structural_perturbations(tmp_path):
    case = _case()
    commit = "a" * 40
    observed_commit, cases = validate_inventory(
        {
            "schema_version": INVENTORY_SCHEMA,
            "datakit_commit": commit,
            "design": {
                "source_scenarios_per_topology": 2,
                "source_dev_scenarios_per_topology": 2,
                "target_scenarios_per_topology": 2,
                "seed_rule": "20260714_plus_frozen_case_index",
                "r002_sha256": "b" * 64,
            },
            "cases": [
                {**case, "network": f"source-{index}", "topology_key": f"source-{index}", "seed": 20260714 + index}
                for index in range(26)
            ]
            + [
                {**case, "network": f"dev-{index}", "topology_key": f"dev-{index}", "split": "source_dev", "seed": 20260740 + index}
                for index in range(2)
            ]
            + [
                {**case, "network": f"target-{index}", "topology_key": f"target-{index}", "split": "target", "seed": 20260742 + index}
                for index in range(27)
            ],
        },
    )
    assert observed_commit == commit
    assert len(cases) == 55
    assert cases[0]["split"] == "source"
    assert cases[-1]["split"] == "target"
    config = render_datakit_config(case, tmp_path / "data", workers=3)
    assert config["topology_perturbation"] == {"type": "none"}
    assert config["generation_perturbation"] == {"type": "none"}
    assert config["admittance_perturbation"] == {"type": "none"}
    assert config["settings"]["mode"] == "pf"
    assert config["settings"]["seed"] == 7
    assert config["settings"]["data_dir"] == str((tmp_path / "data").resolve())


def _write_raw(raw: Path, config: dict, *, varying_y: bool = False) -> None:
    raw.mkdir(parents=True)
    scenarios = [0, 1]
    bus_rows = [
        {"scenario": scenario, "id": bus, "Pd": 1.0, "Qd": 0.2, "Vm": 1.0}
        for scenario in scenarios
        for bus in range(2)
    ]
    gen_rows = [{"scenario": scenario, "id": 0} for scenario in scenarios]
    branch_rows = [
        {
            "scenario": scenario,
            "from_bus": 0,
            "to_bus": 1,
            "br_status": 1,
        }
        for scenario in scenarios
    ]
    y_rows = []
    for scenario in scenarios:
        for row, col, real, imag in (
            (0, 0, 2.0, -4.0),
            (0, 1, -2.0, 4.0),
            (1, 0, -2.0, 4.0),
            (1, 1, 2.0, -4.0),
        ):
            y_rows.append(
                {
                    "scenario": scenario,
                    "index1": row,
                    "index2": col,
                    "G": real + (0.5 if varying_y and scenario == 1 else 0.0),
                    "B": imag,
                },
            )
    pd.DataFrame(bus_rows).to_parquet(raw / "bus_data.parquet")
    pd.DataFrame(branch_rows).to_parquet(raw / "branch_data.parquet")
    pd.DataFrame(gen_rows).to_parquet(raw / "gen_data.parquet")
    pd.DataFrame(y_rows).to_parquet(raw / "y_bus_data.parquet")
    (raw / "args.log").write_text(
        "\nNew generation started at 2026-07-15 00:00:00\n"
        + yaml.safe_dump(config, sort_keys=False),
    )


def _draft(tmp_path: Path, varying_y: bool = False):
    case = _case()
    config = render_datakit_config(case, tmp_path / "data", workers=1)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "case2_test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    draft = {
        "schema_version": SCHEMA_VERSION,
        "topologies": {
            "case2_test": {
                "topology_key": "case2-test-v1",
                "baseMVA": 100.0,
                "provenance_group": "synthetic",
                "split": "source",
                "bus_count": 2,
                "scenario_count": 2,
                "datakit_commit": "a" * 40,
                "config_sha256": file_sha256(config_path),
                "integrity_status": "PENDING",
            },
        },
    }
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(yaml.safe_dump(draft, sort_keys=False))
    _write_raw(
        tmp_path / "data" / "case2_test" / "raw",
        config,
        varying_y=varying_y,
    )
    return draft_path, config_dir


def test_finalize_data_proves_static_complete_scenarios_and_hashes(tmp_path):
    draft, config_dir = _draft(tmp_path)
    output = tmp_path / "topology-manifest.yaml"
    manifest = finalize(draft, config_dir, tmp_path / "data", output)
    record = manifest["topologies"]["case2_test"]
    assert record["integrity_status"] == "PASS"
    assert len(record["raw_sha256"]) == 64
    assert len(record["data_hash"]) == 64
    assert yaml.safe_load(output.read_text()) == manifest


def test_finalize_data_accepts_partitioned_parquet_datasets(tmp_path):
    draft, config_dir = _draft(tmp_path)
    raw = tmp_path / "data" / "case2_test" / "raw"
    for name in RAW_FILES:
        path = raw / name
        frame = pd.read_parquet(path)
        path.unlink()
        frame["scenario_partition"] = frame["scenario"] % 2
        frame.to_parquet(
            path,
            partition_cols=["scenario_partition"],
            index=False,
        )
    output = tmp_path / "topology-manifest.yaml"
    manifest = finalize(draft, config_dir, tmp_path / "data", output)
    assert manifest["topologies"]["case2_test"]["integrity_status"] == "PASS"


def test_finalize_data_rejects_scenario_varying_admittance(tmp_path):
    draft, config_dir = _draft(tmp_path, varying_y=True)
    with pytest.raises(ContractError, match="vary by scenario"):
        finalize(draft, config_dir, tmp_path / "data", tmp_path / "output.yaml")


def test_split_materialization_is_per_network_hashed_and_target_test_only(tmp_path):
    topology = {
        "schema_version": SCHEMA_VERSION,
        "topologies": {
            "source": {
                "topology_key": "source",
                "baseMVA": 100,
                "provenance_group": "source-group",
                "split": "source",
                "bus_count": 2,
                "scenario_count": 3,
            },
            "target": {
                "topology_key": "target",
                "baseMVA": 100,
                "provenance_group": "target-group",
                "split": "target",
                "bus_count": 2,
                "scenario_count": 2,
            },
        },
    }
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(yaml.safe_dump(topology))
    spec = {
        "schema_version": "fm-scaling-splits-v1",
        "splits": {
            "source": {"train": [0], "val": [1], "test": [2]},
            "target": {"train": [], "val": [], "test": [0, 1]},
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    output_root = tmp_path / "splits"
    manifest_path = tmp_path / "split-manifest.yaml"
    materialize(spec_path, topology_path, output_root, manifest_path)
    observed = validate_materialized_splits(
        manifest_path,
        output_root,
        topology,
    )
    assert observed["splits"]["target"]["train"] == []
    assert (output_root / "source" / "train.pt").is_file()


def test_target_freeze_enforces_whole_groups_and_terciles(tmp_path):
    topologies = {}
    for index in range(26):
        topologies[f"source-{index}"] = {
            "topology_key": f"source-{index}",
            "baseMVA": 100,
            "provenance_group": f"source-group-{index // 4}",
            "split": "source",
            "bus_count": index + 2,
        }
    for index in range(12):
        network = f"target-{index}"
        topologies[network] = {
            "topology_key": network,
            "baseMVA": 100,
            "provenance_group": f"target-group-{index // 2}",
            "split": "target",
            "bus_count": index + 50,
        }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "selection_freeze": {"inventory_sha256": "a" * 64},
                "topologies": topologies,
            },
        ),
    )
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "fm-scaling-target-freeze-v1",
                "inventory_sha256": "a" * 64,
            },
        ),
    )
    frozen = freeze_targets(
        manifest_path,
        selection_path,
        tmp_path / "frozen.yaml",
    )
    assert frozen["topologies"]["target-11"]["size_tercile"] == "largest"
    assert frozen["target_freeze"]["extrapolative_count"] == 12

    bad_selection = tmp_path / "bad-selection.yaml"
    bad_selection.write_text(
        yaml.safe_dump(
            {
                "schema_version": "fm-scaling-target-freeze-v1",
                "inventory_sha256": "b" * 64,
            },
        ),
    )
    with pytest.raises(ContractError, match="pre-generation inventory"):
        freeze_targets(manifest_path, bad_selection, tmp_path / "bad.yaml")
