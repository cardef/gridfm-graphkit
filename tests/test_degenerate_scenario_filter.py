"""R007 filter: scenarios with a zero-load bus collapsed to V~0 (a
degenerate fast-PF artifact, see experiments/m0/r007_outlier_triage.py)
must be excluded from HeteroGridDatasetDisk/HeteroGridDatasetMMap."""

import os
import os.path as osp

import pandas as pd
import pytest

from gridfm_graphkit.datasets.powergrid_hetero_dataset import (
    DEGENERATE_VM_THRESHOLD,
    HeteroGridDatasetDisk,
    HeteroGridDatasetMMap,
    _degenerate_scenarios,
)

BUS_COLUMNS = [
    "scenario",
    "bus",
    "Pd",
    "Qd",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "min_vm_pu",
    "max_vm_pu",
    "GS",
    "BS",
    "vn_kv",
]
GEN_COLUMNS = [
    "scenario",
    "bus",
    "in_service",
    "p_mw",
    "min_p_mw",
    "max_p_mw",
    "cp0_eur",
    "cp1_eur_per_mw",
    "cp2_eur_per_mw2",
    "min_q_mvar",
    "max_q_mvar",
]
BRANCH_COLUMNS = [
    "scenario",
    "from_bus",
    "to_bus",
    "tap",
    "ang_min",
    "ang_max",
    "rate_a",
    "br_status",
    "pf",
    "qf",
    "Yff_r",
    "Yff_i",
    "Yft_r",
    "Yft_i",
    "pt",
    "qt",
    "Ytt_r",
    "Ytt_i",
    "Ytf_r",
    "Ytf_i",
]

# scenario 2's leaf bus is zero-load with |Vm| far below DEGENERATE_VM_THRESHOLD.
ALL_SCENARIOS = {
    0: {"Pd": 10.0, "Qd": 2.0, "Vm": 0.98},
    1: {"Pd": 8.0, "Qd": 1.0, "Vm": 1.01},
    2: {"Pd": 0.0, "Qd": 0.0, "Vm": 1e-8},
}


class _NoOpNormalizer:
    def transform(self, data):
        return data


def _write_raw(root, scenarios):
    bus_rows, gen_rows, branch_rows = [], [], []
    for s, leaf in scenarios.items():
        bus_rows.append(
            [s, 0, 0.0, 0.0, 0.0, 1.0, 0.0, 0, 0, 1, 0.9, 1.1, 0.0, 0.0, 138.0],
        )
        bus_rows.append(
            [
                s,
                1,
                leaf["Pd"],
                leaf["Qd"],
                0.0,
                leaf["Vm"],
                0.0,
                1,
                0,
                0,
                0.9,
                1.1,
                0.0,
                0.0,
                138.0,
            ],
        )
        gen_rows.append([s, 0, 1, 10.0, 0.0, 20.0, 0.0, 0.0, 0.0, -10.0, 10.0])
        branch_rows.append(
            [s, 0, 1, 1.0, -360.0, 360.0, 100.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    raw_dir = osp.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    pd.DataFrame(bus_rows, columns=BUS_COLUMNS).to_parquet(
        osp.join(raw_dir, "bus_data.parquet"),
    )
    pd.DataFrame(gen_rows, columns=GEN_COLUMNS).to_parquet(
        osp.join(raw_dir, "gen_data.parquet"),
    )
    pd.DataFrame(branch_rows, columns=BRANCH_COLUMNS).to_parquet(
        osp.join(raw_dir, "branch_data.parquet"),
    )


@pytest.fixture
def synthetic_root(tmp_path):
    root = str(tmp_path / "toy_grid")
    _write_raw(root, ALL_SCENARIOS)
    return root


def test_degenerate_scenarios_predicate(synthetic_root):
    assert _degenerate_scenarios(osp.join(synthetic_root, "raw")) == {2}


def test_disk_dataset_excludes_degenerate_scenario(synthetic_root):
    dataset = HeteroGridDatasetDisk(
        root=synthetic_root,
        data_normalizer=_NoOpNormalizer(),
    )
    assert len(dataset) == 2
    seen = {int(dataset[i]["scenario_id"].item()) for i in range(len(dataset))}
    assert seen == {0, 1}


def test_mmap_dataset_excludes_degenerate_scenario(synthetic_root):
    dataset = HeteroGridDatasetMMap(
        root=synthetic_root,
        data_normalizer=_NoOpNormalizer(),
    )
    assert len(dataset) == 2
    seen = {int(dataset[i]["scenario_id"].item()) for i in range(len(dataset))}
    assert seen == {0, 1}


def test_no_filtering_when_all_scenarios_healthy(tmp_path):
    root = str(tmp_path / "healthy_grid")
    healthy = {k: v for k, v in ALL_SCENARIOS.items() if k != 2}
    _write_raw(root, healthy)
    assert _degenerate_scenarios(osp.join(root, "raw")) == set()
    dataset = HeteroGridDatasetDisk(root=root, data_normalizer=_NoOpNormalizer())
    assert len(dataset) == 2
    assert DEGENERATE_VM_THRESHOLD == 0.1
