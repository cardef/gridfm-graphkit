# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R001 (M0): emit datakit generation configs for the 8-grid M0 dataset.

Scenario counts are M0-scale (local Mac, ~1.5 GB free disk at run time --
full-scale regeneration happens on the cluster before M1). Perturbations are
OFF so the admittance matrix Y is fixed per grid, matching the frozen
per-grid-operator design in refine-logs/FINAL_PROPOSAL.md.
"""

import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "data"))
DATAKIT_GRIDS = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "gridfm-datakit", "scripts", "grids"),
)

# (name, source, scenarios) -- local M0 scale (disk freed 2026-07-07; ~1 GB
# total). Full-scale (10k/grid) still happens on the cluster before M1.
GRIDS = [
    ("case14_ieee", "pglib", 2048),
    ("case30_ieee", "pglib", 2048),
    ("case57_ieee", "pglib", 2048),
    ("case118_ieee", "pglib", 1024),
    ("case500_goc", "pglib", 512),
    ("case2000_goc", "pglib", 256),
    ("Texas2k_case1_2016summerpeak", "file", 256),
]


def make_config(name: str, source: str, scenarios: int) -> dict:
    return {
        "network": {
            "name": name,
            "source": source,
            "network_dir": DATAKIT_GRIDS,
        },
        "load": {
            "generator": "agg_load_profile",
            "agg_profile": "default",
            "scenarios": scenarios,
            "sigma": 0.2,
            "change_reactive_power": True,
            "global_range": 0.4,
            "max_scaling_factor": 4.0,
            "step_size": 0.1,
            "start_scaling_factor": 1.0,
        },
        # Perturbations OFF: fixed Y per grid (per-grid Kron/Schur operators).
        "topology_perturbation": {"type": "none"},
        "generation_perturbation": {"type": "none"},
        "admittance_perturbation": {"type": "none"},
        "settings": {
            "num_processes": 6,
            "data_dir": DATA_DIR,
            "large_chunk_size": 512,
            "overwrite": True,
            "mode": "pf",
            "include_dc_res": True,
            "enable_solver_logs": False,
            "pf_fast": True,
            "dcpf_fast": True,
            "max_iter": 200,
            "seed": 0,
        },
    }


if __name__ == "__main__":
    cfg_dir = os.path.join(HERE, "datakit_configs")
    os.makedirs(cfg_dir, exist_ok=True)
    for name, source, scenarios in GRIDS:
        path = os.path.join(cfg_dir, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(make_config(name, source, scenarios), f, sort_keys=False)
        print(f"wrote {path} ({scenarios} scenarios)")
