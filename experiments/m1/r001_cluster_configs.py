# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R001 (cluster scale): datakit generation configs for the M1 dataset.

Target: ~10k SOLVED scenarios per grid (the full-scale volume the frozen
experiment plan requires before M1). Requested counts bake in the measured
M0 fast-PF yields (EXPERIMENT_RESULTS.md, R001):

- case14/30/57/118/500: yield ~100%  -> request 10000
- Texas2k:              yield 99.2%  -> request 10240
- case2000_goc:         yield 25%    -> request 40960 (~4.1x)

case2000 decision (named): keep ``pf_fast: true`` and oversubscribe 4x
rather than softening the load scaling or falling back to Ipopt
(``pf_fast: false``). Softening only case2000's scaling would give it a
different (easier) scenario distribution than every other grid, confounding
the B1 size-vs-accuracy frontier; Ipopt solves the stressed cases but at
order-of-magnitude solve cost for 40k scenarios. Standing caveat either
way: convergence-filtering biases the surviving case2000 scenarios toward
less-stressed load regimes — a property of fast-PF data generation itself,
independent of the request volume.

Perturbations stay OFF (fixed Y per grid — the per-grid Kron-Schur operator
design; ``datasets/hierarchy.py`` hard-fails on per-scenario Y).

Paths are relative: run datakit from the gridfm-graphkit repo root on the
cluster with gridfm-datakit checked out as a sibling directory (mirroring
the local layout). Adjust ``network_dir``/``data_dir`` if the cluster
layout differs.

Disk estimate (linear extrapolation of measured M0 MB/scenario, raw +
processed): case14 ~0.13 GB, case30 ~0.07, case57 ~0.12, case118 ~1.0,
case500 ~2.3, case2000 ~18.4 (per SOLVED scenario), Texas2k ~4.1 —
~26 GB total, before the mmap ``consolidated.pt`` copies (E003; roughly
raw-sized again for the training grids).

After generation, build the hierarchy caches (R002 precompute) and the
consolidated stores; all M1+ training configs set ``data.consolidated:
true`` per E003.
"""

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, source, requested scenarios) — see module docstring for yields
GRIDS = [
    ("case14_ieee", "pglib", 10000),
    ("case30_ieee", "pglib", 10000),
    ("case57_ieee", "pglib", 10000),
    ("case118_ieee", "pglib", 10000),
    ("case500_goc", "pglib", 10000),
    ("case2000_goc", "pglib", 40960),
    ("Texas2k_case1_2016summerpeak", "file", 10240),
]


def make_config(name: str, source: str, scenarios: int) -> dict:
    return {
        "network": {
            "name": name,
            "source": source,
            # sibling checkout on the cluster; adjust if the layout differs
            "network_dir": "../gridfm-datakit/scripts/grids",
        },
        # identical load-scenario family to M0 (only the volume changes)
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
            "num_processes": 16,  # cluster CPUs; bump to the allocation
            "data_dir": "data",
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
        print(f"wrote {path} ({scenarios} scenarios requested)")
