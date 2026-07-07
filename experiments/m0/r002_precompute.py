# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R002 (M0): Kron-Schur operator precompute + hard asserts, all grids.

Two paths:
- grids with datakit parquet data under data/<name>/raw: full precompute
  (operators + per-scenario V_aff/coarse injections) via
  gridfm_graphkit.datasets.hierarchy.build_grid_hierarchy, plus a
  harmonic-extension quality diagnostic against the true AC solution;
- grids without local scenario data (disk-blocked at M0): operators-only
  sweep from the pglib/.m case file -- the P-mass hard assert and Y_red
  density gates are topology-level and need no scenarios.

Writes experiments/m0/results/r002_precompute.json (+ .log).
"""

import json
import os
import os.path as osp
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(REPO, "idea-stage"))

from pilot_cpu import load_case, ybus  # noqa: E402  (pilot's MATPOWER Ybus)
from gridfm_graphkit.datasets.hierarchy import (  # noqa: E402
    build_grid_hierarchy,
    build_operators,
    select_boundary,
    hierarchy_cache_name,
)

DATAKIT_GRIDS = osp.join(REPO, "..", "gridfm-datakit", "gridfm_datakit", "grids")
DATAKIT_FILE_GRIDS = osp.join(REPO, "..", "gridfm-datakit", "scripts", "grids")

GRIDS = {
    "case14_ieee": osp.join(DATAKIT_GRIDS, "pglib_opf_case14_ieee_corrected.m"),
    "case30_ieee": osp.join(DATAKIT_GRIDS, "pglib_opf_case30_ieee_corrected.m"),
    "case57_ieee": osp.join(DATAKIT_GRIDS, "pglib_opf_case57_ieee_corrected.m"),
    "case118_ieee": osp.join(DATAKIT_GRIDS, "pglib_opf_case118_ieee_corrected.m"),
    "case500_goc": osp.join(DATAKIT_GRIDS, "pglib_opf_case500_goc_corrected.m"),
    "case2000_goc": osp.join(DATAKIT_GRIDS, "pglib_opf_case2000_goc_corrected.m"),
    "Texas2k_case1_2016summerpeak": osp.join(
        DATAKIT_FILE_GRIDS,
        "Texas2k_case1_2016summerpeak.m",
    ),
    "case9241_pegase": osp.join(DATAKIT_GRIDS, "pglib_opf_case9241_pegase_corrected.m"),
}


def mfile_operator_stats(name, mpath):
    """Operators-only path: boundary + Kron-Schur gates from the case file."""
    bus, branch, gen = load_case(mpath)
    Y, _, _, _ = ybus(bus, branch)
    # bus[:,1]: MATPOWER type (1 PQ, 2 PV, 3 REF); bus[:,9]: base_kv
    bus0 = pd.DataFrame(
        {
            "PV": (bus[:, 1] == 2).astype(int),
            "REF": (bus[:, 1] == 3).astype(int),
            "vn_kv": bus[:, 9],
        },
    )
    boundary, interior = select_boundary(bus0)
    ops, lu, stats = build_operators(Y, boundary, interior)
    stats["source"] = "mfile"
    return stats


def harmonic_quality(root, name):
    """Median |V_hat - V_true| on interior buses using the sparsified P.

    V_hat = V_aff + P_sp V_B with TRUE boundary voltages: measures the
    linearized-interior iterate + sparsification quality (diagnostic, not a
    gate; the model consumes V_hat as features through a residual stream).
    """
    cache = torch.load(
        osp.join(root, "processed", hierarchy_cache_name()),
        weights_only=True,
    )
    bus = pd.read_parquet(osp.join(root, "raw", "bus_data.parquet"))
    b_idx = cache["boundary_idx"].numpy()
    pe = cache["prolong_edge_index"].numpy()
    pv = cache["prolong_edge_attr"].numpy()
    P = sp.coo_matrix(
        (pv[:, 0] + 1j * pv[:, 1], (pe[1], pe[0])),
    ).tocsr()  # bus x cbus
    errs = []
    for s, bs in bus.groupby("scenario"):
        bs = bs.sort_values("bus")
        vm = bs["Vm"].to_numpy()
        va = np.deg2rad(bs["Va"].to_numpy())
        V = vm * np.exp(1j * va)
        v_aff = cache["v_aff"][s].numpy()
        vhat = (v_aff[:, 0] + 1j * v_aff[:, 1]) + P @ V[b_idx]
        interior = np.setdiff1d(np.arange(len(vm)), b_idx)
        errs.append(np.abs(vhat[interior] - V[interior]))
    errs = np.concatenate(errs)
    return {
        "vhat_abs_err_median": float(np.median(errs)),
        "vhat_abs_err_p90": float(np.percentile(errs, 90)),
        "vhat_abs_err_max": float(errs.max()),
    }


if __name__ == "__main__":
    results = {}
    for name, mpath in GRIDS.items():
        root = osp.join(REPO, "data", name)
        t0 = time.perf_counter()
        try:
            if osp.exists(osp.join(root, "raw", "bus_data.parquet")):
                stats = build_grid_hierarchy(root)
                stats["source"] = "parquet+v_aff"
                stats.update(harmonic_quality(root, name))
            else:
                stats = mfile_operator_stats(name, mpath)
            stats["total_time_s"] = time.perf_counter() - t0
            stats["status"] = "PASS"
        except AssertionError as e:
            stats = {"status": "HARD_ASSERT_FAIL", "error": str(e)}
        results[name] = stats
        print(f"[{stats['status']}] {name}: {json.dumps(stats)}", flush=True)

    os.makedirs(osp.join(HERE, "results"), exist_ok=True)
    out = osp.join(HERE, "results", "r002_precompute.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")
    failed = [n for n, s in results.items() if s["status"] != "PASS"]
    if failed:
        print(f"HARD STOP: {failed}")
        sys.exit(1)
    print("ALL GATES PASS")
