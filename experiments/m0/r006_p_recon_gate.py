# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R006 (M0 gate-integrity patch, Amendment A1): P reconstruction gate.

The M0 sparsification deliverable gated on retained |P| mass >= 95%; the
HELM pilot (idea-stage/HELM_UNPOOL_NOTE.md, F3) showed mass is the wrong
proxy on case2000: 96.6% mass at k=16 coexists with ~2.6e-2 p.u. median
reconstruction error |P_sp V_b - P V_b|, the same order as the linearization
error, dominating the unpool budget on the M2-gate grid.

This script rebuilds every local hierarchy cache through the amended gate in
gridfm_graphkit.datasets.hierarchy.build_operators: per-grid k escalation
16 -> 32 -> 64 -> dense P until BOTH the mass assert (>=95%) and the new
reconstruction assert (median <= RECON_ASSERT = 1e-3 p.u. over <=64 sample
scenarios, true V_b) pass. It reports the full escalation table (before =
k chosen by mass only, after = k chosen by mass+recon) and re-measures the
affine unpool quality (harmonic_quality) on the rebuilt caches.

Honest expectation (pre-registered): R006 alone does NOT materially improve
the affine unpool error on case2000 -- the linearization error is the same
size (pilot REPO with dense P: 3.0e-2). Its purpose is falsifiability: it
makes R020 interpretable and R021 (HELM2, where the P floor would cap the
arm at 2.6e-2) falsifiable.

Consequence tracked here: a k change alters nnz_prolong, so the R003/R010
matched-FLOP matrix must be regenerated (this script prints the delta).

Usage:  ../.venv/bin/python experiments/m0/r006_p_recon_gate.py
"""

import json
import os
import os.path as osp
import sys

import torch

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from r002_precompute import GRIDS, harmonic_quality  # noqa: E402
from gridfm_graphkit.datasets.hierarchy import (  # noqa: E402
    build_grid_hierarchy,
    hierarchy_cache_name,
)


def old_nnz(root):
    path = osp.join(root, "processed", hierarchy_cache_name())
    if not osp.exists(path):
        return None
    cache = torch.load(path, weights_only=True)
    return int(cache["prolong_edge_index"].shape[1])


if __name__ == "__main__":
    results = {}
    for name in GRIDS:
        root = osp.join(REPO, "data", name)
        if not osp.exists(osp.join(root, "raw", "bus_data.parquet")):
            results[name] = {"status": "SKIP", "reason": "no local scenario data"}
            print(f"[SKIP] {name}: no local data (cluster re-run required)")
            continue
        nnz_before = old_nnz(root)
        try:
            stats = build_grid_hierarchy(root)
            stats.update(harmonic_quality(root, name))
            stats["status"] = "PASS"
        except AssertionError as e:
            stats = {"status": "HARD_ASSERT_FAIL", "error": str(e)}
            results[name] = stats
            print(f"[FAIL] {name}: {e}")
            continue
        nnz_after = old_nnz(root)
        stats["nnz_prolong_before"] = nnz_before
        stats["nnz_prolong_after"] = nnz_after
        results[name] = stats

        esc = stats["p_escalation"]
        print(f"[PASS] {name}: k={stats['k_used']} nnz {nnz_before} -> {nnz_after}")
        for step in esc:
            print(
                f"    k={step['k']:4d} mass={step['mass']:.4f} "
                f"recon_med={step.get('recon_median', float('nan')):.3e} "
                f"recon_p90={step.get('recon_p90', float('nan')):.3e}",
            )
        print(
            f"    affine unpool (true V_b): med="
            f"{stats['vhat_abs_err_median']:.3e} "
            f"p90={stats['vhat_abs_err_p90']:.3e}",
        )

    results_root = os.environ.get("GRIDFM_RESULTS_ROOT", osp.join(HERE, "results"))
    os.makedirs(results_root, exist_ok=True)
    out = osp.join(results_root, "r006_p_recon_gate.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")
    failed = [n for n, s in results.items() if s["status"] == "HARD_ASSERT_FAIL"]
    if failed:
        print(f"HARD STOP: {failed}")
        sys.exit(1)
    changed = [
        n
        for n, s in results.items()
        if s["status"] == "PASS"
        and s["nnz_prolong_before"] not in (None, s["nnz_prolong_after"])
    ]
    if changed:
        print(f"nnz_prolong changed for {changed}: regenerate R003/R010 matrix")
    print("R006 GATE PASS on all local grids")
