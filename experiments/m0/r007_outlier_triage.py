# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R007 (M0 data triage, Amendment A1): locate the case2000 unpool outlier.

The HELM pilot (idea-stage/HELM_UNPOOL_NOTE.md, F5) found one persistent
case2000 reconstruction outlier: max |V_hat - V| ~= 1.07e0 p.u. in EVERY
unpool variant including H0 = P V_b (injection-insensitive). Suspect: a
fast-PF non-converged sample (the grid has a known 25% fast-PF yield) or a
degenerate bus.

Diagnosis path:
1. per-(scenario, interior bus) error of H0 with dense P and TRUE V_b --
   the injection-free reconstruction, so any large error is either a bad
   solution (PF not converged) or a bad operator row;
2. top-10 offending (scenario, bus) pairs: one bus recurring across
   scenarios => operator/bus problem; one scenario across buses => bad
   sample;
3. AC power-mismatch check per scenario at gen-free buses,
   |S_spec - V conj(Y V)| with S_spec = -(Pd + jQd)/base: a converged PF
   solution has ~0 mismatch; a non-converged sample shows a large one;
4. diagnostics for the implicated bus (degree, |Yii diag|, kV, load, |V|).

Gate metrics stay median/p90 regardless (pre-registered); this decides
whether to drop specific scenarios at M1 data prep.

Usage:  ../.venv/bin/python experiments/m0/r007_outlier_triage.py
"""

import json
import os.path as osp
import sys

import numpy as np
import scipy.sparse.linalg as spla

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from gridfm_graphkit.datasets.hierarchy import (  # noqa: E402
    BASE_MVA,
    _load_raw,
    _ybus_csc,
    select_boundary,
)

GRID = "case2000_goc"
TOP_K = 10


def main():
    root = osp.join(REPO, "data", GRID)
    bus, gen, ybus = _load_raw(root)
    scenarios = np.sort(bus["scenario"].unique())
    bus0 = bus[bus["scenario"] == scenarios[0]].sort_values("bus")
    nb = len(bus0)
    Y = _ybus_csc(
        ybus[ybus["scenario"] == scenarios[0]].sort_values(["index1", "index2"]),
        nb,
    )
    boundary, interior = select_boundary(bus0)
    Yib = Y[np.ix_(interior, boundary)].tocsc()
    Yii = Y[np.ix_(interior, interior)].tocsc()
    lu = spla.splu(Yii)
    P = -lu.solve(Yib.toarray())  # dense: no sparsification confound
    Y_csr = Y.tocsr()

    gen = gen[gen["in_service"] == 1]
    gen_buses = np.unique(gen["bus"].to_numpy().astype(int))
    genfree = np.setdiff1d(np.arange(nb), gen_buses)

    bgrp = bus.groupby("scenario")
    rows = []  # (err, scenario, interior_pos)
    mismatch = {}  # scenario -> max |S| mismatch at gen-free buses
    for s in scenarios:
        bs = bgrp.get_group(s).sort_values("bus")
        V = bs["Vm"].to_numpy() * np.exp(1j * np.deg2rad(bs["Va"].to_numpy()))
        err = np.abs(P @ V[boundary] - V[interior])
        top = np.argpartition(err, -3)[-3:]
        rows += [(float(err[i]), int(s), int(i)) for i in top]

        S_calc = V * np.conj(Y_csr @ V)
        S_spec = (-bs["Pd"].to_numpy() - 1j * bs["Qd"].to_numpy()) / BASE_MVA
        mm = np.abs(S_spec[genfree] - S_calc[genfree])
        mismatch[int(s)] = {"max": float(mm.max()), "median": float(np.median(mm))}

    rows.sort(reverse=True)
    top_rows = [
        {
            "err": e,
            "scenario": s,
            "bus": int(interior[i]),
            "pf_mismatch_max": mismatch[s]["max"],
            "pf_mismatch_median": mismatch[s]["median"],
        }
        for e, s, i in rows[:TOP_K]
    ]

    # is it one bus or one scenario?
    top_buses = [r["bus"] for r in top_rows]
    top_scens = [r["scenario"] for r in top_rows]

    # bus diagnostics for the worst offender
    b = top_rows[0]["bus"]
    s = top_rows[0]["scenario"]
    bs = bgrp.get_group(s).sort_values("bus")
    deg = int(np.diff(Y_csr.indptr)[b]) - 1
    ipos = int(np.where(interior == b)[0][0])
    diag = {
        "bus": b,
        "scenario": s,
        "degree": deg,
        "abs_yii_diag": float(np.abs(Yii.diagonal()[ipos])),
        "vn_kv": float(bs["vn_kv"].to_numpy()[b]),
        "Pd_mw": float(bs["Pd"].to_numpy()[b]),
        "Qd_mvar": float(bs["Qd"].to_numpy()[b]),
        "vm_true": float(bs["Vm"].to_numpy()[b]),
        "va_true_deg": float(bs["Va"].to_numpy()[b]),
        "is_gen_free": bool(b in genfree),
        "pf_mismatch_at_bus": float(
            np.abs(
                (-bs["Pd"].to_numpy()[b] - 1j * bs["Qd"].to_numpy()[b]) / BASE_MVA
                - (bs["Vm"].to_numpy() * np.exp(1j * np.deg2rad(bs["Va"].to_numpy())))[
                    b
                ]
                * np.conj(
                    Y_csr[b]
                    @ (
                        bs["Vm"].to_numpy()
                        * np.exp(1j * np.deg2rad(bs["Va"].to_numpy()))
                    ),
                )[0],
            ),
        )
        if b in genfree
        else None,
    }

    # degenerate-solution check: the power mismatch is BLIND at V=0
    # (S = V conj(I) = 0 identically), so also check KCL in current
    # coordinates at the worst bus: |Y[b,:] V| must be ~0 for a physically
    # consistent dead bus (it is not, if the neighbor is energized and the
    # branch is in service -- fixed-Y across scenarios says it is).
    V_s = bs["Vm"].to_numpy() * np.exp(1j * np.deg2rad(bs["Va"].to_numpy()))
    diag["kcl_current_residual_at_bus"] = float(np.abs((Y_csr[b] @ V_s)[0]))
    # how often is this bus dead, and is it alone?
    vm_all = bus[bus["bus"] == b]["Vm"].to_numpy()
    diag["n_scen_dead"] = int((np.abs(vm_all) < 1e-3).sum())
    dead_any = bus[np.abs(bus["Vm"]) < 1e-3]["bus"].nunique()
    diag["n_buses_ever_dead_in_grid"] = int(dead_any)

    all_mm_max = np.array([m["max"] for m in mismatch.values()])
    out = {
        "grid": GRID,
        "n_scenarios": int(len(scenarios)),
        "top_offenders": top_rows,
        "unique_top_buses": sorted(set(top_buses)),
        "unique_top_scenarios": sorted(set(top_scens)),
        "worst_bus_diagnostics": diag,
        "pf_mismatch_max_over_scenarios": {
            "median": float(np.median(all_mm_max)),
            "p90": float(np.percentile(all_mm_max, 90)),
            "max": float(all_mm_max.max()),
        },
    }
    # verdict heuristic, stated not asserted
    one_bus = len(set(top_buses)) <= 2 and len(set(top_scens)) > 2
    dead_bus = one_bus and abs(diag["vm_true"]) < 1e-3
    kcl_bad = diag["kcl_current_residual_at_bus"] > 1e-6
    bad_pf = top_rows[0]["pf_mismatch_max"] > 10 * np.median(all_mm_max)
    if dead_bus and kcl_bad:
        out["verdict"] = (
            "degenerate fast-PF solution: zero-load leaf bus collapsed to "
            "the V=0 branch (power mismatch blind there, KCL violated in "
            "current coordinates); data artifact, not model error. M1 data "
            "prep: flag scenarios where a zero-load bus has |V|<0.1 p.u."
        )
    elif one_bus:
        out["verdict"] = "recurring bus (operator/bus problem)"
    elif bad_pf:
        out["verdict"] = "bad scenario (fast-PF non-converged sample)"
    else:
        out["verdict"] = "mixed/inconclusive -- inspect top_offenders"

    path = osp.join(HERE, "results", "r007_outlier_triage.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
