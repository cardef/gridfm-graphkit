# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""HELM-unpool pilot: is the harmonic extension the germ of a boundary-
conditioned HELM series, and do higher orders buy reconstruction accuracy?

Setting (matches r002 harmonic_quality, but with FULL dense P — operator
ceiling, no sparsification confound): reconstruct interior V from TRUE
boundary V_b and interior loads S_I, per scenario, per grid.

Embedding (interior PQ-only, shunts kept inside Y_ii, V_b fixed):
    Y_ii V(s) + Y_ib V_b = s * conj(S_I) .* W(s),   W(s) = 1 / conj(V)(s)
Recurrence (per-bus elementwise, one LU solve per order):
    c_0 = -Y_ii^-1 Y_ib V_b                      (harmonic extension)
    w_0 = 1/conj(c_0);  w_n = -(1/conj(c_0)) sum_{t=0}^{n-1} w_t conj(c_{n-t})
    c_{n+1} = Y_ii^-1 (conj(S_I) .* w_n)
Variants measured:
    H0    : c_0
    REPO  : c_0 + Y_ii^-1 conj(S_I)              (current v_aff, w_0 ~ 1)
    HELM1 : c_0 + c_1                            (= 1 implicit-Z-bus step)
    HELM2 : + c_2
    HELM3 : + c_3
    PADE  : c_0 + c_1 / (1 - c_2/c_1)            (per-bus [1/1] Pade tail)
    FP2   : 2 fixed-point steps V <- c_0 + Y_ii^-1(conj(S_I)/conj(V))
Plus: sanity residual of the TRUE solution in these conventions (fail loud
if parquet sign/base conventions differ), and top-k mass retention of
M = Y_ii^-1 (the operator HELM1 needs at runtime) + HELM1/2 error with
sparsified M.

Runtime-feasible variants (no dense M):
    JACk  : HELM2 with every Y_ii solve replaced by K damped-Jacobi sweeps
            (K sparse matvecs with Y_ii = K message-passing rounds)
    Psp   : HELM2 with c_0 built from the SPARSIFIED P (k=16->32, the
            operator the model actually uses) and exact tail solves --
            the deliverable ceiling including P-sparsification error.

Writes idea-stage/helm_unpool_results.json and prints a summary table.
"""

import json
import os.path as osp
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, ".."))
sys.path.insert(0, REPO)

from gridfm_graphkit.datasets.hierarchy import (  # noqa: E402
    BASE_MVA,
    _load_raw,
    _ybus_csc,
    select_boundary,
)

GRIDS = [
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case118_ieee",
    "case500_goc",
    "case2000_goc",
    "Texas2k_case1_2016summerpeak",
]
MAX_SCEN = 256
M_KS = (16, 32, 64)
JAC_KS = (4, 8, 16, 32)
JAC_OMEGA = 0.8


def jacobi(Yii, dinv, rhs, K, omega=JAC_OMEGA):
    """K damped-Jacobi sweeps for Yii x = rhs, x0 = D^-1 rhs (K sparse matvecs)."""
    x = dinv * rhs
    for _ in range(K):
        x = x + omega * dinv * (rhs - Yii @ x)
    return x


def summarize(errs):
    e = np.concatenate(errs)
    return {
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "max": float(e.max()),
    }


def run_grid(root):
    bus, gen, ybus = _load_raw(root)
    scenarios = np.sort(bus["scenario"].unique())[:MAX_SCEN]
    bus0 = bus[bus["scenario"] == scenarios[0]].sort_values("bus")
    nb = len(bus0)
    ygrp = ybus.groupby("scenario")
    Y = _ybus_csc(ygrp.get_group(scenarios[0]).sort_values(["index1", "index2"]), nb)
    boundary, interior = select_boundary(bus0)
    nb_i = len(interior)

    Yib = Y[np.ix_(interior, boundary)].tocsc()
    Yii = Y[np.ix_(interior, interior)].tocsc()
    lu = spla.splu(Yii)
    dinv = 1.0 / Yii.diagonal()
    Yii_csr = Yii.tocsr()

    # sparsified P exactly as build_operators does (k=16 -> 32, 95% mass)
    P_full = -lu.solve(Yib.toarray())
    absP = np.abs(P_full)
    P_sp = None
    for k in (16, 32):
        kk = min(k, len(boundary))
        idx = np.argpartition(absP, -kk, axis=1)[:, -kk:]
        keep = np.zeros_like(absP, dtype=bool)
        np.put_along_axis(keep, idx, True, axis=1)
        if absP[keep].sum() / absP.sum() >= 0.95:
            P_sp = np.where(keep, P_full, 0)
            break
    if P_sp is None:
        P_sp = np.where(keep, P_full, 0)  # last k, mirrors hard-assert regime

    # dense M = Yii^-1 for the sparsified-runtime feasibility check
    M = lu.solve(np.eye(nb_i, dtype=complex))
    absM = np.abs(M)
    m_mass = {}
    m_sparse = {}
    for k in M_KS:
        kk = min(k, nb_i)
        idx = np.argpartition(absM, -kk, axis=1)[:, -kk:]
        keep = np.zeros_like(absM, dtype=bool)
        np.put_along_axis(keep, idx, True, axis=1)
        m_mass[k] = float(absM[keep].sum() / absM.sum())
        m_sparse[k] = np.where(keep, M, 0)

    bgrp = bus.groupby("scenario")
    errs = {v: [] for v in ["H0", "REPO", "HELM1", "HELM2", "HELM3", "PADE", "FP2"]}
    errs.update({f"HELM1_Mk{k}": [] for k in M_KS})
    errs.update({f"HELM2_Mk{k}": [] for k in M_KS})
    errs.update({f"HELM2_JAC{k}": [] for k in JAC_KS})
    errs.update({v: [] for v in ["H0_Psp", "REPO_Psp", "HELM2_Psp"]})
    resid, minc0, loads = [], [], []
    per_scen_med = {v: [] for v in ["H0", "REPO", "HELM2"]}

    t0 = time.perf_counter()
    for s in scenarios:
        bs = bgrp.get_group(s).sort_values("bus")
        vm = bs["Vm"].to_numpy()
        va = np.deg2rad(bs["Va"].to_numpy())
        V = vm * np.exp(1j * va)
        Vb, Vi = V[boundary], V[interior]
        S_I = (
            -bs["Pd"].to_numpy()[interior] - 1j * bs["Qd"].to_numpy()[interior]
        ) / BASE_MVA
        cS = np.conj(S_I)

        # sanity: true solution must satisfy the interior equation as written
        resid.append(
            np.abs(Yii @ Vi + Yib @ Vb - cS / np.conj(Vi)),
        )

        c0 = lu.solve(-(Yib @ Vb))
        minc0.append(float(np.abs(c0).min()))
        w0 = 1.0 / np.conj(c0)
        c1 = lu.solve(cS * w0)
        w1 = -w0 * np.conj(c1) * w0
        c2 = lu.solve(cS * w1)
        w2 = -w0 * (w0 * np.conj(c2) + w1 * np.conj(c1))
        c3 = lu.solve(cS * w2)

        v = {
            "H0": c0,
            "REPO": c0 + lu.solve(cS),
            "HELM1": c0 + c1,
            "HELM2": c0 + c1 + c2,
            "HELM3": c0 + c1 + c2 + c3,
        }
        # per-bus [1/1] Pade on the tail; fall back to partial sum where c1 ~ 0
        den = 1.0 - np.divide(c2, c1, out=np.zeros_like(c2), where=np.abs(c1) > 1e-14)
        pade = np.where(
            (np.abs(c1) > 1e-14) & (np.abs(den) > 1e-6),
            c0 + c1 / np.where(np.abs(den) > 1e-6, den, 1),
            c0 + c1 + c2,
        )
        v["PADE"] = pade
        vfp = c0 + c1  # FP1 == HELM1
        vfp = c0 + lu.solve(cS / np.conj(vfp))
        v["FP2"] = vfp
        for k in M_KS:
            c1m = m_sparse[k] @ (cS * w0)
            w1m = -w0 * np.conj(c1m) * w0
            v[f"HELM1_Mk{k}"] = c0 + c1m
            v[f"HELM2_Mk{k}"] = c0 + c1m + m_sparse[k] @ (cS * w1m)

        # runtime-feasible: damped-Jacobi solves (K sparse matvecs each)
        for K in JAC_KS:
            c1j = jacobi(Yii_csr, dinv, cS * w0, K)
            w1j = -w0 * np.conj(c1j) * w0
            v[f"HELM2_JAC{K}"] = c0 + c1j + jacobi(Yii_csr, dinv, cS * w1j, K)

        # deliverable ceiling: sparsified-P c0, exact tail
        c0s = P_sp @ Vb
        w0s = 1.0 / np.conj(c0s)
        c1s = lu.solve(cS * w0s)
        w1s = -w0s * np.conj(c1s) * w0s
        v["H0_Psp"] = c0s
        v["REPO_Psp"] = c0s + lu.solve(cS)
        v["HELM2_Psp"] = c0s + c1s + lu.solve(cS * w1s)

        for name, vh in v.items():
            errs[name].append(np.abs(vh - Vi))
        for name in per_scen_med:
            per_scen_med[name].append(float(np.median(np.abs(v[name] - Vi))))
        loads.append(float(np.abs(S_I).sum()))

    out = {
        "n_bus": nb,
        "n_interior": nb_i,
        "n_scen": len(scenarios),
        "true_resid_median": float(np.median(np.concatenate(resid))),
        "min_abs_c0": float(min(minc0)),
        "M_topk_mass": m_mass,
        "time_s": time.perf_counter() - t0,
        "err": {name: summarize(e) for name, e in errs.items()},
    }
    # error vs load terciles (per-scenario medians)
    loads = np.array(loads)
    if len(loads) >= 9:
        q = np.quantile(loads, [1 / 3, 2 / 3])
        terc = np.digitize(loads, q)
        out["load_terciles"] = {
            name: [float(np.median(np.array(m)[terc == t])) for t in range(3)]
            for name, m in per_scen_med.items()
        }
    return out


if __name__ == "__main__":
    results = {}
    for name in GRIDS:
        root = osp.join(REPO, "data", name)
        if not osp.exists(osp.join(root, "raw", "bus_data.parquet")):
            print(f"[skip] {name}: no local data")
            continue
        results[name] = run_grid(root)
        r = results[name]
        print(
            f"[{name}] N={r['n_bus']} resid={r['true_resid_median']:.2e} "
            f"min|c0|={r['min_abs_c0']:.3f} Mmass k16/32/64="
            f"{r['M_topk_mass'][16]:.3f}/{r['M_topk_mass'][32]:.3f}/{r['M_topk_mass'][64]:.3f}",
            flush=True,
        )
        for v in [
            "H0",
            "REPO",
            "HELM1",
            "HELM2",
            "HELM3",
            "PADE",
            "FP2",
            "HELM1_Mk32",
            "HELM2_Mk32",
            "HELM2_JAC4",
            "HELM2_JAC8",
            "HELM2_JAC16",
            "HELM2_JAC32",
            "H0_Psp",
            "REPO_Psp",
            "HELM2_Psp",
        ]:
            e = r["err"][v]
            print(
                f"    {v:12s} med={e['median']:.2e} p90={e['p90']:.2e} max={e['max']:.2e}",
            )
    out = osp.join(HERE, "helm_unpool_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")
