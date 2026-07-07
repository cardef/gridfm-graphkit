# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Kron-Schur hierarchy operators for the 2-level hierarchical GNN.

Precomputes, per grid (fixed topology / fixed Y across scenarios):

- boundary selection B = REF + PV (generator) buses + HV backbone
  (kV threshold chosen so |B|/N is closest to ``target_frac``),
- coarse admittance ``Y_red = Y_bb - Y_bi Y_ii^-1 Y_ib`` (Schur complement),
  thresholded at ``|Y| > tol`` (directed edges, (G, B) channels),
- prolongation ``P = -Y_ii^-1 Y_ib`` sparsified to the top-k entries per
  interior row, with a HARD ASSERT that the retained |P| mass is >= 95%
  (k=16, escalated to 32) -- the M0 stop condition from the experiment plan,
- per-scenario affine term ``V_aff = Y_ii^-1 conj(S_I)`` (harmonic-extension
  constant; interior buses are PQ-only, so S_I is a pure PF input),
- per-scenario Ward-restricted coarse injections
  ``I_c = conj(S_B_masked) - Y_bi V_aff`` (masked = PF-input convention:
  slack Pg and all Qg are unknown at inference and enter as 0).

All physical quantities are in p.u. on the case base (baseMVA=100), matching
y_bus_data.parquet; they are attached to HeteroData *after* the learned
normalizer runs and are deliberately not rescaled by it -- the coarse
physical channel carries the exactness claims on the case base.

The results are cached in one file per grid under ``<root>/processed/`` and
attached to each sample at runtime by :class:`AddHierarchy`.
"""

import json
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch_geometric.transforms import BaseTransform

BASE_MVA = 100.0  # case base of y_bus_data.parquet / raw power columns
MASS_ASSERT = 0.95
K_POOL = (16, 32)
Y_RED_TOL = 1e-3
TARGET_BOUNDARY_FRAC = 0.27  # aim 25-30% per FINAL_PROPOSAL.md


def hierarchy_cache_name(target_frac=TARGET_BOUNDARY_FRAC, tol=Y_RED_TOL):
    return f"hierarchy_b{target_frac:g}_tol{tol:g}.pt"


def _load_raw(root):
    raw = osp.join(root, "raw")
    bus = pd.read_parquet(osp.join(raw, "bus_data.parquet"))
    gen = pd.read_parquet(osp.join(raw, "gen_data.parquet"))
    ybus = pd.read_parquet(osp.join(raw, "y_bus_data.parquet"))
    return bus, gen, ybus


def _ybus_csc(ydf, nb):
    Y = sp.coo_matrix(
        (
            ydf["G"].to_numpy() + 1j * ydf["B"].to_numpy(),
            (ydf["index1"].to_numpy(), ydf["index2"].to_numpy()),
        ),
        shape=(nb, nb),
    )
    return Y.tocsc()


def select_boundary(bus0, target_frac=TARGET_BOUNDARY_FRAC):
    """Boundary = REF + PV buses + HV buses (kV threshold closest to target).

    Interior is PQ-only by construction, so interior injections are pure PF
    inputs (loads).
    """
    nb = len(bus0)
    gen_mask = (bus0["PV"].to_numpy() == 1) | (bus0["REF"].to_numpy() == 1)
    vn = bus0["vn_kv"].to_numpy()
    # candidates: gen-only (kv threshold = +inf) plus gen | HV(kv >= t);
    # gen-only covers grids with constant/absent kV data (e.g. case14).
    best = (gen_mask.sum() / nb, gen_mask)
    for kv in np.unique(vn):
        cand = gen_mask | (vn >= kv)
        frac = cand.sum() / nb
        if abs(frac - target_frac) < abs(best[0] - target_frac):
            best = (frac, cand)
    boundary = np.flatnonzero(best[1])
    interior = np.flatnonzero(~best[1])
    return boundary, interior


def build_operators(Y, boundary, interior, tol=Y_RED_TOL):
    """Core Kron-Schur operator computation for a fixed admittance matrix.

    Returns (ops, lu, stats): sparse operator tensors ready for HeteroData
    relations, the Yii LU factorization (for per-scenario V_aff solves), and
    the M0 gate statistics. Raises AssertionError if the sparsified
    prolongation retains < 95% of |P| mass (the declared stop condition).
    """
    t0 = time.perf_counter()
    nb_c, nb_i = len(boundary), len(interior)

    Ybb = Y[np.ix_(boundary, boundary)]
    Ybi = Y[np.ix_(boundary, interior)].tocsc()
    Yib = Y[np.ix_(interior, boundary)].tocsc()
    Yii = Y[np.ix_(interior, interior)].tocsc()
    lu = spla.splu(Yii)

    # --- prolongation P = -Yii^-1 Yib (dense interior x boundary) ---
    P = -lu.solve(Yib.toarray())
    absP = np.abs(P)
    total_mass = absP.sum()
    retained_mass, k_used, keep = None, None, None
    for k in K_POOL:
        kk = min(k, nb_c)
        thr_idx = np.argpartition(absP, -kk, axis=1)[:, -kk:]
        keep_k = np.zeros_like(absP, dtype=bool)
        np.put_along_axis(keep_k, thr_idx, True, axis=1)
        m = absP[keep_k].sum() / total_mass
        retained_mass, k_used, keep = m, kk, keep_k
        if m >= MASS_ASSERT:
            break
    # HARD ASSERT (M0 stop condition): sparsified prolongation must keep >=95%
    assert retained_mass >= MASS_ASSERT, (
        f"P top-k sparsification retains {retained_mass:.4f} < "
        f"{MASS_ASSERT} of |P| mass at k={k_used}. STOP (M0 kill criterion)."
    )

    ii, jj = np.nonzero(keep)  # ii: interior row, jj: boundary col
    # prolong relation cbus->bus: identity on boundary + retained interior rows
    prol_src = np.concatenate([np.arange(nb_c), jj])
    prol_dst = np.concatenate([boundary, interior[ii]])
    prol_val = np.concatenate([np.ones(nb_c, dtype=complex), P[ii, jj]])

    # --- coarse network Y_red = Ybb - Ybi Yii^-1 Yib = Ybb + Ybi P ---
    Yred = Ybb.toarray() + Ybi.toarray() @ P
    ydiag_red = np.diag(Yred).copy()
    off = Yred.copy()
    np.fill_diagonal(off, 0)
    keep_e = np.abs(off) > tol
    kept_edge_mass = np.abs(off[keep_e]).sum() / max(np.abs(off).sum(), 1e-30)
    ei, ej = np.nonzero(keep_e)
    coarse_density = keep_e.sum() / max(nb_c * (nb_c - 1), 1)

    # --- Ward restriction W_int = -Ybi Yii^-1 (kept for provenance) ---
    Wint = -lu.solve(Ybi.toarray().T, trans="T").T

    ops = {
        "boundary_idx": torch.tensor(boundary, dtype=torch.long),
        # message direction j -> i carrying Y_ij: V_j enters the current
        # equation of row i (matters when taps make Y_red non-symmetric)
        "coarse_edge_index": torch.tensor(np.stack([ej, ei]), dtype=torch.long),
        "coarse_edge_attr": torch.tensor(
            np.stack([Yred[ei, ej].real, Yred[ei, ej].imag], axis=1),
            dtype=torch.float,
        ),
        "prolong_edge_index": torch.tensor(
            np.stack([prol_src, prol_dst]),
            dtype=torch.long,
        ),
        "prolong_edge_attr": torch.tensor(
            np.stack([prol_val.real, prol_val.imag], axis=1),
            dtype=torch.float,
        ),
        "ydiag_red": torch.tensor(
            np.stack([ydiag_red.real, ydiag_red.imag], axis=1),
            dtype=torch.float,
        ),
        "wint_dense": torch.tensor(
            np.stack([Wint.real, Wint.imag], axis=0),
            dtype=torch.float,
        ),
    }
    stats = {
        "n_bus": nb_c + nb_i,
        "n_boundary": nb_c,
        "boundary_frac": nb_c / (nb_c + nb_i),
        "k_used": int(k_used),
        "p_retained_mass": float(retained_mass),
        "coarse_offdiag_density": float(coarse_density),
        "coarse_edge_mass_kept": float(kept_edge_mass),
        "operator_time_s": time.perf_counter() - t0,
        "tol": tol,
    }
    return ops, lu, stats


def build_grid_hierarchy(root, target_frac=TARGET_BOUNDARY_FRAC, tol=Y_RED_TOL):
    """Precompute operators + per-scenario tensors for one grid. Returns stats."""
    bus, gen, ybus = _load_raw(root)
    scenarios = np.sort(bus["scenario"].unique())
    bus0 = bus[bus["scenario"] == scenarios[0]].sort_values("bus")
    nb = len(bus0)

    # --- fixed-Y check: operators are per-grid; refuse per-scenario Y ---
    ygrp = ybus.groupby("scenario")
    y0 = ygrp.get_group(scenarios[0]).sort_values(["index1", "index2"])
    ref_key = (
        y0["index1"].to_numpy(),
        y0["index2"].to_numpy(),
        y0["G"].to_numpy(),
        y0["B"].to_numpy(),
    )
    for s in scenarios[1:]:
        ys = ygrp.get_group(s).sort_values(["index1", "index2"])
        same = (
            len(ys) == len(y0)
            and np.array_equal(ys["index1"].to_numpy(), ref_key[0])
            and np.array_equal(ys["index2"].to_numpy(), ref_key[1])
            and np.allclose(ys["G"].to_numpy(), ref_key[2])
            and np.allclose(ys["B"].to_numpy(), ref_key[3])
        )
        if not same:
            raise AssertionError(
                f"{root}: Y differs between scenario {scenarios[0]} and {s}. "
                "Per-grid Kron-Schur operators require fixed topology/admittance "
                "(regenerate data with topology/admittance perturbations off, "
                "or extend the cache to per-topology keys).",
            )

    Y = _ybus_csc(y0, nb)
    boundary, interior = select_boundary(bus0, target_frac)
    nb_c = len(boundary)
    ops, lu, op_stats = build_operators(Y, boundary, interior, tol)
    Ybi = Y[np.ix_(boundary, interior)].tocsc()

    # --- per-scenario V_aff and coarse injections ---
    bgrp = bus.groupby("scenario")
    gen = gen[gen["in_service"] == 1]
    ggrp = gen.groupby("scenario")
    v_aff = np.zeros((len(scenarios), nb, 2), dtype=np.float32)
    cbus_x = np.zeros((len(scenarios), nb_c, 6), dtype=np.float32)
    vn_norm = bus0["vn_kv"].to_numpy() / max(bus0["vn_kv"].max(), 1e-9)
    pv0 = bus0["PV"].to_numpy()[boundary].astype(np.float32)
    ref0 = bus0["REF"].to_numpy()[boundary].astype(np.float32)

    t1 = time.perf_counter()
    for si, s in enumerate(scenarios):
        bs = bgrp.get_group(s).sort_values("bus")
        # interior = PQ only: S_I = (-Pd - jQd)/base; I_I = conj(S_I)
        S_I = (
            -bs["Pd"].to_numpy()[interior] - 1j * bs["Qd"].to_numpy()[interior]
        ) / BASE_MVA
        I_I = np.conj(S_I)
        va = lu.solve(I_I)
        v_aff[si, interior, 0] = va.real
        v_aff[si, interior, 1] = va.imag

        # boundary injections, PF-input (masked) convention: Qg unknown (0);
        # Pg unknown (0) for ALL generators at REF buses, mirroring
        # AddPFHeteroMask (which masks every gen connected to a REF bus).
        gs = ggrp.get_group(s)
        ref_bus = bs["REF"].to_numpy().astype(bool)
        gs_known = gs[~ref_bus[gs["bus"].to_numpy().astype(int)]]
        pg_bus = np.zeros(nb)
        np.add.at(
            pg_bus,
            gs_known["bus"].to_numpy().astype(int),
            gs_known["p_mw"].to_numpy(),
        )
        S_B = (
            (pg_bus[boundary] - bs["Pd"].to_numpy()[boundary])
            - 1j * bs["Qd"].to_numpy()[boundary]
        ) / BASE_MVA
        I_c = np.conj(S_B) - Ybi @ va
        S_c = np.conj(I_c)
        cbus_x[si, :, 0] = S_c.real
        cbus_x[si, :, 1] = S_c.imag
        # Vm setpoint known at PV buses only (repo PF mask: REF Vm is masked)
        vm = bs["Vm"].to_numpy()[boundary]
        cbus_x[si, :, 2] = np.where(pv0 == 1, vm, 0.0)
        cbus_x[si, :, 3] = pv0
        cbus_x[si, :, 4] = ref0
        cbus_x[si, :, 5] = vn_norm[boundary]
    t_scen = time.perf_counter() - t1

    cache = dict(ops)
    cache["v_aff"] = torch.tensor(v_aff)
    cache["cbus_x"] = torch.tensor(cbus_x)
    stats = dict(op_stats)
    stats.update(
        {
            "n_scenarios": int(len(scenarios)),
            "per_scenario_time_s": t_scen,
            "target_frac": target_frac,
        },
    )
    cache["meta"] = json.dumps(stats)

    out = osp.join(root, "processed", hierarchy_cache_name(target_frac, tol))
    os.makedirs(osp.dirname(out), exist_ok=True)
    torch.save(cache, out)
    return stats


class AddHierarchy(BaseTransform):
    """Attach precomputed Kron-Schur hierarchy tensors to each sample.

    Adds a ``cbus`` node type with per-scenario physical features, the
    (bus, seeds, cbus) selection relation, the thresholded coarse network
    (cbus, connects, cbus), the sparsified prolongation (cbus, prolong, bus),
    and the per-scenario affine term ``bus.v_aff``.

    The cache file is resolved via :meth:`set_root`, which the datamodule
    calls with the per-network data root before first use.
    """

    def __init__(self, args):
        super().__init__()
        h = getattr(args.data, "hierarchy", None)
        self.target_frac = getattr(h, "target_frac", TARGET_BOUNDARY_FRAC)
        self.tol = getattr(h, "tol", Y_RED_TOL)
        self._root = None
        self._cache = None
        self._seeds = None

    def set_root(self, root):
        self._root = root
        self._cache = None

    def _load(self):
        if self._cache is None:
            if self._root is None:
                raise RuntimeError(
                    "AddHierarchy has no data root; the datamodule must call "
                    "set_root(<root>) after get_task_transforms().",
                )
            path = osp.join(
                self._root,
                "processed",
                hierarchy_cache_name(self.target_frac, self.tol),
            )
            if not osp.exists(path):
                raise FileNotFoundError(
                    f"Hierarchy cache missing: {path}. Run "
                    "gridfm_graphkit.datasets.hierarchy.build_grid_hierarchy first "
                    "(experiments/m0/r002_precompute.py).",
                )
            self._cache = torch.load(path, weights_only=True)
            n_c = self._cache["boundary_idx"].numel()
            self._seeds = torch.stack(
                [self._cache["boundary_idx"], torch.arange(n_c)],
            )
        return self._cache

    def forward(self, data):
        c = self._load()
        idx = int(data["scenario_id"].item())
        # per-scenario features + static coarse self-admittance (G,B) diag
        data["cbus"].x = torch.cat([c["cbus_x"][idx], c["ydiag_red"]], dim=1)
        data["bus"].v_aff = c["v_aff"][idx]
        data["bus", "seeds", "cbus"].edge_index = self._seeds
        data["cbus", "connects", "cbus"].edge_index = c["coarse_edge_index"]
        data["cbus", "connects", "cbus"].edge_attr = c["coarse_edge_attr"]
        data["cbus", "prolong", "bus"].edge_index = c["prolong_edge_index"]
        data["cbus", "prolong", "bus"].edge_attr = c["prolong_edge_attr"]
        return data
