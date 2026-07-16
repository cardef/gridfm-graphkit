"""CPU pilots for idea-creator run 2026-07-06 (no GPU available; training pilots deferred).

Pilot A (I9 kill-criterion): Kron/Schur fill-in density.
  Success metric (declared upfront):
    coarse off-diagonal density < 20%  -> POSITIVE (2-level Kron U-Net viable with sparse MP)
    20-60%                             -> WEAK POSITIVE (coarse level needs attention/dense ops)
    > 60%                              -> NEGATIVE for naive use; thresholded-mass analysis decides
  Also reports: how much off-diagonal admittance mass survives magnitude thresholding
  (if >95% mass in a sparse subset, thresholded Kron pooling is safe).

Pilot B (I1-lite premise): PTDF localization — electrical vs hop distance.
  Success metric (declared upfront):
    log|PTDF| decays with effective-resistance distance clearly better than with hop
    distance (Spearman |rho_elec| - |rho_hop| > 0.10) and median decay length << diameter
    -> POSITIVE signal for electrical-locality premise underlying I1/I9/I3.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.stats import spearmanr
from matpowercaseframes import CaseFrames

RNG = np.random.default_rng(0)


def load_case(path):
    cf = CaseFrames(path)
    bus = cf.bus.to_numpy(dtype=float)
    branch = cf.branch.to_numpy(dtype=float)
    gen = cf.gen.to_numpy(dtype=float)
    return bus, branch, gen


def ybus(bus, branch):
    """Textbook MATPOWER Ybus (in-service branches, taps/shifts, line charging, bus shunts)."""
    nb = bus.shape[0]
    busmap = {int(b): i for i, b in enumerate(bus[:, 0])}
    st = branch[:, 10] > 0
    br = branch[st]
    f = np.array([busmap[int(x)] for x in br[:, 0]])
    t = np.array([busmap[int(x)] for x in br[:, 1]])
    ys = 1.0 / (br[:, 2] + 1j * br[:, 3])
    bc = br[:, 4]
    tap = np.where(br[:, 8] == 0.0, 1.0, br[:, 8]) * np.exp(1j * np.deg2rad(br[:, 9]))
    ytt = ys + 1j * bc / 2
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    ysh = (
        bus[:, 4] + 1j * bus[:, 5]
    ) / 1.0  # GS+jBS in MVA at V=1pu, baseMVA=100 handled as p.u. in pglib
    Y = sp.csr_matrix(
        (
            np.concatenate([yff, ytt, yft, ytf]),
            (np.concatenate([f, t, f, t]), np.concatenate([f, t, t, f])),
        ),
        shape=(nb, nb),
        dtype=complex,
    )
    Y = Y + sp.diags(
        ysh / 100.0,
    )  # ponytail: baseMVA=100 for all pglib/Texas cases used here
    return Y.tocsc(), f, t, br


def kron_fillin(name, bus, branch, gen, kv_quantile):
    Y, f, t, br = ybus(bus, branch)
    nb = Y.shape[0]
    base_kv = bus[:, 9]
    genbus = {int(g[0]) for g in gen if g[7] > 0}
    busmap = {int(b): i for i, b in enumerate(bus[:, 0])}
    genidx = {busmap[b] for b in genbus if b in busmap}
    # pick the kv threshold whose boundary fraction (HV buses ∪ gen buses) is closest to target
    target = kv_quantile
    best = None
    for kv in np.unique(base_kv):
        cand = set(np.where(base_kv >= kv)[0]) | genidx
        fr = len(cand) / nb
        if best is None or abs(fr - target) < abs(best[1] - target):
            best = (kv, fr, cand)
    kv_thr = best[0]
    boundary = np.array(sorted(best[2]))
    interior = np.setdiff1d(np.arange(nb), boundary)
    frac = len(boundary) / nb
    Ybb = Y[np.ix_(boundary, boundary)]
    Ybi = Y[np.ix_(boundary, interior)]
    Yib = Y[np.ix_(interior, boundary)]
    Yii = Y[np.ix_(interior, interior)].tocsc()
    lu = spla.splu(Yii)
    X = lu.solve(Yib.toarray())  # interior x boundary dense solve
    Yred = Ybb.toarray() - Ybi.toarray() @ X
    off = Yred.copy()
    np.fill_diagonal(off, 0)
    mag = np.abs(off)
    nbb = len(boundary)
    tot_mass = mag.sum()
    dens = {}
    for tol in (1e-6, 1e-4, 1e-3):
        keep = mag > tol
        dens[tol] = (keep.sum() / (nbb * (nbb - 1)), mag[keep].sum() / tot_mass)
    # mass concentration: smallest edge fraction holding 95% of off-diag mass
    flat = np.sort(mag[mag > 0])[::-1]
    cum = np.cumsum(flat) / tot_mass
    k95 = int(np.searchsorted(cum, 0.95)) + 1
    edge_frac_95 = k95 / (nbb * (nbb - 1))
    fine_edges = Y.nnz - nb
    print(
        f"\n[Pilot A] {name}: nb={nb} boundary={nbb} ({frac:.1%}, kv>={kv_thr:.0f} or gen) "
        f"fine offdiag nnz={fine_edges}",
    )
    for tol, (d, m) in dens.items():
        print(f"  tol={tol:g}: coarse offdiag density={d:.1%}, mass kept={m:.4f}")
    print(
        f"  95% of off-diag mass in {edge_frac_95:.2%} of possible coarse edges (k={k95})",
    )
    return dens, edge_frac_95


def ptdf_locality(name, bus, branch, n_src=80):
    """DC PTDF columns for random injection buses; |PTDF| vs hop and vs eff-resistance distance."""
    nb = bus.shape[0]
    busmap = {int(b): i for i, b in enumerate(bus[:, 0])}
    st = branch[:, 10] > 0
    br = branch[st]
    f = np.array([busmap[int(x)] for x in br[:, 0]])
    t = np.array([busmap[int(x)] for x in br[:, 1]])
    x = br[:, 3] * np.where(br[:, 8] == 0.0, 1.0, br[:, 8])  # DC: x * tap
    bsus = 1.0 / x
    A = sp.csr_matrix(
        (np.ones(len(f)), (np.arange(len(f)), f)),
        shape=(len(f), nb),
    ) - sp.csr_matrix((np.ones(len(t)), (np.arange(len(t)), t)), shape=(len(f), nb))
    Bf = sp.diags(bsus) @ A
    B = A.T @ Bf
    adj = sp.csr_matrix((np.ones(len(f)), (f, t)), shape=(nb, nb))
    adj = adj + adj.T
    ncomp, labels = connected_components(adj, directed=False)
    main = np.where(labels == np.bincount(labels).argmax())[0]
    slack = main[0]
    keep = np.setdiff1d(main, [slack])
    Bk = B[np.ix_(keep, keep)].tocsc()
    lu = spla.splu(Bk)
    # effective resistance via dense inverse on main component (nb<=2k ok)
    Linv = np.zeros((nb, nb))
    Linv[np.ix_(keep, keep)] = lu.solve(np.eye(len(keep)))
    dL = np.diag(Linv)
    srcs = RNG.choice(keep, size=min(n_src, len(keep)), replace=False)
    hop = shortest_path(adj, method="D", unweighted=True, indices=srcs)
    rows_h, rows_e, rows_v = [], [], []
    for si, j in enumerate(srcs):
        e = np.zeros(len(keep))
        e[np.searchsorted(keep, j)] = 1.0
        th = np.zeros(nb)
        th[keep] = lu.solve(e)
        flow = np.abs(Bf @ th)  # |PTDF_{l,j}| with slack ref
        # attribute each line to its nearer endpoint's distances from j
        d_h = np.minimum(hop[si, f], hop[si, t])
        r_j = dL[f] + dL[j] - 2 * Linv[f, j]
        mask = (np.isfinite(d_h)) & (d_h > 0) & (flow > 1e-12)
        rows_h.append(d_h[mask])
        rows_e.append(r_j[mask])
        rows_v.append(flow[mask])
    dh = np.concatenate(rows_h)
    de = np.concatenate(rows_e)
    dv = np.log10(np.concatenate(rows_v))
    rho_h = spearmanr(dh, dv).statistic
    rho_e = spearmanr(de, dv).statistic
    diam = np.nanmax(hop[np.isfinite(hop)])
    # decay length: hops to drop |PTDF| by 10x (median-based binned slope)
    med = [
        np.median(dv[dh == d])
        for d in range(1, int(min(diam, 40)))
        if (dh == d).sum() > 50
    ]
    slope = np.polyfit(range(1, len(med) + 1), med, 1)[0] if len(med) > 3 else np.nan
    print(f"\n[Pilot B] {name}: diameter~{diam:.0f}, sources={len(srcs)}")
    print(f"  Spearman(log10|PTDF|, hop)  = {rho_h:.3f}")
    print(
        f"  Spearman(log10|PTDF|, Reff) = {rho_e:.3f}   (delta={abs(rho_e) - abs(rho_h):+.3f})",
    )
    print(
        f"  median log10|PTDF| slope per hop = {slope:.3f} (10x drop every {abs(1 / slope) if slope else float('nan'):.1f} hops)",
    )
    return rho_h, rho_e, slope


if __name__ == "__main__":
    cases = {
        "case1354_pegase": "../gridfm-datakit/gridfm_datakit/grids/pglib_opf_case1354_pegase_corrected.m",
        "Texas2k": "../gridfm-datakit/scripts/grids/Texas2k_case1_2016summerpeak.m",
    }
    for name, path in cases.items():
        bus, branch, gen = load_case(path)
        kron_fillin(name, bus, branch, gen, kv_quantile=0.25)
        ptdf_locality(name, bus, branch)
