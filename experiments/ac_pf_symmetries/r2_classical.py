"""Classical references for the R2 transfer study, on the zero-shot scenarios, from the inputs only.

- dc: the DC power flow of `r2_baselines.py` / `dcprior.py`.
- dcloss: DC with loss compensation. Three times: each branch's series loss g (2 - 2 cos(theta_f - theta_t)) at
  |V| = 1 (g = -Re Y_ft) is added as load, half at each end, together with the shunt conductance, and the DC system is
  solved again.
- nr1: one full AC Newton-Raphson step from the DC angles and the voltage setpoints (|V| = 1 at PQ buses).
- nr: Newton-Raphson to convergence from the same start and from a flat start, with step halving when a full step
  would increase the mismatch (2026-10-09; identical to the plain iteration whenever that converges); the iterations are reported.
- dcv, dcvloss (step 1b): DC with each susceptance weighted by V_f V_t (V from the setpoints), without and with the
  loss compensation (losses g |V_f e^{j theta_f} - V_t e^{j theta_t}|^2 at those voltages).
- picard: the P-theta fixed-point iteration theta <- theta + L^-1 (P_spec - P_ac(V, theta)), L the DC matrix, 12 steps
  from theta_DC with V fixed at the setpoints (the P37s reference); fdpf: the full fast-decoupled iteration (B' = L,
  B'' = -Im Ybus on the PQ buses), 12 steps from the same start.
Every row also reports the signed per-graph offset and VM on the PQ buses only; per grid, the in-sample R^2 of the
REF voltage on the PV setpoints (mean, min, max) and the total loads.

The PF mask of the framework hides the REF bus's voltage magnitude (and QG): with P and Q at REF unknown too, the
inputs leave the AC solution one degree of freedom short, essentially the REF voltage. Newton runs twice: with the
labelled REF voltage ("oracle", one input more than the learned models have; its converged error against the labels
checks that this solver reproduces the data's physics, it must be ~0) and with the mean of the graph's PV voltage
setpoints ("inputs", the same information as the models).

Same masks and split of the angle error as run.accuracy (offset relative to the REF bus, rest, branch differences).

python -m experiments.ac_pf_symmetries.r2_classical case14_ieee case30_ieee case57_ieee case118_ieee
"""

import math
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu, spsolve
from torch_geometric.loader import DataLoader

from gridfm_graphkit.datasets.globals import (
    BS,
    GS,
    PD_H,
    PG_H,
    QD_H,
    VA_H,
    VA_OUT,
    VM_H,
    VM_OUT,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)
from gridfm_graphkit.tasks.pf_task import _build_bus_target

from .dcprior import dc_angles
from .hodge import solve_anchored
from .multigrid import grid_splits
from .run import make_args
from .symmetries import E, line_pairs, pred_masks

D = 180 / math.pi


def network(data):
    """Bus admittance matrix (sparse CSR, complex), specified injections and bus sets of one graph (float64 numpy).
    Sparse since the GOC grids (2026-10-09): the dense Jacobian products and solves cost O(n^3) per Newton step, hours
    per 1024 scenarios of case2000_goc; the numbers are the same (duplicates are summed as np.add.at did)."""
    x = data["bus"].x.double().numpy()
    ei = data[E].edge_index.numpy()
    ea = data[E].edge_attr.double().numpy()
    n = x.shape[0]
    rows = np.r_[ei[0], ei[0], np.arange(n)]  # each directed row: its own end, the other end, then the bus shunt
    cols = np.r_[ei[0], ei[1], np.arange(n)]
    vals = np.r_[ea[:, YFF_TT_R] + 1j * ea[:, YFF_TT_I], ea[:, YFT_TF_R] + 1j * ea[:, YFT_TF_I], x[:, GS] + 1j * x[:, BS]]
    y = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    gi, gb = (v.numpy() for v in data.edge_index_dict[("gen", "connected_to", "bus")])
    known = ~data.mask_dict["gen"][:, PG_H].numpy()[gi]
    pg = np.zeros(n)
    np.add.at(pg, gb, data["gen"].x.double().numpy()[gi, PG_H] * known)
    ref, pv = data.mask_dict["REF"].numpy(), data.mask_dict["PV"].numpy()
    vm0 = np.where(pv, x[:, VM_H], 1.0)  # the REF voltage is masked in the input: set by the caller
    vm0[ref] = x[pv, VM_H].mean() if pv.any() else 1.0
    return y, pg - x[:, PD_H], -x[:, QD_H], ref, pv, ~(ref | pv), vm0, x[:, VA_H]


def mismatch(y, p_spec, q_spec, pvpq, pqi, vm, va):
    v = vm * np.exp(1j * va)
    s = v * np.conj(y @ v)
    return np.r_[s.real[pvpq] - p_spec[pvpq], s.imag[pqi] - q_spec[pqi]]


def newton(y, p_spec, q_spec, pv, pq, vm, va, max_it, tol=1e-9, damped=False):
    """Polar Newton-Raphson (MATPOWER's Jacobian); returns |V|, angles, iterations and final max mismatch.
    damped=True halves the step (up to 10 times) until the max mismatch decreases: on case2000_goc the plain
    iteration diverges from the DC and the flat start on 14 of 1024 scenarios (gate of 2026-10-09); a full step that
    already decreases the mismatch is taken unchanged, so converging runs are identical to the undamped ones."""
    vm, va = vm.copy(), va.copy()
    pvpq, pqi = np.r_[np.flatnonzero(pv), np.flatnonzero(pq)], np.flatnonzero(pq)
    for it in range(max_it + 1):
        v = vm * np.exp(1j * va)
        i = y @ v
        s = v * np.conj(i)
        f = np.r_[s.real[pvpq] - p_spec[pvpq], s.imag[pqi] - q_spec[pqi]]
        if np.abs(f).max() < tol or it == max_it:
            return vm, va, it, np.abs(f).max()
        dv, di, dvn = sp.diags(v), sp.diags(i), sp.diags(v / np.abs(v))
        dva = (1j * dv @ (di - y @ dv).conj()).tocsr()
        dvm = (dv @ (y @ dvn).conj() + di.conj() @ dvn).tocsr()
        jac = sp.bmat([
            [dva.real[pvpq][:, pvpq], dvm.real[pvpq][:, pqi]],
            [dva.imag[pqi][:, pvpq], dvm.imag[pqi][:, pqi]],
        ], format="csc")
        dx = -spsolve(jac, f)
        step = 1.0
        if damped:
            f0 = np.abs(f).max()
            for _ in range(10):
                va_t, vm_t = va.copy(), vm.copy()
                va_t[pvpq] += step * dx[: len(pvpq)]
                vm_t[pqi] += step * dx[len(pvpq):]
                if np.abs(mismatch(y, p_spec, q_spec, pvpq, pqi, vm_t, va_t)).max() < f0:
                    break
                step *= 0.5
        va[pvpq] += step * dx[: len(pvpq)]
        vm[pqi] += step * dx[len(pvpq):]


def dc_with_losses(data, iters=3):
    _, fwd, _ = line_pairs(data)
    ei, ea = data[E].edge_index, data[E].edge_attr.double()
    f, t, w, g = ei[0, fwd], ei[1, fwd], ea[fwd, YFT_TF_I], -ea[fwd, YFT_TF_R]
    gi, gb = data.edge_index_dict[("gen", "connected_to", "bus")]
    known = (~data.mask_dict["gen"][gi, PG_H]).double()
    x = data["bus"].x.double()
    p0 = torch.zeros(x.size(0), dtype=torch.float64).index_add(0, gb, data["gen"].x[gi, PG_H].double() * known)
    p0 = p0 - x[:, PD_H] - x[:, GS]
    ref, batch = data.mask_dict["REF"], torch.zeros(x.size(0), dtype=torch.long)
    theta_ref = x[ref, VA_H]
    th = dc_angles(data).double()
    for _ in range(iters):
        loss = g * (2 - 2 * torch.cos(th[f] - th[t]))
        p = p0 - 0.5 * torch.zeros_like(p0).index_add(0, f, loss) - 0.5 * torch.zeros_like(p0).index_add(0, t, loss)
        th = solve_anchored(p, f, t, w, ref, theta_ref, batch, 1)
    return th.numpy()


def dc_matrix(data):
    """The DC Laplacian (weights Im Y_ft on the forward rows) as a sparse CSR matrix, with its branches."""
    _, fwd, _ = line_pairs(data)
    ei, ea = data[E].edge_index.numpy(), data[E].edge_attr.double().numpy()
    f, t, w = ei[0, fwd.numpy()], ei[1, fwd.numpy()], ea[fwd.numpy(), YFT_TF_I]
    n = data["bus"].x.size(0)
    lap = laplacian(f, t, w, n)
    return lap, f, t, w, -ea[fwd.numpy(), YFT_TF_R]


def laplacian(f, t, w, n):
    return sp.csr_matrix((np.r_[w, w, -w, -w], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(n, n))


def factor(mat, idx):
    """LU of the rows and columns `idx` of a sparse matrix (one factorization, reused across the iteration steps)."""
    return splu(mat[idx][:, idx].tocsc())


def dc_weighted(data, vm, losses, iters=3):
    """DC with susceptances weighted by V_f V_t, optionally with loss compensation at those voltages."""
    lap, f, t, w, g = dc_matrix(data)
    y, p_spec, _, ref, _, _, _, va_in = network(data)
    x = data["bus"].x.double().numpy()
    wv = w * vm[f] * vm[t]
    lv = laplacian(f, t, wv, lap.shape[0])
    free = np.flatnonzero(~ref)
    lu = factor(lv, free)
    p0 = p_spec - x[:, GS] * vm**2
    th = np.full(len(vm), va_in[ref][0])
    for _ in range(1 + (iters if losses else 0)):
        loss = g * (vm[f] ** 2 + vm[t] ** 2 - 2 * vm[f] * vm[t] * np.cos(th[f] - th[t])) if losses else 0 * g
        p = p0.copy()
        np.add.at(p, f, -0.5 * loss)
        np.add.at(p, t, -0.5 * loss)
        th = np.full(len(vm), va_in[ref][0])
        th[free] += lu.solve(p[free])
    return th


def picard(y, p_spec, ref, vm, th0, lap, steps=12):
    th, free = th0.copy(), np.flatnonzero(~ref)
    lu = factor(lap, free)
    for _ in range(steps):
        v = vm * np.exp(1j * th)
        th[free] += lu.solve((p_spec - (v * np.conj(y @ v)).real)[free])
    return th


def fdpf(y, p_spec, q_spec, pq, ref, vm0, th0, lap, steps=12):
    vm, th, free = vm0.copy(), th0.copy(), np.flatnonzero(~ref)
    pqi = np.flatnonzero(pq)
    lu, bpp = factor(lap, free), factor(-y.imag, pqi)
    for _ in range(steps):
        v = vm * np.exp(1j * th)
        th[free] += lu.solve(((p_spec - (v * np.conj(y @ v)).real) / vm)[free])
        v = vm * np.exp(1j * th)
        vm[pqi] += bpp.solve(((q_spec - (v * np.conj(y @ v)).imag) / vm)[pqi])
    return vm, th


def add(s, k, sq, c):
    s.setdefault(k, [0.0, 0.0])
    s[k][0] += float(sq)
    s[k][1] += float(c)


a = SimpleNamespace(epochs=1, batch_size=1, patience=1, layers=12, hidden=48)
for case in sys.argv[1:]:
    args = make_args(a)
    args.data.networks = [case]
    h = grid_splits(case, args, 0, 1024)
    s, iters, vref, feats = {}, {"dc start": [], "flat start": []}, [], []
    for data in DataLoader(h["all"], batch_size=1):
        target, _, _ = _build_bus_target(data, data["bus"].x.size(0))
        target = target.double().numpy()
        _, m = pred_masks(data)["VA"]
        m = m.numpy()
        _, vmask = pred_masks(data)["VM"]
        vmask = vmask.numpy()
        _, fwd, _ = line_pairs(data)
        f, t = (v.numpy() for v in data[E].edge_index[:, fwd])
        y, p_spec, q_spec, ref, pv, pq, vm0, va_in = network(data)
        vm_oracle = vm0.copy()
        vm_oracle[ref] = target[ref, VM_OUT]
        vref.append((target[ref, VM_OUT][0], vm0[ref][0]))
        xb = data["bus"].x.double().numpy()
        setp = xb[pv, VM_H]
        feats.append([1.0, setp.mean(), setp.min(), setp.max(), xb[:, PD_H].sum(), xb[:, QD_H].sum()])
        th_dc = dc_angles(data).double().numpy()
        lap = dc_matrix(data)[0]
        states = {"dc": (vm0, th_dc), "dcloss": (vm0, dc_with_losses(data))}
        states["dcv"] = (vm0, dc_weighted(data, vm0, losses=False))
        states["dcvloss"] = (vm0, dc_weighted(data, vm0, losses=True))
        states["picard"] = (vm0, picard(y, p_spec, ref, vm0, th_dc, lap))
        states["picard oracle"] = (vm_oracle, picard(y, p_spec, ref, vm_oracle, th_dc, lap))
        states["fdpf"] = fdpf(y, p_spec, q_spec, pq, ref, vm0, th_dc, lap)
        for tag, v0 in (("oracle", vm_oracle), ("inputs", vm0)):
            states[f"nr1 {tag}"] = newton(y, p_spec, q_spec, pv, pq, v0, th_dc, 1)[:2]
            vm_c, va_c, it_dc, _ = newton(y, p_spec, q_spec, pv, pq, v0, th_dc, 30, damped=True)
            states[f"nr {tag}"] = (vm_c, va_c)
            if tag == "oracle":
                iters["dc start"].append(it_dc)
                flat = np.where(ref, va_in, va_in[ref][0])
                iters["flat start"].append(newton(y, p_spec, q_spec, pv, pq, v0, flat, 30, damped=True)[2])
        for k, (vm, va) in states.items():
            e = np.remainder(va - target[:, VA_OUT] + math.pi, 2 * math.pi)[m] - math.pi
            add(s, f"{k} VA", (e**2).sum(), m.sum())
            add(s, f"{k} offset", m.sum() * e.mean() ** 2, m.sum())
            add(s, f"{k} signed", m.sum() * e.mean(), m.sum())
            add(s, f"{k} rest", ((e - e.mean()) ** 2).sum(), m.sum())
            de = np.remainder((va[f] - va[t]) - (target[f, VA_OUT] - target[t, VA_OUT]) + math.pi, 2 * math.pi) - math.pi
            add(s, f"{k} dVA", (de**2).sum(), de.size)
            add(s, f"{k} VM", ((vm - target[:, VM_OUT])[vmask] ** 2).sum(), vmask.sum())
            add(s, f"{k} VM_PQ", ((vm - target[:, VM_OUT])[vmask & pq] ** 2).sum(), (vmask & pq).sum())
    out = {k: (v[0] / v[1] if k.endswith("signed") else math.sqrt(v[0] / v[1])) * (1 if "VM" in k else D)
           for k, v in s.items()}
    xr, yr = np.array(feats), np.array(vref)[:, 0]
    coef = np.linalg.lstsq(xr, yr, rcond=None)[0]
    res, tot = ((yr - xr @ coef) ** 2).sum(), ((yr - yr.mean()) ** 2).sum()
    r2 = "n/a (constant)" if tot < 1e-12 else f"{1 - res / tot:.3f}"
    vr = np.array(vref)
    print(case, "| oracle Newton iterations to 1e-9 from DC start mean %.2f max %d, from flat start mean %.2f max %d" % (
        np.mean(iters["dc start"]), max(iters["dc start"]), np.mean(iters["flat start"]), max(iters["flat start"])),
        "| REF voltage label min %.4f max %.4f, its input estimate RMS error %.4f" % (
        vr[:, 0].min(), vr[:, 0].max(), np.sqrt(((vr[:, 0] - vr[:, 1]) ** 2).mean())),
        f"| REF voltage on PV setpoints and loads: R^2 {r2}, RMSE {math.sqrt(res / len(yr)):.4f}")
    for k in ("dc", "dcloss", "dcv", "dcvloss", "picard", "picard oracle", "fdpf",
              "nr1 oracle", "nr oracle", "nr1 inputs", "nr inputs"):
        print(f"  {k:13s} VA {out[k + ' VA']:.4g} | offset {out[k + ' offset']:.4g} (signed {out[k + ' signed']:+.4g}) "
              f"rest {out[k + ' rest']:.4g} dVA {out[k + ' dVA']:.4g} | VM {out[k + ' VM']:.4g} "
              f"VM_PQ {out[k + ' VM_PQ']:.4g}", flush=True)
