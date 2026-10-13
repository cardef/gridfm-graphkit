"""Piano X, X0: the physics frontier on grids no model was trained on.

For every grid, on the fixed 1024 zero-shot scenarios of R2 (multigrid.grid_splits, seed 0), from the inputs only:
the DC start (dcprior.dc_angles, the campaign's DC), then the fast-decoupled iteration (FDPF: B' = the DC Laplacian,
B'' = -Im Ybus on the PQ buses, as r2_classical.fdpf) and damped Newton (as r2_classical.newton), each recorded after
every iteration k, with the REF voltage given (the well-posed task) and from the inputs (mean PV setpoint). Angle error
split as run.accuracy (offset relative to REF, rest, branch differences), VM on PQ buses. Wall-clock per scenario with
the same sparse code (scipy SuperLU, one thread), factorizations included; the DC cost is the sparse factorization and
solve of the same Laplacian. For scale, the forward time of the GENCO backbone (12 layers; hidden 12 / 24 / 48 =
Tiny / Small / Base, random weights: the cost does not depend on them) on the same graphs, batch 1 and batch 32.

python -m experiments.ac_pf_symmetries.frontier case14_ieee case30_ieee case57_ieee case118_ieee case500_goc case2000_goc
    [--n 1024] [--out results/pianox]   # one json per grid in --out, a text summary on stdout
X0b: --methods anderson2 anderson3 --tag anderson --no-backbone (Anderson acceleration of the fast-decoupled map)
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import spsolve
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from gridfm_graphkit.datasets.globals import VA_OUT, VM_OUT
from gridfm_graphkit.io.param_handler import get_task
from gridfm_graphkit.tasks.pf_task import _build_bus_target

from .dcprior import dc_angles
from .multigrid import grid_splits
from .run import make_args
from .symmetries import E, line_pairs, pred_masks

_argv, sys.argv = sys.argv, sys.argv[:1]  # r2_classical runs its CLI loop at import time
from .r2_classical import dc_matrix, factor, mismatch, network  # noqa: E402

sys.argv = _argv

OUT = Path(__file__).resolve().parent / "results" / "pianox"
D = 180 / math.pi
Q = ("VA", "offset", "rest", "dVA", "VM_PQ")


def fdpf_trace(y, p_spec, q_spec, pqi, free, vm0, th0, lu_p, lu_q, steps):
    """FDPF from (vm0, th0) with B' and B'' factorized once; state and wall-clock after every full iteration."""
    vm, th = vm0.copy(), th0.copy()
    states, times = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        v = vm * np.exp(1j * th)
        th[free] += lu_p.solve(((p_spec - (v * np.conj(y @ v)).real) / vm)[free])
        v = vm * np.exp(1j * th)
        vm[pqi] += lu_q.solve(((q_spec - (v * np.conj(y @ v)).imag) / vm)[pqi])
        times.append(time.perf_counter() - t0)
        states.append((vm.copy(), th.copy()))
    return states, times


def fdpf_map(y, p_spec, q_spec, pqi, free, vm, th, lu_p, lu_q):
    """One full fast-decoupled iteration (P-theta half then Q-V half), as in fdpf_trace."""
    vm, th = vm.copy(), th.copy()
    v = vm * np.exp(1j * th)
    th[free] += lu_p.solve(((p_spec - (v * np.conj(y @ v)).real) / vm)[free])
    v = vm * np.exp(1j * th)
    vm[pqi] += lu_q.solve(((q_spec - (v * np.conj(y @ v)).imag) / vm)[pqi])
    return vm, th


def anderson_trace(y, p_spec, q_spec, pqi, free, vm0, th0, lu_p, lu_q, steps, mem):
    """Anderson acceleration (type II, memory `mem`, full mixing) of the fast-decoupled map: its fixed points are the
    map's, i.e. the exact AC solution; each step costs one map evaluation plus a least squares of size `mem`. The
    memory restarts when the fixed-point residual grows. State and wall-clock after every step."""
    nf = len(free)

    def pack(vm, th):
        return np.r_[th[free], vm[pqi]]

    def unpack(x):
        vm, th = vm0.copy(), th0.copy()
        th[free], vm[pqi] = x[:nf], x[nf:]
        return vm, th

    x = pack(vm0, th0)
    gs, fs, prev = [], [], np.inf
    states, times = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        gx = pack(*fdpf_map(y, p_spec, q_spec, pqi, free, *unpack(x), lu_p, lu_q))
        f = gx - x
        nrm = np.linalg.norm(f)
        if nrm > prev:
            gs, fs = [], []
        prev = nrm
        gs.append(gx)
        fs.append(f)
        if len(fs) > mem + 1:
            gs.pop(0)
            fs.pop(0)
        if len(fs) == 1:
            x = gx
        else:
            d_f = np.stack([fs[i + 1] - fs[i] for i in range(len(fs) - 1)], axis=1)
            d_g = np.stack([gs[i + 1] - gs[i] for i in range(len(gs) - 1)], axis=1)
            x = gx - d_g @ np.linalg.lstsq(d_f, f, rcond=None)[0]
        times.append(time.perf_counter() - t0)
        states.append(unpack(x))
    return states, times


def newton_trace(y, p_spec, q_spec, pv, pq, vm0, va0, steps, tol=1e-9):
    """Damped polar Newton (r2_classical.newton with damped=True); state and wall-clock after every iteration."""
    vm, va = vm0.copy(), va0.copy()
    pvpq, pqi = np.r_[np.flatnonzero(pv), np.flatnonzero(pq)], np.flatnonzero(pq)
    states, times = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        v = vm * np.exp(1j * va)
        i = y @ v
        s = v * np.conj(i)
        f = np.r_[s.real[pvpq] - p_spec[pvpq], s.imag[pqi] - q_spec[pqi]]
        if np.abs(f).max() >= tol:
            dv, di, dvn = sp.diags(v), sp.diags(i), sp.diags(v / np.abs(v))
            dva = (1j * dv @ (di - y @ dv).conj()).tocsr()
            dvm = (dv @ (y @ dvn).conj() + di.conj() @ dvn).tocsr()
            jac = sp.bmat([
                [dva.real[pvpq][:, pvpq], dvm.real[pvpq][:, pqi]],
                [dva.imag[pqi][:, pvpq], dvm.imag[pqi][:, pqi]],
            ], format="csc")
            dx = -spsolve(jac, f)
            step, f0 = 1.0, np.abs(f).max()
            for _ in range(10):
                va_t, vm_t = va.copy(), vm.copy()
                va_t[pvpq] += step * dx[: len(pvpq)]
                vm_t[pqi] += step * dx[len(pvpq):]
                if np.abs(mismatch(y, p_spec, q_spec, pvpq, pqi, vm_t, va_t)).max() < f0:
                    break
                step *= 0.5
            va[pvpq] += step * dx[: len(pvpq)]
            vm[pqi] += step * dx[len(pvpq):]
        times.append(time.perf_counter() - t0)
        states.append((vm.copy(), va.copy()))
    return states, times


def errors(acc, key, vm, va, target, m, vmask, pq, f, t):
    """Accumulate squared errors for one scenario; return its VA rmse in deg."""
    e = np.remainder(va - target[:, VA_OUT] + math.pi, 2 * math.pi)[m] - math.pi
    n = m.sum()
    for q, sq in (("VA", (e**2).sum()), ("offset", n * e.mean() ** 2), ("rest", ((e - e.mean()) ** 2).sum())):
        acc[key, q][0] += float(sq)
        acc[key, q][1] += float(n)
    de = np.remainder((va[f] - va[t]) - (target[f, VA_OUT] - target[t, VA_OUT]) + math.pi, 2 * math.pi) - math.pi
    acc[key, "dVA"][0] += float((de**2).sum())
    acc[key, "dVA"][1] += float(de.size)
    w = vmask & pq
    acc[key, "VM_PQ"][0] += float(((vm - target[:, VM_OUT])[w] ** 2).sum())
    acc[key, "VM_PQ"][1] += float(w.sum())
    return math.sqrt((e**2).mean()) * D


def time_backbone(case, h, hidden, n_graphs=32):
    args = make_args(SimpleNamespace(epochs=1, batch_size=1, patience=1, layers=12, hidden=hidden))
    args.data.networks = [case]
    model = get_task(args, [h["normalizer"]]).model.eval()
    out = {"hidden": hidden, "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    sub = Subset(h["all"], range(min(n_graphs, len(h["all"]))))
    with torch.no_grad():
        for bs in (1, 32):
            batches = list(DataLoader(sub, batch_size=bs))
            model(batches[0])  # warm-up
            t0 = time.perf_counter()
            for b in batches:
                model(b)
            out[f"ms_per_graph_batch{bs}"] = (time.perf_counter() - t0) / len(sub) * 1e3
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cases", nargs="+")
    p.add_argument("--n", type=int, default=1024, help="first n of the fixed 1024 zero-shot scenarios")
    p.add_argument("--fdpf-steps", type=int, default=12)
    p.add_argument("--newton-steps", type=int, default=6)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--no-backbone", action="store_true")
    p.add_argument("--methods", nargs="+", default=["fdpf", "newton"], help="fdpf, newton, anderson<m>")
    p.add_argument("--tag", default="", help="suffix of the json files (frontier_<tag>_<case>.json)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    args = make_args(SimpleNamespace(epochs=1, batch_size=1, patience=1, layers=12, hidden=48))
    for case in a.cases:
        args.data.networks = [case]
        h = grid_splits(case, args, 0, 1024)
        sub = Subset(h["all"], range(min(a.n, len(h["all"]))))
        acc, worst, tim, unconv, n_bus = defaultdict(lambda: [0.0, 0.0]), defaultdict(float), defaultdict(list), defaultdict(int), None
        for data in DataLoader(sub, batch_size=1):
            n_bus = data["bus"].x.size(0)
            target = _build_bus_target(data, n_bus)[0].double().numpy()
            m = pred_masks(data)["VA"][1].numpy()
            vmask = pred_masks(data)["VM"][1].numpy()
            _, fwd, _ = line_pairs(data)
            f, t = (v.numpy() for v in data[E].edge_index[:, fwd])
            y, p_spec, q_spec, ref, pv, pq, vm_in, va_in = network(data)
            vm_given = vm_in.copy()
            vm_given[ref] = target[ref, VM_OUT]
            free, pqi = np.flatnonzero(~ref), np.flatnonzero(pq)
            pvpq = np.r_[np.flatnonzero(pv), pqi]
            lap = dc_matrix(data)[0]
            t0 = time.perf_counter()
            lu_p = factor(lap, free)
            t1 = time.perf_counter()
            lu_p.solve(p_spec[free])
            t2 = time.perf_counter()
            lu_q = factor(-y.imag, pqi)
            t3 = time.perf_counter()
            tim["dc factor"].append(t1 - t0)
            tim["dc solve"].append(t2 - t1)
            tim["B'' factor"].append(t3 - t2)
            th_dc = dc_angles(data).double().numpy()
            for vref, vm0 in (("given", vm_given), ("inputs", vm_in)):
                errors(acc, ("dc", vref, 0), vm0, th_dc, target, m, vmask, pq, f, t)
                for meth in a.methods:
                    if meth == "fdpf":
                        states, times = fdpf_trace(y, p_spec, q_spec, pqi, free, vm0, th_dc, lu_p, lu_q, a.fdpf_steps)
                    elif meth == "newton":
                        states, times = newton_trace(y, p_spec, q_spec, pv, pq, vm0, th_dc, a.newton_steps)
                    else:
                        states, times = anderson_trace(y, p_spec, q_spec, pqi, free, vm0, th_dc, lu_p, lu_q,
                                                       a.fdpf_steps, int(meth.removeprefix("anderson")))
                    tim[f"{meth} iteration"] += times
                    for k, (vm, va) in enumerate(states, 1):
                        r = errors(acc, (meth, vref, k), vm, va, target, m, vmask, pq, f, t)
                        worst[meth, vref, k] = max(worst[meth, vref, k], r)
                    if not np.abs(mismatch(y, p_spec, q_spec, pvpq, pqi, *states[-1])).max() < 1e-4:
                        unconv[meth, vref] += 1

        def val(meth, vref, k, q):
            s = acc[(meth, vref, k), q]
            return math.sqrt(s[0] / s[1]) * (1.0 if q == "VM_PQ" else D) if s[1] else float("nan")

        med = {k: float(np.median(v)) * 1e3 for k, v in tim.items()}
        dc_ms, bpp_ms = med["dc factor"] + med["dc solve"], med["B'' factor"]
        cost = {}
        for meth in a.methods:
            it = med[f"{meth} iteration"]
            cost.update({f"{meth} k={k}": (dc_ms + (0 if meth == "newton" else bpp_ms) + k * it) / dc_ms for k in (1, 2, 3, 5)})
        steps = {meth: a.newton_steps if meth == "newton" else a.fdpf_steps for meth in a.methods}
        res = {"case": case, "n": len(sub), "n_bus": n_bus, "ms_median": med, "cost_rel_dc": cost, "metrics": {},
               "worst_scenario_VA": {}, "unconverged_1e-4": {f"{k[0]} {k[1]}": v for k, v in unconv.items()}}
        for vref in ("given", "inputs"):
            res["metrics"][f"dc {vref} k=0"] = {q: val("dc", vref, 0, q) for q in Q}
            for meth, n_it in steps.items():
                for k in range(1, n_it + 1):
                    res["metrics"][f"{meth} {vref} k={k}"] = {q: val(meth, vref, k, q) for q in Q}
                    res["worst_scenario_VA"][f"{meth} {vref} k={k}"] = worst[meth, vref, k]
        print(f"{case} | {len(sub)} scenarios, {n_bus} buses | median ms per scenario: DC {dc_ms:.3g} (factor "
              f"{med['dc factor']:.3g}, solve {med['dc solve']:.3g}), B'' factor {bpp_ms:.3g}, "
              + ", ".join(f"{meth} iteration {med[meth + ' iteration']:.3g}" for meth in a.methods)
              + " | cost relative to DC: " + ", ".join(f"{k} {v:.2f}" for k, v in cost.items()), flush=True)
        for meth, n_it in steps.items():
            for vref in ("given", "inputs"):
                va = [val("dc", vref, 0, "VA")] + [val(meth, vref, k, "VA") for k in range(1, n_it + 1)]
                first = {th: next((k for k, v in enumerate(va) if v < th), None) for th in (0.1, 0.01)}
                print(f"  {meth:6s} {vref:6s} VA by k: " + " ".join(f"{k}:{v:.3g}" for k, v in enumerate(va))
                      + f" | first k below 0.1 deg: {first[0.1]}, below 0.01: {first[0.01]} | worst scenario at k=2: "
                      f"{worst[meth, vref, 2]:.3g} | unconverged after {n_it}: {unconv[meth, vref]}", flush=True)
                for k in (1, 2, 5):
                    if k <= n_it:
                        print(f"      k={k}: " + " ".join(f"{q} {val(meth, vref, k, q):.3g}" for q in Q), flush=True)
        if not a.no_backbone:
            res["backbone"] = [time_backbone(case, h, hd) for hd in (12, 24, 48)]
            print("  backbone (12 layers, random weights): " + "; ".join(
                f"hidden {b['hidden']} {b['params_M']:.2f} M params {b['ms_per_graph_batch1']:.3g} ms/graph batch 1, "
                f"{b['ms_per_graph_batch32']:.3g} batch 32" for b in res["backbone"]), flush=True)
        (a.out / f"frontier_{a.tag + '_' if a.tag else ''}{case}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
