"""Piano S, S0 (PREREGISTRATION.md, S0-P1-S0-P3): physics-only state estimation on the unseen grids, no learning.

Per scenario, measurements are drawn from the labelled AC state as graphkit's `SimulateMeasurements` does: bus voltage
magnitude, bus net injection (P and Q together), and the flow at the sending end of every directed branch row (P and
Q together), each present with probability equal to the coverage, with Gaussian noise of `--noise` (relative) times
the true value. Unmeasured injections get a pseudo-measurement: the bus's nominal injection, the mean over the
scenarios of the fixed subset that are not evaluated (a stand-in for the planning base case), with standard deviation
30% of it plus a floor; buses whose injection is zero in every one of those scenarios enter as constraints.

AC WLS: polar state (angles of the non-REF buses, the REF angle known; magnitudes of all buses), Gauss-Newton on the
sparse gain matrix H^T W H, flat start, step halving while the weighted residual grows, at most 30 iterations, stop
when the largest step is below 1e-8. DC WLS (exploratory): angles only, P injections and P flows through the DC
matrix, one linear solve. Errors as in run.accuracy: VA on the non-REF buses with the offset / rest / branch-difference
split, VM on all buses.

python -m experiments.ac_pf_symmetries.s0_wls --out results/pianos/s0 case14_ieee ...
"""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from torch_geometric.loader import DataLoader

from gridfm_graphkit.datasets.globals import BS, GS, VA_OUT, VM_OUT, YFF_TT_I, YFF_TT_R, YFT_TF_I, YFT_TF_R
from gridfm_graphkit.tasks.pf_task import _build_bus_target

from .multigrid import grid_splits
from .run import make_args
from .symmetries import E, line_pairs

D = 180 / math.pi
FLOOR = 1e-5  # smallest standard deviation (exact measurements, constraints)


def grid_model(data):
    """Bus admittance matrix, branch-row matrices (from-end admittances and incidence) and REF mask, float64."""
    x = data["bus"].x.double().numpy()
    ei = data[E].edge_index.numpy()
    ea = data[E].edge_attr.double().numpy()
    n, m = x.shape[0], ei.shape[1]
    rows = np.r_[ei[0], ei[0], np.arange(n)]
    cols = np.r_[ei[0], ei[1], np.arange(n)]
    vals = np.r_[ea[:, YFF_TT_R] + 1j * ea[:, YFF_TT_I], ea[:, YFT_TF_R] + 1j * ea[:, YFT_TF_I], x[:, GS] + 1j * x[:, BS]]
    ybus = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    r = np.arange(m)
    yf = sp.csr_matrix((np.r_[ea[:, YFF_TT_R] + 1j * ea[:, YFF_TT_I], ea[:, YFT_TF_R] + 1j * ea[:, YFT_TF_I]],
                        (np.r_[r, r], np.r_[ei[0], ei[1]])), shape=(m, n))
    cf = sp.csr_matrix((np.ones(m), (r, ei[0])), shape=(m, n))
    w = ea[:, YFT_TF_I]  # DC weight of each directed row, as dcprior
    return ybus, yf, cf, ei, w, data.mask_dict["REF"].numpy()


def injections(ybus, v):
    return v * np.conj(ybus @ v)


def flows(yf, cf, v):
    return (cf @ v) * np.conj(yf @ v)


def jacobians(ybus, yf, cf, v):
    """dS/dVa and dS/dVm for the bus injections and the branch-row flows (MATPOWER's dSbus_dV and dSbr_dV)."""
    vn = v / np.abs(v)
    i = ybus @ v
    dv, di, dvn = sp.diags(v), sp.diags(i), sp.diags(vn)
    s_va = 1j * dv @ (di - ybus @ dv).conj()
    s_vm = dv @ (ybus @ dvn).conj() + di.conj() @ dvn
    i_f, v_f = yf @ v, cf @ v
    f_va = 1j * (sp.diags(i_f.conj()) @ cf @ dv - sp.diags(v_f) @ (yf @ dv).conj())
    f_vm = sp.diags(v_f) @ (yf @ dvn).conj() + sp.diags(i_f.conj()) @ cf @ dvn
    return s_va.tocsr(), s_vm.tocsr(), f_va.tocsr(), f_vm.tocsr()


def measure(v, ybus, yf, cf, coverage, noise, rng):
    """Measured sets (boolean) and noisy values; P and Q of an injection or a flow are measured together."""
    n, m = len(v), yf.shape[0]
    s, sf = injections(ybus, v), flows(yf, cf, v)
    mv, ms, mf = (rng.random(k) < coverage for k in (n, n, m))
    z = {"vm": np.abs(v) * (1 + noise * rng.standard_normal(n)),
         "p": s.real + noise * np.abs(s.real) * rng.standard_normal(n),
         "q": s.imag + noise * np.abs(s.imag) * rng.standard_normal(n),
         "pf": sf.real + noise * np.abs(sf.real) * rng.standard_normal(m),
         "qf": sf.imag + noise * np.abs(sf.imag) * rng.standard_normal(m)}
    sd = {"vm": np.maximum(noise * np.abs(v), FLOOR), "p": np.maximum(noise * np.abs(s.real), FLOOR),
          "q": np.maximum(noise * np.abs(s.imag), FLOOR), "pf": np.maximum(noise * np.abs(sf.real), FLOOR),
          "qf": np.maximum(noise * np.abs(sf.imag), FLOOR)}
    return mv, ms, mf, z, sd


def assemble(ms, z, sd, prior):
    """Injection rows: measured value where measured, else the nominal pseudo-measurement; zero-injection buses
    always enter as constraints."""
    p = np.where(ms, z["p"], prior["p"])
    q = np.where(ms, z["q"], prior["q"])
    sp_ = np.where(ms, sd["p"], prior["sp"])
    sq_ = np.where(ms, sd["q"], prior["sq"])
    zero = prior["zero"]
    return (np.where(zero, 0.0, p), np.where(zero, 0.0, q), np.where(zero, FLOOR, sp_), np.where(zero, FLOOR, sq_))


def ac_wls(model, mv, ms, mf, z, sd, prior, theta_ref, max_it=30, tol=1e-8, start=None, lm=False):
    """start: (va, vm) initial state (default flat); lm=True: Levenberg-Marquardt damping (lambda on the diagonal of
    the gain, x0.1 after an accepted step, x10 and retry after a rejected one) instead of step halving."""
    ybus, yf, cf, _, _, ref = model
    n = ybus.shape[0]
    free = np.flatnonzero(~ref)
    p, q, s_p, s_q = assemble(ms, z, sd, prior)
    iv, ifl = np.flatnonzero(mv), np.flatnonzero(mf)
    zz = np.r_[z["vm"][iv], p, q, z["pf"][ifl], z["qf"][ifl]]
    w = 1.0 / np.r_[sd["vm"][iv], s_p, s_q, sd["pf"][ifl], sd["qf"][ifl]] ** 2
    va, vm = (np.full(n, theta_ref), np.ones(n)) if start is None else (start[0].copy(), start[1].copy())
    lam = 1e-3

    def residual(va, vm):
        v = vm * np.exp(1j * va)
        s, sf = injections(ybus, v), flows(yf, cf, v)
        return zz - np.r_[vm[iv], s.real, s.imag, sf.real[ifl], sf.imag[ifl]], v

    r, v = residual(va, vm)
    cost = float((w * r * r).sum())
    for it in range(1, max_it + 1):
        s_va, s_vm, f_va, f_vm = jacobians(ybus, yf, cf, v)
        h_vm_meas = sp.csr_matrix((np.ones(len(iv)), (np.arange(len(iv)), iv)), shape=(len(iv), n))
        h = sp.bmat([
            [sp.csr_matrix((len(iv), len(free))), h_vm_meas],
            [s_va.real[:, free], s_vm.real],
            [s_va.imag[:, free], s_vm.imag],
            [f_va.real[ifl][:, free], f_vm.real[ifl]],
            [f_va.imag[ifl][:, free], f_vm.imag[ifl]],
        ], format="csr")
        hw = h.T @ sp.diags(w)
        gain, rhs = (hw @ h).tocsc(), hw @ r
        step = 1.0
        if lm:
            dg = sp.diags(gain.diagonal())
            for _ in range(12):
                dx = splu((gain + lam * dg).tocsc()).solve(rhs)
                va_t, vm_t = va.copy(), vm.copy()
                va_t[free] += dx[: len(free)]
                vm_t += dx[len(free):]
                r_t, v_t = residual(va_t, vm_t)
                cost_t = float((w * r_t * r_t).sum())
                if cost_t <= cost:
                    lam = max(lam * 0.1, 1e-12)
                    break
                lam *= 10
        else:
            dx = splu(gain).solve(rhs)
            for _ in range(12):  # step halving while the weighted residual grows
                va_t, vm_t = va.copy(), vm.copy()
                va_t[free] += step * dx[: len(free)]
                vm_t += step * dx[len(free):]
                r_t, v_t = residual(va_t, vm_t)
                cost_t = float((w * r_t * r_t).sum())
                if cost_t <= cost or step < 1e-3:
                    break
                step *= 0.5
        va, vm, r, v, cost = va_t, vm_t, r_t, v_t, cost_t
        if np.abs(step * dx).max() < tol:
            return va, vm, it, True
    return va, vm, max_it, False


def dc_wls(model, ms, mf, z, sd, prior, theta_ref):
    """Angles from the P injections (measured or nominal, zero-injection constraints) and the measured P flows through
    the DC matrix; one linear solve."""
    ybus, yf, cf, ei, w, ref = model
    n, m = ybus.shape[0], len(w)
    free = np.flatnonzero(~ref)
    p, _, s_p, _ = assemble(ms, z, sd, prior)
    r = np.arange(m)
    a_flow = sp.csr_matrix((np.r_[w, -w], (np.r_[r, r], np.r_[ei[0], ei[1]])), shape=(m, n))  # P_row = w (th_f - th_t)
    lap = (a_flow.T @ sp.csr_matrix((np.ones(m), (r, ei[0])), shape=(m, n))).T  # sum of the rows leaving each bus
    lap = sp.csr_matrix(lap)
    ifl = np.flatnonzero(mf)
    h = sp.vstack([lap, a_flow[ifl]]).tocsr()
    zz = np.r_[p, z["pf"][ifl]]
    wt = 1.0 / np.r_[s_p, sd["pf"][ifl]] ** 2
    zz = zz - h[:, np.flatnonzero(ref)] @ np.full(int(ref.sum()), theta_ref)
    hf = h[:, free]
    hw = hf.T @ sp.diags(wt)
    th = np.full(n, theta_ref)
    th[free] = splu((hw @ hf).tocsc()).solve(hw @ zz)
    return th


def errors(va, vm, target, free, f, t):
    e = np.remainder(va - target[:, VA_OUT] + math.pi, 2 * math.pi)[free] - math.pi
    de = np.remainder((va[f] - va[t]) - (target[f, VA_OUT] - target[t, VA_OUT]) + math.pi, 2 * math.pi) - math.pi
    out = {"VA": (e**2).sum(), "offset": len(e) * e.mean() ** 2, "rest": ((e - e.mean()) ** 2).sum(), "n": len(e),
           "dVA": (de**2).sum(), "n_d": len(de)}
    if vm is not None:
        out["VM"], out["n_v"] = ((vm - target[:, VM_OUT]) ** 2).sum(), len(vm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grids", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--coverage", nargs="+", type=float, default=[1.0, 0.8, 0.6, 0.4, 0.2])
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--prior-sd", type=float, default=0.3)
    ap.add_argument("--start", choices=["flat", "dc", "truth"], default="flat", help="AC WLS start: flat, the DC WLS "
                    "angles with |V| = 1 and the measured magnitudes where measured, or the labelled state (S0b oracle)")
    ap.add_argument("--lm", action="store_true", help="Levenberg-Marquardt damping instead of step halving")
    ap.add_argument("--max-it", type=int, default=30)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    args = make_args(SimpleNamespace(epochs=1, batch_size=1, patience=1, layers=12, hidden=48))
    for case in a.grids:
        args.data.networks = [case]
        data_list = list(DataLoader(grid_splits(case, args, 0, 1024)["all"], batch_size=1))
        evals, hist = data_list[: a.n_eval], data_list[a.n_eval:]
        # nominal injections and zero-injection buses from the scenarios not evaluated
        sh = []
        for d in hist:
            tg, _, _ = _build_bus_target(d, d["bus"].x.size(0))
            tg = tg.double().numpy()
            v = tg[:, VM_OUT] * np.exp(1j * tg[:, VA_OUT])
            sh.append(injections(grid_model(d)[0], v))
        sh = np.array(sh)
        nom = sh.mean(0)
        scale = np.abs(nom).mean()
        prior = {"p": nom.real, "q": nom.imag, "sp": a.prior_sd * np.abs(nom.real) + 0.01 * scale,
                 "sq": a.prior_sd * np.abs(nom.imag) + 0.01 * scale, "zero": np.abs(sh).max(0) < 1e-9}
        res = {"case": case, "n_eval": len(evals), "n_hist": len(hist), "noise": a.noise, "prior_sd": a.prior_sd,
               "start": a.start, "lm": a.lm, "max_it": a.max_it,
               "zero_injection_buses": int(prior["zero"].sum()), "n_bus": int(sh.shape[1]), "gate": {}, "runs": {}}
        t0 = time.time()
        # S0-P1 gate: zero noise, full coverage
        gmax_va = gmax_vm = 0.0
        for k, d in enumerate(evals[:32]):
            tg, _, _ = _build_bus_target(d, d["bus"].x.size(0))
            tg = tg.double().numpy()
            model = grid_model(d)
            ref = model[5]
            v = tg[:, VM_OUT] * np.exp(1j * tg[:, VA_OUT])
            mv, ms, mf, z, sd = measure(v, model[0], model[1], model[2], 1.0, 0.0, np.random.default_rng(k))
            th_ref = tg[ref, VA_OUT][0]
            start = None if a.start == "flat" else (dc_wls(model, ms, mf, z, sd, prior, th_ref), np.where(mv, z["vm"], 1.0))
            va, vm, _, _ = ac_wls(model, mv, ms, mf, z, sd, prior, th_ref, max_it=a.max_it, start=start, lm=a.lm)
            e = np.abs(np.remainder(va - tg[:, VA_OUT] + math.pi, 2 * math.pi) - math.pi)[~ref]
            gmax_va, gmax_vm = max(gmax_va, float(e.max()) * D), max(gmax_vm, float(np.abs(vm - tg[:, VM_OUT]).max()))
        res["gate"] = {"max_VA_deg": gmax_va, "max_VM": gmax_vm, "scenarios": min(32, len(evals))}
        for c in a.coverage:
            acc = {"ac": {}, "dc": {}}
            its, fails, per = [], 0, []
            for k, d in enumerate(evals):
                tg, _, _ = _build_bus_target(d, d["bus"].x.size(0))
                tg = tg.double().numpy()
                model = grid_model(d)
                ref = model[5]
                _, fwd, _ = line_pairs(d)
                f, t = (x.numpy() for x in d[E].edge_index[:, fwd])
                v = tg[:, VM_OUT] * np.exp(1j * tg[:, VA_OUT])
                rng = np.random.default_rng([k, int(round(c * 1000))])
                mv, ms, mf, z, sd = measure(v, model[0], model[1], model[2], c, a.noise, rng)
                th_ref = tg[ref, VA_OUT][0]
                th_dc = dc_wls(model, ms, mf, z, sd, prior, th_ref)
                start = {"flat": None, "dc": (th_dc, np.where(mv, z["vm"], 1.0)),
                         "truth": (tg[:, VA_OUT].copy(), tg[:, VM_OUT].copy())}[a.start]
                va, vm, it, ok = ac_wls(model, mv, ms, mf, z, sd, prior, th_ref, max_it=a.max_it, start=start, lm=a.lm)
                its.append(it)
                fails += not ok
                e_s = np.remainder(va - tg[:, VA_OUT] + math.pi, 2 * math.pi)[~ref] - math.pi
                per.append(float(np.sqrt((e_s**2).mean())) * D)
                for name, (va_, vm_) in (("ac", (va, vm)), ("dc", (th_dc, None))):
                    for key, val in errors(va_, vm_, tg, ~ref, f, t).items():
                        acc[name][key] = acc[name].get(key, 0.0) + val
            row = {}
            for name, s in acc.items():
                row[name] = {"VA": math.sqrt(s["VA"] / s["n"]) * D, "offset": math.sqrt(s["offset"] / s["n"]) * D,
                             "rest": math.sqrt(s["rest"] / s["n"]) * D, "dVA": math.sqrt(s["dVA"] / s["n_d"]) * D}
                if "VM" in s:
                    row[name]["VM"] = math.sqrt(s["VM"] / s["n_v"])
            row["ac"].update(iterations_mean=float(np.mean(its)), iterations_max=int(max(its)), not_converged=fails,
                             VA_scenario_median=float(np.median(per)), VA_scenario_p95=float(np.percentile(per, 95)),
                             VA_scenario_max=float(max(per)))
            res["runs"][f"{c:.2f}"] = row
            print(case, f"c={c:.2f}", json.dumps(row), flush=True)
        res["time_s"] = time.time() - t0
        (a.out / f"s0_{case}.json").write_text(json.dumps(res, indent=1))
        print(case, "gate", json.dumps(res["gate"]), f"{res['time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
