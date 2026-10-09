"""Per-scenario gate on a large grid: does the oracle Newton (labelled REF voltage) reproduce the labels, and if not,
why: non-convergence of plain Newton, islanded buses, PV voltages that differ from their setpoints (Q-limit switching
in the data), or labels that are not solutions of our Ybus physics (P47-P50 block of VERDICTS.md, 2026-10-09).

python -m experiments.ac_pf_symmetries.gate_extrap case2000_goc"""
import math
import sys
from types import SimpleNamespace

case = sys.argv[1]
sys.argv = sys.argv[:1]  # r2_classical runs its loop over sys.argv at import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from torch_geometric.loader import DataLoader

from experiments.ac_pf_symmetries.dcprior import dc_angles
from experiments.ac_pf_symmetries.multigrid import grid_splits
from experiments.ac_pf_symmetries.r2_classical import D, fdpf, dc_matrix, network, newton
from experiments.ac_pf_symmetries.run import make_args
from experiments.ac_pf_symmetries.symmetries import E, pred_masks
from gridfm_graphkit.datasets.globals import VA_OUT, VM_H, VM_OUT
from gridfm_graphkit.tasks.pf_task import _build_bus_target

a = SimpleNamespace(epochs=1, batch_size=1, patience=1, layers=12, hidden=48)
args = make_args(a)
args.data.networks = [case]
h = grid_splits(case, args, 0, 1024)
rows = []
for k, data in enumerate(DataLoader(h["all"], batch_size=1)):
    n = data["bus"].x.size(0)
    target = _build_bus_target(data, n)[0].double().numpy()
    y, p_spec, q_spec, ref, pv, pq, vm0, va_in = network(data)
    x = data["bus"].x.double().numpy()
    # labels in our physics: mismatch of the labelled state
    v_lab = target[:, VM_OUT] * np.exp(1j * target[:, VA_OUT])
    s_lab = v_lab * np.conj(y @ v_lab)
    pvpq = pv | pq
    lab_mis = max(np.abs(s_lab.real[pvpq] - p_spec[pvpq]).max(), np.abs(s_lab.imag[pq] - q_spec[pq]).max())
    # islands: buses not connected to the REF bus through the bus-bus edges
    ei = data[E].edge_index.numpy()
    adj = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    _, comp = connected_components(adj, directed=False)
    islanded = int((comp != comp[np.flatnonzero(ref)[0]]).sum())
    pv_dev = float(np.abs(target[pv, VM_OUT] - x[pv, VM_H]).max()) if pv.any() else 0.0
    vm_or = vm0.copy()
    vm_or[ref] = target[ref, VM_OUT]
    th_dc = dc_angles(data).double().numpy()
    vm_c, va_c, it, mis = newton(y, p_spec, q_spec, pv, pq, vm_or, th_dc, 30)
    _, m = pred_masks(data)["VA"]
    m = m.numpy()
    e = np.remainder(va_c - target[:, VA_OUT] + math.pi, 2 * math.pi)[m] - math.pi
    vmf, thf = fdpf(y, p_spec, q_spec, pq, ref, vm0, th_dc, dc_matrix(data)[0])
    ef = np.remainder(thf - target[:, VA_OUT] + math.pi, 2 * math.pi)[m] - math.pi
    rows.append(dict(k=k, it=it, mis=mis, lab_mis=lab_mis, islanded=islanded, pv_dev=pv_dev,
                     va_rmse=math.sqrt((e**2).mean()) * D, vm_max=float(np.abs(vm_c - target[:, VM_OUT]).max()),
                     lab_va_max=float(np.abs(target[:, VA_OUT]).max() * D), fdpf_va=math.sqrt((ef**2).mean()) * D,
                     fdpf_vm=float(np.sqrt(((vmf - target[:, VM_OUT])[pq] ** 2).mean()))))
r = {k: np.array([row[k] for row in rows]) for k in rows[0]}
conv = r["mis"] < 1e-6
print(f"{case}: {len(rows)} scenarios; oracle Newton converged (mismatch < 1e-6) in {conv.sum()}, not in {(~conv).sum()} "
      f"(iterations at cap 30: {(r['it'] >= 30).sum()})")
print(f"labels' mismatch in our Ybus physics: max {r['lab_mis'].max():.2e}, median {np.median(r['lab_mis']):.2e}; "
      f"scenarios with label mismatch > 1e-4: {(r['lab_mis'] > 1e-4).sum()}")
print(f"islanded buses: scenarios with any {(r['islanded'] > 0).sum()}, max {r['islanded'].max()}; "
      f"PV voltage label vs setpoint: max deviation {r['pv_dev'].max():.2e}, scenarios > 1e-4: {(r['pv_dev'] > 1e-4).sum()}")
for name, mask in (("converged", conv), ("not converged", ~conv)):
    if mask.any():
        print(f"{name}: VA rmse of Newton vs labels median {np.median(r['va_rmse'][mask]):.3g} max {r['va_rmse'][mask].max():.3g} deg; "
              f"max |label VA| median {np.median(r['lab_va_max'][mask]):.1f} max {r['lab_va_max'][mask].max():.1f} deg; "
              f"islanded median {np.median(r['islanded'][mask]):.0f}; label mismatch median {np.median(r['lab_mis'][mask]):.2e}; "
              f"fdpf from inputs VA median {np.median(r['fdpf_va'][mask]):.3g} max {r['fdpf_va'][mask].max():.3g}, "
              f"VM_PQ median {np.median(r['fdpf_vm'][mask]):.2e}")
print("fdpf from inputs over all scenarios: VA rmse %.4g deg (converged-only %.4g), VM_PQ rmse %.3e" % (
    math.sqrt((r['fdpf_va'] ** 2).mean()), math.sqrt((r['fdpf_va'][conv] ** 2).mean()), math.sqrt((r['fdpf_vm'] ** 2).mean())))
print("worst 8 scenarios by Newton mismatch: k, it, mismatch, label mismatch, islanded, pv_dev, max|label VA|, fdpf VA")
for i in np.argsort(-r["mis"])[:8]:
    print("  ", r["k"][i], r["it"][i], f"{r['mis'][i]:.2e}", f"{r['lab_mis'][i]:.2e}", r["islanded"][i], f"{r['pv_dev'][i]:.2e}",
          f"{r['lab_va_max'][i]:.1f}", f"{r['fdpf_va'][i]:.3g}")
