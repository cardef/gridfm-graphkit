"""Trivial references for the R2 angle errors, on the scenarios of the zero-shot evaluation.

flat: every predicted angle = theta_ref (= 0 in these data), every predicted VM = 1.
DC: B theta = P on the non-REF buses, B_ft = Im(Y_ft) from the forward edge rows, P = Pg (inputs) - Pd,
theta_ref fixed: one linear solve per graph from inputs only (S1/S3/S4-equivariant by construction).
Same scenarios and masks as the zero-shot evaluation of multigrid.py (the fixed 1024-scenario subset).

python -m experiments.ac_pf_symmetries.r2_baselines case14_ieee case30_ieee case57_ieee case118_ieee
"""

import math
import sys
from types import SimpleNamespace

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from experiments.ac_pf_symmetries.multigrid import grid_splits
from experiments.ac_pf_symmetries.run import make_args
from experiments.ac_pf_symmetries.symmetries import E, line_pairs, pred_masks, wrap_angle
from gridfm_graphkit.datasets.globals import PD_H, PG_H, YFT_TF_I
from gridfm_graphkit.tasks.pf_task import _build_bus_target

D = 180 / math.pi
a = SimpleNamespace(epochs=1, batch_size=1, patience=1, layers=12, hidden=48)


def add(s, k, sq, c):
    s.setdefault(k, [0.0, 0.0])
    s[k][0] += float(sq)
    s[k][1] += float(c)


for case in sys.argv[1:]:
    args = make_args(a)
    args.data.networks = [case]
    h = grid_splits(case, args, 0, 1024)
    s, vmax = {}, 0.0
    for data in DataLoader(h["all"], batch_size=1):
        n = data["bus"].x.size(0)
        target, _, _ = _build_bus_target(data, n)
        col, m = pred_masks(data)["VA"]
        theta = target[:, col].double()
        vmax = max(vmax, float(theta.abs().max()))
        _, fwd, _ = line_pairs(data)
        f, t = data[E].edge_index[:, fwd]
        w = data[E].edge_attr[fwd, YFT_TF_I].double()
        lap = torch.zeros(n, n, dtype=torch.float64)
        for (i, j), sg in (((f, f), 1.0), ((t, t), 1.0), ((f, t), -1.0), ((t, f), -1.0)):
            lap.index_put_((i, j), sg * w, accumulate=True)
        gi, gb = data.edge_index_dict[("gen", "connected_to", "bus")]
        known = (~data.mask_dict["gen"][gi, PG_H]).double()
        p = scatter(data["gen"].x[gi, PG_H].double() * known, gb, dim=0, dim_size=n, reduce="sum")
        p = p - data["bus"].x[:, PD_H].double()
        ref = data.mask_dict["REF"]
        free = ~ref
        th = theta.clone()  # REF entries: the known theta_ref
        th[free] = torch.linalg.solve(lap[free][:, free], p[free] - lap[free][:, ref] @ theta[ref])
        e = wrap_angle(th - theta)[m]
        add(s, "VA flat", (theta[m] ** 2).sum(), m.sum())
        add(s, "VA DC", (e**2).sum(), m.sum())
        add(s, "VA DC offset", m.sum() * e.mean() ** 2, m.sum())
        add(s, "VA DC rest", ((e - e.mean()) ** 2).sum(), m.sum())
        add(s, "dVA flat", ((theta[f] - theta[t]) ** 2).sum(), f.numel())
        add(s, "dVA DC", (wrap_angle((th[f] - th[t]) - (theta[f] - theta[t])) ** 2).sum(), f.numel())
        vc, vm = pred_masks(data)["VM"]
        add(s, "VM flat", ((target[vm, vc].double() - 1) ** 2).sum(), vm.sum())
    out = {k: math.sqrt(v[0] / v[1]) * (1 if k.startswith("VM") else D) for k, v in s.items()}
    print(case, f"max|VA| {vmax * D:.1f} deg |", " | ".join(f"{k} {v:.3g}" for k, v in out.items()), flush=True)
