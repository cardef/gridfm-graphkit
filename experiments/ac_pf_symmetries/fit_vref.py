"""Fit the linear regressor of the REF bus's voltage magnitude on the PV setpoints and the total loads (the C7 fact:
within a grid the setpoint is a near-deterministic function of the inputs) on the training scenarios of a grid, and
save its coefficients for r2_classical.py --vref-coef (Piano K's hybrid reference PH, K-P4).

python -m experiments.ac_pf_symmetries.fit_vref case118_k --split data/case118_k/splits/official_seed0.json --out <coef.json>
"""

import argparse
import json
import math
from types import SimpleNamespace

import numpy as np
from torch_geometric.loader import DataLoader

from gridfm_graphkit.datasets.globals import PD_H, QD_H, VM_H, VM_OUT
from gridfm_graphkit.tasks.pf_task import _build_bus_target

from .multigrid import grid_splits
from .run import make_args


def features(data):
    x = data["bus"].x.double().numpy()
    pv = data.mask_dict["PV"].numpy()
    setp = x[pv, VM_H]
    return [1.0, setp.mean(), setp.min(), setp.max(), x[:, PD_H].sum(), x[:, QD_H].sum()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("--split", required=True, help="{train, val, test} json; the regressor is fitted on train")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    args = make_args(SimpleNamespace(epochs=1, batch_size=1, patience=1, layers=12, hidden=48))
    args.data.networks = [a.case]
    split = json.load(open(a.split))
    h = grid_splits(a.case, args, 0, 1024, split=split)
    X, y = [], []
    for part in ("train", "val"):
        for data in DataLoader(h[part], batch_size=1):
            target = _build_bus_target(data, data["bus"].x.size(0))[0].double().numpy()
            ref = data.mask_dict["REF"].numpy()
            if part == "train":
                X.append(features(data))
                y.append(float(target[ref, VM_OUT][0]))
    X, y = np.array(X), np.array(y)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    res, tot = ((y - X @ coef) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    out = {"case": a.case, "split": a.split, "n_train": len(y), "features": ["1", "mean PV setpoint", "min", "max", "sum Pd", "sum Qd"],
           "coef": coef.tolist(), "r2_train": float(1 - res / tot) if tot > 0 else None, "rmse_train": math.sqrt(res / len(y))}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("n_train", "r2_train", "rmse_train")}))


if __name__ == "__main__":
    main()
