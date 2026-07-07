# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R004 (M0): evaluate the KS-2-level overfit gate on the case14 train split.

Loads the best (val-selected) checkpoint from the given mlruns artifact dir,
recomputes the identical train split (same seed/config), and reports masked
VM/VA RMSE (p.u. / radians) plus the coarse-decoder boundary RMSE.
Gate: train VM/VA RMSE ~ 0 (wiring works, model can fit).
"""

import json
import os.path as osp
import sys

import torch
import torch.nn.functional as F
import yaml

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from gridfm_graphkit.io.param_handler import NestedNamespace, get_task  # noqa: E402
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (  # noqa: E402
    LitGridHeteroDataModule,
)
from gridfm_graphkit.datasets.globals import VM_H, VA_H, VM_OUT, VA_OUT  # noqa: E402


class _T:
    is_global_zero = True
    logger = None


def main(ckpt, config_path):
    with open(config_path) as f:
        args = NestedNamespace(**yaml.safe_load(f))
    import lightning as L

    L.seed_everything(args.seed, workers=True)
    dm = LitGridHeteroDataModule(args, data_dir=osp.join(REPO, "data"))
    dm.trainer = _T()
    dm.setup("fit")
    task = get_task(args, dm.data_normalizers)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    task.load_state_dict(state)
    task.eval()

    se_vm, se_va, n_vm, n_va, se_c, n_c = 0.0, 0.0, 0, 0, 0.0, 0
    with torch.no_grad():
        for batch in dm.train_dataloader():
            out = task.model(batch)
            mask = batch.mask_dict["bus"]
            y = batch.y_dict["bus"]
            pred = out["bus"]
            m_vm, m_va = mask[:, VM_H], mask[:, VA_H]
            se_vm += F.mse_loss(
                pred[m_vm, VM_OUT],
                y[m_vm, VM_H],
                reduction="sum",
            ).item()
            se_va += F.mse_loss(
                pred[m_va, VA_OUT],
                y[m_va, VA_H],
                reduction="sum",
            ).item()
            n_vm += int(m_vm.sum())
            n_va += int(m_va.sum())
            sb, sc = batch.edge_index_dict[("bus", "seeds", "cbus")]
            se_c += F.mse_loss(
                out["cbus"][sc],
                y[sb][:, [VM_H, VA_H]],
                reduction="sum",
            ).item()
            n_c += 2 * int(sc.numel())

    res = {
        "train_rmse_vm_pu": (se_vm / n_vm) ** 0.5,
        "train_rmse_va_rad": (se_va / n_va) ** 0.5,
        "train_rmse_coarse_vmva": (se_c / n_c) ** 0.5,
        "n_masked_vm": n_vm,
        "n_masked_va": n_va,
        "checkpoint": ckpt,
    }
    print(json.dumps(res, indent=2))
    out_path = osp.join(HERE, "results", "r004_overfit.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
