# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R004 (M0): direct overfit gate — train to FINAL weights, no val selection.

The Lightning path only persists the best-val checkpoint, which in an
overfit regime freezes early (val loss worsens as train fit deepens). This
script trains GNS_hetero_hier in-process on the case14 train split and
reports the final-weights masked train VM/VA RMSE — the actual M0 gate.
"""

import json
import os.path as osp
import sys
import time

import torch
import torch.nn.functional as F
import yaml

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from gridfm_graphkit.io.param_handler import (  # noqa: E402
    NestedNamespace,
    load_model,
    get_loss_function,
)
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (  # noqa: E402
    LitGridHeteroDataModule,
)
from gridfm_graphkit.datasets.globals import VM_H, VA_H, VM_OUT, VA_OUT  # noqa: E402

EPOCHS = 3000


class _T:
    is_global_zero = True
    logger = None


def train_rmse(model, loader):
    model.eval()
    se_vm = se_va = 0.0
    n_vm = n_va = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            mask, y, pred = batch.mask_dict["bus"], batch.y_dict["bus"], out["bus"]
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
    model.train()
    return (se_vm / n_vm) ** 0.5, (se_va / n_va) ** 0.5


if __name__ == "__main__":
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "KS_PF_case14_overfit.yaml"
    out_name = sys.argv[2] if len(sys.argv) > 2 else "r004_overfit_direct.json"
    with open(osp.join(HERE, cfg_name)) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["batch_size"] = 64  # full-batch
    args = NestedNamespace(**cfg)
    import lightning as L

    L.seed_everything(args.seed, workers=True)
    dm = LitGridHeteroDataModule(args, data_dir=osp.join(REPO, "data"))
    dm.trainer = _T()
    dm.setup("fit")
    loader = dm.train_dataloader()

    model = load_model(args)
    loss_fn = get_loss_function(args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        for batch in loader:
            opt.zero_grad()
            out = model(batch)
            ld = loss_fn(
                out,
                batch.y_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
                batch.mask_dict,
                model=model,
                x_dict=batch.x_dict,
            )
            ld["loss"].backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 500 == 0:
            rv, ra = train_rmse(model, loader)
            print(
                f"epoch {epoch + 1}: loss={float(ld['loss']):.2e} "
                f"train RMSE VM={rv:.2e} VA={ra:.2e}",
                flush=True,
            )

    rv, ra = train_rmse(model, loader)
    res = {
        "final_train_rmse_vm_pu": rv,
        "final_train_rmse_va_rad": ra,
        "epochs": EPOCHS,
        "train_time_s": time.perf_counter() - t0,
        "config": f"{cfg_name} (batch 64, cosine 1e-3->1e-5)",
    }
    print(json.dumps(res, indent=2))
    with open(osp.join(HERE, "results", out_name), "w") as f:
        json.dump(res, f, indent=2)
