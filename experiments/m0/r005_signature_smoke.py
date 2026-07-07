# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R005 (M0): smoke-test the two M2 discriminating signatures are measurable.

Before spending cluster GPU-hours on the M2 gate (R020), confirm end-to-end on
LOCAL case14 data that both discriminating signatures come out finite and
separated, for BOTH models (KS + flat) through one identical eval harness:

  (1) per-quantity error split: VM RMSE vs VA RMSE  (long-range = angle).
      Already proven by r004_eval; re-checked here.
  (2) error vs load level: per-sample error bucketed by demand tercile.
      NEW instrument -- the failure-mode-#4 axis (linearization weakens under
      load). See memory m2-falsification-risks.

This is a PLUMBING check, not the gate: case14 is too small to show the KS
effect, and the models are trained only briefly. PASS = the instruments run
and produce sane, populated, distinct numbers -- NOT "KS wins".

Load-level proxy: sum over buses of |Pd|+|Qd| from the (globally MVA-normalized)
bus input x[:, PD_H/QD_H]. Monotone in true demand under HeteroDataMVANormalizer
(affine per-feature), so tercile ordering is valid. The cluster version (R010
eval) should use raw MVA demand.

Usage:  ../.venv/bin/python experiments/m0/r005_signature_smoke.py [epochs]
"""

import json
import os.path as osp
import sys

import torch
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
from gridfm_graphkit.datasets.globals import (  # noqa: E402
    PD_H,
    QD_H,
    VM_H,
    VA_H,
    VM_OUT,
    VA_OUT,
)
from gridfm_graphkit.utils.scatter import scatter_add  # noqa: E402

N_BUCKETS = 3


class _T:
    is_global_zero = True
    logger = None


def _load(cfg_name):
    with open(osp.join(HERE, cfg_name)) as f:
        cfg = yaml.safe_load(f)
    args = NestedNamespace(**cfg)
    import lightning as L

    L.seed_everything(args.seed, workers=True)
    dm = LitGridHeteroDataModule(args, data_dir=osp.join(REPO, "data"))
    dm.trainer = _T()
    dm.setup("fit")
    return args, dm


def _train(model, loss_fn, loader, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
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


def _collect(model, loader):
    """Per-sample arrays: load level and masked squared error for VM and VA."""
    model.eval()
    load, se_vm, n_vm, se_va, n_va = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            x = batch.x_dict["bus"]
            y = batch.y_dict["bus"]
            pred = out["bus"]
            mask = batch.mask_dict["bus"]
            bnode = batch["bus"].batch  # node -> graph id
            g = int(batch["bus"].ptr.numel() - 1)

            load_node = x[:, PD_H].abs() + x[:, QD_H].abs()
            lg = scatter_add(load_node, bnode, dim=0, dim_size=g)

            m_vm = mask[:, VM_H]
            m_va = mask[:, VA_H]
            se_vm_node = ((pred[:, VM_OUT] - y[:, VM_H]) ** 2) * m_vm
            se_va_node = ((pred[:, VA_OUT] - y[:, VA_H]) ** 2) * m_va
            se_vm_g = scatter_add(se_vm_node, bnode, dim=0, dim_size=g)
            se_va_g = scatter_add(se_va_node, bnode, dim=0, dim_size=g)
            n_vm_g = scatter_add(m_vm.float(), bnode, dim=0, dim_size=g)
            n_va_g = scatter_add(m_va.float(), bnode, dim=0, dim_size=g)

            load += lg.tolist()
            se_vm += se_vm_g.tolist()
            se_va += se_va_g.tolist()
            n_vm += n_vm_g.tolist()
            n_va += n_va_g.tolist()
    return (
        torch.tensor(load),
        torch.tensor(se_vm),
        torch.tensor(n_vm),
        torch.tensor(se_va),
        torch.tensor(n_va),
    )


def _rmse(se, n):
    n_t = float(n.sum())
    return (float(se.sum()) / n_t) ** 0.5 if n_t > 0 else float("nan")


def _buckets(load, se_vm, n_vm, se_va, n_va):
    """Tercile by load; per-bucket VM/VA RMSE."""
    order = torch.argsort(load)
    chunks = torch.tensor_split(order, N_BUCKETS)
    rows = []
    for bi, idx in enumerate(chunks):
        rows.append(
            {
                "bucket": bi,
                "n_samples": int(idx.numel()),
                "mean_load": float(load[idx].mean()),
                "vm_rmse": _rmse(se_vm[idx], n_vm[idx]),
                "va_rmse": _rmse(se_va[idx], n_va[idx]),
            },
        )
    return rows


def run_one(cfg_name, epochs):
    args, dm = _load(cfg_name)
    loader = dm.train_dataloader()
    model = load_model(args)
    loss_fn = get_loss_function(args)
    _train(model, loss_fn, loader, epochs)
    load, se_vm, n_vm, se_va, n_va = _collect(model, loader)
    return {
        "config": cfg_name,
        "model": args.model.type,
        "n_samples": int(load.numel()),
        "overall": {
            "vm_rmse": _rmse(se_vm, n_vm),
            "va_rmse": _rmse(se_va, n_va),
        },
        "load_buckets": _buckets(load, se_vm, n_vm, se_va, n_va),
    }


def _check(res):
    """PASS = both instruments produced sane, populated, distinct output."""
    o = res["overall"]
    assert o["vm_rmse"] == o["vm_rmse"] and o["vm_rmse"] > 0, "VM RMSE degenerate"
    assert o["va_rmse"] == o["va_rmse"] and o["va_rmse"] > 0, "VA RMSE degenerate"
    # per-quantity split is real, not aliased
    assert abs(o["vm_rmse"] - o["va_rmse"]) > 1e-9, "VM/VA aliased"
    pop = [b for b in res["load_buckets"] if b["n_samples"] > 0]
    assert len(pop) >= 2, "fewer than 2 load buckets populated"
    for b in pop:
        assert b["vm_rmse"] == b["vm_rmse"], "NaN VM RMSE in a load bucket"
        assert b["va_rmse"] == b["va_rmse"], "NaN VA RMSE in a load bucket"
    loads = [b["mean_load"] for b in pop]
    assert loads == sorted(loads), "load buckets not ordered by demand"


def main(epochs):
    out = {"epochs": epochs, "models": []}
    for cfg in ("KS_PF_case14_overfit.yaml", "FLAT_PF_case14_overfit.yaml"):
        res = run_one(cfg, epochs)
        _check(res)
        out["models"].append(res)

        print(f"\n=== {res['model']} ({cfg}, {res['n_samples']} samples) ===")
        o = res["overall"]
        print(f"  overall   VM={o['vm_rmse']:.3e}  VA={o['va_rmse']:.3e}")
        print("  load-bucket   n   mean_load     VM_rmse     VA_rmse")
        for b in res["load_buckets"]:
            print(
                f"    {b['bucket']} (lo→hi)   {b['n_samples']:2d}   "
                f"{b['mean_load']:8.3f}   {b['vm_rmse']:.3e}   {b['va_rmse']:.3e}",
            )
    out_path = osp.join(HERE, "results", "r005_signature_smoke.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPASS — both signatures measurable for both models. wrote {out_path}")
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
