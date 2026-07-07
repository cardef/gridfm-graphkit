# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R021 wiring smoke (M0-local): the HELM2 unpool through the real model path.

The R021 arm itself (KS-HELM2 vs R020 at case500/case2000, matched FLOPs) is
a cluster experiment, gated on R006. This smoke validates everything that
can fail locally, on case14, BEFORE cluster hours are spent:

Phase A -- operator ceiling through the model's runtime machinery.
  With TRUE boundary voltages fed to the prolongation (bypassing the learned
  coarse decoder), the model's helm2 path (runtime-file registry, batched
  gather, dense-LU solves, scatter back) must reproduce the pilot's
  operator-level result (idea-stage/helm_unpool_results.json, case14:
  HELM2 1.1e-4 vs affine 8.8e-3 median): helm2 error < 1e-3 p.u. median and
  >= 5x below the affine unpool on the same batches. This is the falsifiable
  check that the model wiring computes the same series the pilot validated.

Phase B -- short training run (r005 harness) with unpool: helm2.
  PASS = losses finite, canary |c2|/|c1| finite and median < 1 (convergent
  regime at case14 nominal load), R005 instruments populated for both arms.
  NOT a performance gate: case14 is too small to show unpool differences
  through a trained coarse decoder (pre-registered expectation).

Usage:  ../.venv/bin/python experiments/m0/r021_wiring_smoke.py [epochs]
"""

import json
import os.path as osp
import sys

import numpy as np
import pandas as pd
import torch

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from r005_signature_smoke import (  # noqa: E402
    _buckets,
    _check,
    _collect,
    _load,
    _rmse,
)
from gridfm_graphkit.io.param_handler import (  # noqa: E402
    get_loss_function,
    load_model,
)
from gridfm_graphkit.models.gnn_hetero_hier import PROLONG_RELATION  # noqa: E402
from gridfm_graphkit.utils.scatter import scatter_add  # noqa: E402

GRID = "case14_ieee"


def _true_voltages(root):
    bus = pd.read_parquet(osp.join(root, "raw", "bus_data.parquet"))
    out = {}
    for s, bs in bus.groupby("scenario"):
        bs = bs.sort_values("bus")
        out[int(s)] = bs["Vm"].to_numpy() * np.exp(1j * np.deg2rad(bs["Va"].to_numpy()))
    return out


def phase_a(epochsless_model, dm, root):
    """Operator ceiling with TRUE V_b through model._helm2_unpool."""
    vtrue = _true_voltages(root)
    rt = torch.load(
        dm.train_datasets[0][0].helm_runtime_path,
        weights_only=True,
    )
    boundary = rt["boundary_idx"].numpy()
    interior = rt["interior_idx"].numpy()

    err_aff, err_helm = [], []
    with torch.no_grad():
        for batch in dm.train_dataloader():
            scen = batch["scenario_id"].view(-1).tolist()
            ptr = batch["bus"].ptr
            num_bus = batch["bus"].num_nodes
            cptr = batch["cbus"].ptr

            # true V_b per cbus node, in cbus (=boundary) ordering
            vb = np.concatenate([vtrue[int(s)][boundary] for s in scen])
            vb_r = torch.tensor(vb.real, dtype=torch.float)
            vb_i = torch.tensor(vb.imag, dtype=torch.float)
            assert vb_r.numel() == int(cptr[-1])

            # c0 = P_sp V_b via the model's prolongation edges
            prol_src, prol_dst = batch.edge_index_dict[PROLONG_RELATION]
            p_attr = batch.edge_attr_dict[PROLONG_RELATION]
            pr, pi = p_attr[:, 0], p_attr[:, 1]
            msg_r = pr * vb_r[prol_src] - pi * vb_i[prol_src]
            msg_i = pr * vb_i[prol_src] + pi * vb_r[prol_src]
            c0_r = scatter_add(msg_r, prol_dst, dim=0, dim_size=num_bus)
            c0_i = scatter_add(msg_i, prol_dst, dim=0, dim_size=num_bus)

            v_aff = batch["bus"].v_aff
            c0 = torch.complex(c0_r, c0_i)
            v_helm = epochsless_model._helm2_unpool(
                batch,
                c0,
                torch.complex(v_aff[:, 0], v_aff[:, 1]),
                num_bus,
            )
            v_affine = c0 + torch.complex(v_aff[:, 0], v_aff[:, 1])

            gidx = ptr[:-1].unsqueeze(1) + torch.tensor(interior).unsqueeze(0)
            vt = torch.tensor(
                np.stack([vtrue[int(s)][interior] for s in scen]),
                dtype=torch.complex64,
            )
            err_aff.append((v_affine[gidx] - vt).abs().flatten())
            err_helm.append((v_helm[gidx] - vt).abs().flatten())

    ea = torch.cat(err_aff)
    eh = torch.cat(err_helm)
    res = {
        "affine_err_median": float(ea.median()),
        "affine_err_p90": float(ea.quantile(0.9)),
        "helm2_err_median": float(eh.median()),
        "helm2_err_p90": float(eh.quantile(0.9)),
        "improvement_x": float(ea.median() / eh.median()),
        "canary_last_batch": float(epochsless_model.helm_canary),
    }
    assert eh.median() < 1e-3, f"helm2 ceiling {eh.median():.2e} >= 1e-3"
    assert res["improvement_x"] >= 5, f"helm2 only {res['improvement_x']:.1f}x"
    return res


def phase_b(cfg_name, epochs):
    """r005-style short train, tracking the divergence canary."""
    args, dm = _load(cfg_name)
    loader = dm.train_dataloader()
    model = load_model(args)
    loss_fn = get_loss_function(args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    canaries, losses = [], []
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
            losses.append(float(ld["loss"].detach()))
            if model.helm_canary is not None:
                canaries.append(float(model.helm_canary))

    load, se_vm, n_vm, se_va, n_va = _collect(model, loader)
    res = {
        "config": cfg_name,
        "model": args.model.type,
        "unpool": getattr(args.model, "unpool", "affine"),
        "n_samples": int(load.numel()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "all_losses_finite": bool(np.isfinite(losses).all()),
        "overall": {
            "vm_rmse": _rmse(se_vm, n_vm),
            "va_rmse": _rmse(se_va, n_va),
        },
        "load_buckets": _buckets(load, se_vm, n_vm, se_va, n_va),
    }
    if canaries:
        res["canary"] = {
            "median": float(np.median(canaries)),
            "max": float(np.max(canaries)),
            "n": len(canaries),
        }
    assert res["all_losses_finite"], "non-finite loss in training"
    return res


def main(epochs):
    # Phase A: untrained model, helm2 config (no learned weights involved)
    args, dm = _load("KS_PF_case14_overfit_helm2.yaml")
    model = load_model(args)
    root = osp.join(REPO, "data", GRID)
    a = phase_a(model, dm, root)
    print("=== Phase A (operator ceiling, TRUE V_b, model runtime path) ===")
    print(
        f"  affine med={a['affine_err_median']:.3e}  "
        f"helm2 med={a['helm2_err_median']:.3e}  "
        f"({a['improvement_x']:.0f}x, canary={a['canary_last_batch']:.3f})",
    )

    out = {"epochs": epochs, "phase_a": a, "phase_b": []}
    print(f"\n=== Phase B (train smoke, {epochs} epochs) ===")
    for cfg in ("KS_PF_case14_overfit_helm2.yaml", "KS_PF_case14_overfit.yaml"):
        res = phase_b(cfg, epochs)
        _check(res)
        out["phase_b"].append(res)
        o = res["overall"]
        c = res.get("canary")
        print(
            f"  {res['unpool']:6s} VM={o['vm_rmse']:.3e} VA={o['va_rmse']:.3e} "
            f"loss {res['loss_first']:.2e}->{res['loss_last']:.2e}"
            + (f" canary med={c['median']:.3f} max={c['max']:.3f}" if c else ""),
        )
        ck = res.get("canary")
        if ck:
            assert ck["median"] < 1.0, "divergence canary >= 1 at case14 nominal"

    path = osp.join(HERE, "results", "r021_wiring_smoke.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPASS -- helm2 wiring validated end-to-end. wrote {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
