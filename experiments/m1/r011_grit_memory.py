# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R011 (M1): GRIT memory curve case14 -> case2000 (B5, the memory-wall run).

Two phases, per grid:

- Phase A (CPU, runs anywhere): full-RRWP precompute cost — seconds per
  sample and the size of the relative PE (nnz = N^2 pairs x ksteps floats),
  the O(N^2) memory driver of full-attention GRIT.
- Phase B (CUDA only): peak GPU memory of one real train step
  (forward + loss + backward) at batch sizes {1, 2, 4, 8}, stopping at the
  first OOM ("no training past OOM" per the frozen tracker) and recording
  the boundary. On a non-CUDA host phase B still executes one CPU step at
  batch size 1 as a wiring check (no memory numbers — CPU allocators don't
  give a meaningful peak, and the B5 claim is about accelerator memory).

Model: the repo's as-shipped GRIT example scale (hidden 496, 7 layers,
heads 8, RWSE node encodings) with FULL attention (``full_attn: true``, no
RRWP top-k) — B5's claim is about full-attention GRIT; the top-k variant
is the mitigation, not the subject.

Quadratic fit: peak bytes at batch 1 vs n_bus, np.polyfit deg 2, over the
grids that completed without OOM (needs >= 3 points and CUDA).

Writes experiments/m1/results/r011_grit_memory.json.

Usage:
    python experiments/m1/r011_grit_memory.py [--grids case14_ieee ...]
                                              [--batch-sizes 1 2 4 8]
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from torch_geometric.data import Batch  # noqa: E402

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (  # noqa: E402
    LitGridHeteroDataModule,
)
from gridfm_graphkit.datasets.posenc_stats import ComputePosencStat  # noqa: E402
from gridfm_graphkit.io.param_handler import (  # noqa: E402
    NestedNamespace,
    get_loss_function,
    load_model,
)

GRIDS = [
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case118_ieee",
    "case500_goc",
    "case2000_goc",
]
KSTEPS = 21


class _T:
    is_global_zero = True
    logger = None


def make_args(network, enable_rrwp):
    """GRIT PF config at the example scale (GRIT_PF_datakit_case14.yaml),
    with full RRWP (no topk) instead of the example's top-k sparsification."""
    return NestedNamespace(
        **{
            "seed": 0,
            "verbose": False,
            "task": {"task_name": "PowerFlow"},
            "data": {
                "baseMVA": 100,
                "mask_type": "rnd",
                "mask_ratio": 0.5,
                "mask_value": 0.0,
                "normalization": "HeteroDataMVANormalizer",
                "networks": [network],
                "scenarios": [16],
                "test_ratio": 0.1,
                "val_ratio": 0.1,
                "workers": 0,
                "split_by_load_scenario_idx": True,
                "posenc_RRWP": {
                    "enable": enable_rrwp,  # no topk key -> full RRWP
                    "ksteps": KSTEPS,
                    "cache": False,  # measure the precompute, don't hide it
                },
                "posenc_RWSE": {
                    "enable": True,
                    "cache": False,
                    "kernel": {"times": KSTEPS},
                },
            },
            "model": {
                "type": "GRIT",
                "attention_head": 8,
                "dropout": 0.1,
                "edge_dim": 10,
                "hidden_size": 496,
                "input_dim": 16,
                "input_bus_dim": 16,
                "input_gen_dim": 6,
                "output_bus_dim": 6,
                "output_gen_dim": 0,
                "num_layers": 7,
                "act": "relu",
                "encoder": {
                    "node_encoder": True,
                    "edge_encoder": True,
                    "node_encoder_name": "RWSE",
                    "node_encoder_bn": True,
                    "edge_encoder_bn": True,
                    "posenc_RWSE": {"pe_dim": 20, "raw_norm_type": "batchnorm"},
                },
                "gt": {
                    "layer_type": "GritTransformer",
                    "layer_norm": False,
                    "batch_norm": True,
                    "update_e": True,
                    "attn_dropout": 0.2,
                    "attn": {
                        "clamp": 5.0,
                        "act": "relu",
                        "full_attn": True,  # full RRWP -> full attention
                        "edge_enhance": True,
                        "O_e": True,
                        "norm_e": True,
                        "signed_sqrt": True,
                        "bn_momentum": 0.1,
                        "bn_no_runner": False,
                    },
                },
            },
            "optimizer": {
                "beta1": 0.9,
                "beta2": 0.999,
                "learning_rate": 1e-4,
                "lr_decay": 0.7,
                "lr_patience": 10,
            },
            "training": {
                "batch_size": 1,
                "epochs": 1,
                "losses": ["PBE", "MaskedReconstructionMSE"],
                "loss_args": [{}, {}],
                "loss_weights": [0.99, 0.01],
            },
        },
    )


def setup_dm(args):
    dm = LitGridHeteroDataModule(args, data_dir=osp.join(REPO, "data"))
    dm.trainer = _T()
    dm.setup("fit")
    return dm


def phase_a(network):
    """Full-RRWP precompute: s/sample + relative-PE size."""
    args = make_args(network, enable_rrwp=False)
    dm = setup_dm(args)
    base = dm.train_datasets[0][0]
    pe = ComputePosencStat(pe_types=["RRWP"], cfg=args.data)
    t0 = time.perf_counter()
    s = pe(base.clone())
    dt = time.perf_counter() - t0
    rel = s["bus", "rrwp", "bus"]
    n = int(s["bus"].num_nodes)
    out = {
        "n_bus": n,
        "rrwp_s_per_sample": dt,
        "rrwp_rel_nnz": int(rel.edge_index.shape[1]),
        "rrwp_rel_bytes": int(rel.edge_attr.numel() * rel.edge_attr.element_size()),
        "rrwp_dense_pairs": n * n,
    }
    print(
        f"{network} phase A: RRWP {dt:.2f} s/sample, "
        f"rel PE {out['rrwp_rel_nnz']} nnz "
        f"({out['rrwp_rel_bytes'] / 1e6:.1f} MB)",
        flush=True,
    )
    return out


def train_step(model, loss_fn, opt, batch):
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


def phase_b(network, batch_sizes):
    """Peak accelerator memory per batch size until OOM."""
    cuda = torch.cuda.is_available()
    args = make_args(network, enable_rrwp=True)
    dm = setup_dm(args)
    ds = dm.train_datasets[0]
    samples = [ds[i] for i in range(min(max(batch_sizes), len(ds)))]

    device = torch.device("cuda" if cuda else "cpu")
    model = load_model(args).to(device)
    loss_fn = get_loss_function(args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    res = {"device": str(device), "peak_bytes": {}, "oom_at_batch_size": None}
    for bs in sorted(batch_sizes):
        if bs > len(samples):
            print(f"{network} phase B: bs={bs} skipped (only {len(samples)} samples)")
            break
        batch = Batch.from_data_list(samples[:bs]).to(device)
        try:
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            train_step(model, loss_fn, opt, batch)
            if cuda:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                res["peak_bytes"][str(bs)] = int(peak)
                print(
                    f"{network} phase B: bs={bs} peak {peak / 1e9:.2f} GB",
                    flush=True,
                )
            else:
                print(f"{network} phase B: bs={bs} CPU wiring step OK", flush=True)
                break  # CPU: one wiring check only, no memory numbers
        except torch.cuda.OutOfMemoryError:
            res["oom_at_batch_size"] = bs
            print(f"{network} phase B: bs={bs} OOM — stopping (no training past OOM)")
            torch.cuda.empty_cache()
            break
        finally:
            del batch
            if cuda:
                torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="*", default=GRIDS)
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument(
        "--phase-a-only",
        action="store_true",
        help="RRWP precompute timing only (the CPU-runnable half); skip the "
        "train-step memory sweep. Full-attention GRIT steps at case500+ "
        "need tens of GB — run phase B on the cluster GPU.",
    )
    ns = ap.parse_args()

    results = {
        "model": "GRIT h496/8x7L, full attention, RRWP ksteps 21",
        "cuda": torch.cuda.is_available(),
        "grids": {},
    }
    for g in ns.grids:
        entry = phase_a(g)
        if not ns.phase_a_only:
            entry.update(phase_b(g, ns.batch_sizes))
        results["grids"][g] = entry

    # quadratic fit of bs=1 peak memory vs n_bus (CUDA numbers only)
    pts = [
        (e["n_bus"], e["peak_bytes"]["1"])
        for e in results["grids"].values()
        if "1" in e.get("peak_bytes", {})
    ]
    if len(pts) >= 3:
        x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        coeffs = np.polyfit(x, y, 2)
        pred = np.polyval(coeffs, x)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        results["quadratic_fit"] = {
            "coeffs_bytes_per_nbus": [float(c) for c in coeffs],
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
            "n_points": len(pts),
        }
        print(f"quadratic fit: {coeffs} (R^2={results['quadratic_fit']['r_squared']})")
    else:
        results["quadratic_fit"] = None
        print("quadratic fit skipped (need >=3 CUDA bs=1 points)")

    os.makedirs(osp.join(HERE, "results"), exist_ok=True)
    out = osp.join(HERE, "results", "r011_grit_memory.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
