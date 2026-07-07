# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""E001 (M0): torch.profiler baseline per grid size, train + inference.

CPU-only pass (no local GPU): rankings gate the CPU-verifiable E-items
(E002 native-scatter migration, E003 mmap store) and give the M0
"real per-epoch timings" re-baseline. GPU op rankings will differ — the
cluster reruns this before any GPU-targeted E-item lands.

Profiles the flat baseline GNS_heterogeneous at example-config size
(hidden 48, heads 8, 12 layers) with the default PF loss recipe.
Writes experiments/m0/results/e001_profile.json.
"""

import json
import os.path as osp
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

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

GRIDS = {  # grid -> batch size (CPU memory / step-time budget)
    "case14_ieee": 32,
    "case118_ieee": 16,
    "case500_goc": 8,
    "case2000_goc": 4,
}
N_STEPS = 10


class _T:
    is_global_zero = True
    logger = None


def make_args(network, batch_size):
    return NestedNamespace(
        **{
            "seed": 0,
            "verbose": False,
            "task": {"task_name": "PowerFlow"},
            "data": {
                "baseMVA": 100,
                "mask_value": 0.0,
                "normalization": "HeteroDataMVANormalizer",
                "networks": [network],
                "scenarios": [100000],
                "test_ratio": 0.1,
                "val_ratio": 0.1,
                "workers": 0,
                "split_by_load_scenario_idx": True,
            },
            "model": {
                "type": "GNS_heterogeneous",
                "attention_head": 8,
                "edge_dim": 10,
                "hidden_size": 48,
                "input_bus_dim": 15,
                "input_gen_dim": 6,
                "output_bus_dim": 2,
                "output_gen_dim": 1,
                "num_layers": 12,
            },
            "optimizer": {
                "beta1": 0.9,
                "beta2": 0.999,
                "learning_rate": 5e-4,
                "lr_decay": 0.7,
                "lr_patience": 5,
            },
            "training": {
                "batch_size": batch_size,
                "epochs": 1,
                "losses": ["LayeredWeightedPhysics", "MaskedBusMSE"],
                "loss_args": [{"base_weight": 0.5}, {}],
                "loss_weights": [0.1, 0.9],
            },
        },
    )


def top_ops(prof, n=10):
    rows = []
    for ev in sorted(
        prof.key_averages(),
        key=lambda e: e.self_cpu_time_total,
        reverse=True,
    )[:n]:
        rows.append(
            {
                "op": ev.key,
                "self_cpu_ms": ev.self_cpu_time_total / 1e3,
                "calls": ev.count,
            },
        )
    return rows


def profile_grid(network, batch_size):
    args = make_args(network, batch_size)
    import lightning as L

    L.seed_everything(0, workers=True)
    dm = LitGridHeteroDataModule(args, data_dir=osp.join(REPO, "data"))
    dm.trainer = _T()
    dm.setup("fit")
    loader = dm.train_dataloader()
    model = load_model(args)
    loss_fn = get_loss_function(args)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    def step(batch, train):
        if train:
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
        else:
            with torch.no_grad():
                model(batch)

    batches = []
    it = iter(loader)
    for _ in range(N_STEPS + 2):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader)
            batches.append(next(it))

    res = {"batch_size": batch_size, "n_steps": N_STEPS}
    for mode, train in (("train", True), ("infer", False)):
        model.train() if train else model.eval()
        step(batches[0], train)  # warmup
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for b in batches[1 : N_STEPS + 1]:
                step(b, train)
        wall = time.perf_counter() - t0
        res[mode] = {
            "s_per_step": wall / N_STEPS,
            "samples_per_s": batch_size * N_STEPS / wall,
            "top10_ops": top_ops(prof),
        }
        print(
            f"{network} {mode}: {wall / N_STEPS:.3f} s/step "
            f"({batch_size * N_STEPS / wall:.1f} samples/s) | top op: "
            f"{res[mode]['top10_ops'][0]['op']}",
            flush=True,
        )
    return res


if __name__ == "__main__":
    results = {"device": "cpu", "model": "GNS_heterogeneous h48/8x12L"}
    for g, bs in GRIDS.items():
        results[g] = profile_grid(g, bs)
    out = osp.join(HERE, "results", "e001_profile.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
