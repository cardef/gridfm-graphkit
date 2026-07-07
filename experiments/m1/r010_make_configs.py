# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R010 (M1): emit the flat depth-sweep + KS run matrix, matched-FLOP.

Arms per grid (case500_goc, case2000_goc):

- ``flat_<grid>_d{8,16,32,48}``: natural-width sweep (hidden 48, heads 8) —
  the R010 depth axis;
- ``flat_<grid>_d{8,16,32,48}_iso``: width-adjusted iso-FLOP arms — hidden
  size chosen per depth so forward FLOPs land within [0.9, 1.1] of the KS
  reference (R003 showed no natural-width depth lands in the window:
  0.84x at d8, 1.78x at d16 @case2000). KS is held FIXED at the reference
  (hidden 48, heads 8, L_f/L_c/L_f' = 4/8/4) and the FLAT side is adjusted:
  the falsifier baseline gets the width benefit (fairness per the
  engineering plan's E7 note), and the treatment stays identical across
  comparisons;
- ``ks_<grid>``: the KS-2-level reference with lambda_v on
  (CoarseVoltageMSE 0.2 alongside the repo's standard PF recipe).

FLOPs come from the pre-registered ``gridfm_graphkit.utils.flops`` (R003).
The full pairing table (per-depth matched widths + ratios, natural-width
ratios) is written to ``results/r010_matrix.json``.

Training recipe: the repo's example HGNS PF recipe
(``examples/config/HGNS_PF_datakit_case500.yaml``): LayeredWeightedPhysics
0.1 (base_weight 0.5) + MaskedBusMSE 0.9, AdamW lr 5e-4; batch size 8
(case500) / 4 (case2000) from the E001 budgets. All configs set
``data.consolidated: true`` (E003) and ``data.same_grid_batches: true``
(E005 -> static shapes for torch.compile).

Seeds: ``--seeds 0 1 2`` writes seed-suffixed copies (the CLI has no seed
override flag; seeds live in the YAML). Default: seed 0 only.
"""

import argparse
import json
import os.path as osp
import sys

import yaml

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(REPO, "idea-stage"))
sys.path.insert(0, osp.join(REPO, "experiments", "m0"))

from r002_precompute import GRIDS  # noqa: E402
from r003_flops import BASE_CFG, grid_sizes  # noqa: E402
from gridfm_graphkit.utils.flops import (  # noqa: E402
    gns_hetero_hier_flops,
    gns_heterogeneous_flops,
)

M1_GRIDS = ["case500_goc", "case2000_goc"]
DEPTHS = (8, 16, 32, 48)
BATCH_SIZE = {"case500_goc": 8, "case2000_goc": 4}
WINDOW = (0.9, 1.1)


def matched_width(sizes, depth, ks_flops):
    """Integer hidden_size whose flat FLOPs at `depth` are closest to KS."""
    cfg = dict(BASE_CFG)
    cfg["num_layers"] = depth
    best = None
    for h in range(8, 129):
        cfg["hidden_size"] = h
        r = gns_heterogeneous_flops(sizes, cfg) / ks_flops
        if best is None or abs(r - 1) < abs(best[1] - 1):
            best = (h, r)
    return best  # (hidden_size, ratio flat/KS)


def common_sections(grid):
    return {
        "callbacks": {"patience": 100, "tol": 0},
        "task": {"task_name": "PowerFlow"},
        "data": {
            "baseMVA": 100,
            "mask_value": 0.0,
            "normalization": "HeteroDataMVANormalizer",
            "networks": [grid],
            "scenarios": [10000],
            "test_ratio": 0.1,
            "val_ratio": 0.1,
            "workers": 8,
            "split_by_load_scenario_idx": True,
            "consolidated": True,  # E003 mmap store
            "same_grid_batches": True,  # E005 static-shape batches
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "learning_rate": 0.0005,
            "lr_decay": 0.7,
            "lr_patience": 5,
        },
        "seed": 0,
        "training": {
            "batch_size": BATCH_SIZE[grid],
            "epochs": 200,
            "loss_weights": [0.1, 0.9],
            "losses": ["LayeredWeightedPhysics", "MaskedBusMSE"],
            "loss_args": [{"base_weight": 0.5}, {}],
            "accelerator": "auto",
            "devices": "auto",
            "strategy": "auto",
        },
        "verbose": False,
    }


def flat_config(grid, depth, hidden):
    cfg = common_sections(grid)
    cfg["model"] = {
        "type": "GNS_heterogeneous",
        "attention_head": 8,
        "edge_dim": 10,
        "hidden_size": hidden,
        "input_bus_dim": 15,
        "input_gen_dim": 6,
        "output_bus_dim": 2,
        "output_gen_dim": 1,
        "num_layers": depth,
    }
    return cfg


def ks_config(grid):
    cfg = common_sections(grid)
    cfg["data"]["hierarchy"] = {"enable": True}  # default target_frac/tol
    cfg["model"] = {
        "type": "GNS_hetero_hier",
        "attention_head": 8,
        "edge_dim": 10,
        "hidden_size": BASE_CFG["hidden_size"],
        "input_bus_dim": 15,
        "input_gen_dim": 6,
        "input_cbus_dim": 8,
        "output_bus_dim": 2,
        "output_gen_dim": 1,
        "num_layers_fine": BASE_CFG["num_layers_fine"],
        "num_layers_coarse": BASE_CFG["num_layers_coarse"],
        "num_layers_fine_post": BASE_CFG["num_layers_fine_post"],
    }
    # lambda_v on: boundary-voltage supervision for the coarse path
    cfg["training"]["losses"].append("CoarseVoltageMSE")
    cfg["training"]["loss_weights"].append(0.2)
    cfg["training"]["loss_args"].append({})
    return cfg


def emit(cfg, name, cfg_dir, seeds):
    for seed in seeds:
        c = dict(cfg)
        c["seed"] = seed
        fname = f"{name}.yaml" if seed == 0 else f"{name}_seed{seed}.yaml"
        path = osp.join(cfg_dir, fname)
        with open(path, "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ns = ap.parse_args()

    import os

    cfg_dir = osp.join(HERE, "configs")
    res_dir = osp.join(HERE, "results")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    matrix = {}
    for grid in M1_GRIDS:
        sizes = grid_sizes(grid, GRIDS[grid])
        ks = gns_hetero_hier_flops(sizes, BASE_CFG)
        short = grid.split("_")[0]
        entry = {"ks_flops": ks, "arms": {}}

        emit(ks_config(grid), f"ks_{short}", cfg_dir, ns.seeds)

        for depth in DEPTHS:
            nat_cfg = dict(BASE_CFG)
            nat_cfg["num_layers"] = depth
            nat = gns_heterogeneous_flops(sizes, nat_cfg)
            h_iso, r_iso = matched_width(sizes, depth, ks)
            in_window = WINDOW[0] <= r_iso <= WINDOW[1]
            entry["arms"][f"d{depth}"] = {
                "natural_h": BASE_CFG["hidden_size"],
                "natural_flops": nat,
                "natural_ratio_over_ks": nat / ks,
                "iso_h": h_iso,
                "iso_ratio_over_ks": r_iso,
                "iso_within_window": in_window,
            }
            emit(
                flat_config(grid, depth, BASE_CFG["hidden_size"]),
                f"flat_{short}_d{depth}",
                cfg_dir,
                ns.seeds,
            )
            emit(
                flat_config(grid, depth, h_iso),
                f"flat_{short}_d{depth}_iso",
                cfg_dir,
                ns.seeds,
            )
            print(
                f"  {grid} d{depth}: natural {nat / ks:.3f}x KS | "
                f"iso h={h_iso} -> {r_iso:.3f}x KS "
                f"{'OK' if in_window else 'OUT OF WINDOW'}",
            )
        matrix[grid] = entry

    out = osp.join(res_dir, "r010_matrix.json")
    with open(out, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"wrote {out}")
