# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""R003 (M0): pre-registered FLOP accounting — per-grid tables + matched pairs.

For every grid, compute per-sample forward FLOPs for the flat baseline at
depths {8,16,32,48} and the KS-2-level reference config, and report the
matched-FLOP flat depth (target ratio in [0.9, 1.1]) that M1/M2 will use.

Graph sizes come from the generated parquet data where available, else from
the pglib .m file; hierarchy sizes from the same operator code as R002.
Writes experiments/m0/results/r003_flops.json.
"""

import json
import os.path as osp
import sys

import pandas as pd
import torch

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(REPO, "idea-stage"))

from pilot_cpu import load_case, ybus  # noqa: E402
from r002_precompute import GRIDS  # noqa: E402
from gridfm_graphkit.datasets.hierarchy import (  # noqa: E402
    build_operators,
    hierarchy_cache_name,
    select_boundary,
)
from gridfm_graphkit.utils.flops import (  # noqa: E402
    GraphSizes,
    gns_hetero_hier_flops,
    matched_flat_depth,
)

# reference configs (hidden/heads follow the example HGNS configs; the KS
# depth split L_f/L_c/L_f' = 4/8/4 is the M1 starting point, tuned there)
BASE_CFG = {
    "hidden_size": 48,
    "attention_head": 8,
    "edge_dim": 10,
    "input_bus_dim": 15,
    "input_gen_dim": 6,
    "input_cbus_dim": 8,
    "output_bus_dim": 2,
    "output_gen_dim": 1,
    "num_layers_fine": 4,
    "num_layers_coarse": 8,
    "num_layers_fine_post": 4,
}


def grid_sizes(name, mpath):
    """Element counts + hierarchy operator counts for one grid.

    Operator counts come from the R006-gated hierarchy cache when local data
    exists (the tensors the model actually consumes; the recon gate needs
    scenarios). Fallback: mass-only operators from the .m file (case9241 at
    M0 -- k there is a lower bound until the cluster re-runs R006).
    """
    bus, branch, gen = load_case(mpath)
    Y, _, _, br = ybus(bus, branch)
    nb = Y.shape[0]
    n_gen = int((gen[:, 7] > 0).sum())
    e_busbus = 2 * br.shape[0]
    cache_path = osp.join(REPO, "data", name, "processed", hierarchy_cache_name())
    if osp.exists(cache_path):
        cache = torch.load(cache_path, weights_only=True)
        n_cbus = int(cache["boundary_idx"].numel())
        e_coarse = int(cache["coarse_edge_index"].shape[1])
        nnz_prolong = int(cache["prolong_edge_index"].shape[1])
    else:
        bus0 = pd.DataFrame(
            {
                "PV": (bus[:, 1] == 2).astype(int),
                "REF": (bus[:, 1] == 3).astype(int),
                "vn_kv": bus[:, 9],
            },
        )
        boundary, interior = select_boundary(bus0)
        ops, _, stats = build_operators(Y, boundary, interior)
        n_cbus = len(boundary)
        e_coarse = int(ops["coarse_edge_index"].shape[1])
        nnz_prolong = int(ops["prolong_edge_index"].shape[1])
    return GraphSizes(
        n_bus=nb,
        n_gen=n_gen,
        e_busbus=e_busbus,
        n_cbus=n_cbus,
        e_coarse=e_coarse,
        nnz_prolong=nnz_prolong,
        n_int=nb - n_cbus,
    )


if __name__ == "__main__":
    results = {}
    for name, mpath in GRIDS.items():
        s = grid_sizes(name, mpath)
        ks = gns_hetero_hier_flops(s, BASE_CFG)
        best_depth, ratio, table = matched_flat_depth(s, BASE_CFG)
        results[name] = {
            "sizes": vars(s),
            "ks_flops": ks,
            "flat_flops_by_depth": {str(k): v for k, v in table["flat"].items()},
            "matched_flat_depth": best_depth,
            "matched_ratio_flat_over_ks": ratio,
            "matched_within_0.9_1.1": bool(0.9 <= ratio <= 1.1),
        }
        print(
            f"{name}: KS={ks / 1e9:.2f} GFLOP | matched flat depth={best_depth} "
            f"(ratio {ratio:.3f}) | flat@8={table['flat'][8] / 1e9:.2f} "
            f"@48={table['flat'][48] / 1e9:.2f}",
        )

    out = osp.join(HERE, "results", "r003_flops.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
