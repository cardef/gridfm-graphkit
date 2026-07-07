# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""E003 (M0, plan E2): mmap consolidated store vs per-file disk dataset.

The repo already ships the graphzero-inspired store as
``HeteroGridDatasetMMap`` (``data.consolidated: true``); this run is the
plan's validation gate for it:
1) byte-identical samples vs ``HeteroGridDatasetDisk`` (same fitted
   normalizer, no task transform -- isolates the data path),
2) dataloader-only samples/s A/B at num_workers {0, 4}.

Writes experiments/m0/results/e003_mmap_ab.json.
"""

import json
import os.path as osp
import sys
import time

import torch
from torch_geometric.loader import DataLoader

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from gridfm_graphkit.datasets.powergrid_hetero_dataset import (  # noqa: E402
    HeteroGridDatasetDisk,
    HeteroGridDatasetMMap,
)
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer  # noqa: E402
from gridfm_graphkit.io.param_handler import NestedNamespace  # noqa: E402

GRIDS = ["case118_ieee", "case2000_goc"]
BATCH = 16
PASSES = 3


def fitted_normalizer(root, n):
    args = NestedNamespace(**{"data": {"baseMVA": 100}})
    norm = HeteroDataMVANormalizer(args)
    norm.fit(osp.join(root, "raw"), list(range(n)))
    return norm


def identical(a, b):
    da, db = a.to_dict(), b.to_dict()
    if set(map(str, da.keys())) != set(map(str, db.keys())):
        return False
    for k, va in da.items():
        vb = db[k]
        for attr, ta in va.items():
            tb = vb.get(attr)
            if not isinstance(ta, torch.Tensor) or not torch.equal(ta, tb):
                return False
    return True


def bench(ds, workers):
    loader = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    n = len(ds)
    # warmup pass (page cache, worker spawn)
    for _ in loader:
        pass
    t0 = time.perf_counter()
    for _ in range(PASSES):
        for _ in loader:
            pass
    dt = time.perf_counter() - t0
    return n * PASSES / dt


if __name__ == "__main__":
    results = {}
    for grid in GRIDS:
        root = osp.join(REPO, "data", grid)
        disk = HeteroGridDatasetDisk(root, data_normalizer=None)
        n = len(disk)
        norm = fitted_normalizer(root, n)
        disk.data_normalizer = norm
        mmap_ds = HeteroGridDatasetMMap(root, data_normalizer=norm)
        assert len(mmap_ds) == n, (len(mmap_ds), n)

        same = all(identical(disk.get(i), mmap_ds.get(i)) for i in range(n))
        r = {"n_scenarios": n, "byte_identical_all_samples": same}
        for w in (0, 4):
            r[f"disk_samples_per_s_w{w}"] = round(bench(disk, w), 1)
            r[f"mmap_samples_per_s_w{w}"] = round(bench(mmap_ds, w), 1)
            r[f"speedup_w{w}"] = round(
                r[f"mmap_samples_per_s_w{w}"] / r[f"disk_samples_per_s_w{w}"],
                3,
            )
        results[grid] = r
        print(grid, json.dumps(r), flush=True)
        if not same:
            print(f"FAIL: {grid} samples differ between disk and mmap store")
            sys.exit(1)

    out = osp.join(HERE, "results", "e003_mmap_ab.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
