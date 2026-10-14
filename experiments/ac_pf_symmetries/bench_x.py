"""Piano X, X2: inference time of GENCO (plain), the injection decoder (dcinjcanon) and the guarded closed loop
(dcloopgcanon), per forward and per scenario, on case118_ieee, case500_goc and case2000_goc, batch 1 and 64, on the
current device. Untrained weights: the time does not depend on them. 3 warm-up forwards, then the mean of --reps.

python -m experiments.ac_pf_symmetries.bench_x --out results/pianox/bench_<gpu>.json
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.loader import DataLoader

from gridfm_graphkit.io.param_handler import get_task

from .dcprior import DCInjectionLayers, DCLoopLayers
from .multigrid import grid_splits
from .run import make_args
from .symmetries import Canonicalize

WRAP = {
    "plain": lambda m, loader: m,
    "dcinjcanon": lambda m, loader: Canonicalize(DCInjectionLayers(m)).fit_scale_ref(loader),
    "dcloopgcanon": lambda m, loader: Canonicalize(DCLoopLayers(m, guard=True)).fit_scale_ref(loader),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grids", nargs="+", default=["case118_ieee", "case500_goc", "case2000_goc"])
    ap.add_argument("--batch", nargs="+", type=int, default=[1, 64])
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    dev = torch.device("cuda")
    args = make_args(SimpleNamespace(epochs=1, batch_size=64, patience=1, layers=12, hidden=48))
    res = {"device": torch.cuda.get_device_name(), "rows": []}
    for case in a.grids:
        args.data.networks = [case]
        s = grid_splits(case, args, 0, 128)
        fit_loader = DataLoader(s["all"], batch_size=64)
        for arm, wrap in WRAP.items():
            torch.manual_seed(0)
            model = wrap(get_task(args, [s["normalizer"]]).model, fit_loader).to(dev).eval()
            for bs in a.batch:
                batch = next(iter(DataLoader(s["all"], batch_size=bs))).to(dev)
                with torch.no_grad():
                    for _ in range(3):
                        model(batch)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(a.reps):
                        model(batch)
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / a.reps * 1e3
                row = {"grid": case, "arm": arm, "batch": bs, "ms_forward": ms, "ms_scenario": ms / bs,
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
                torch.cuda.reset_peak_memory_stats()
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
            del model
            torch.cuda.empty_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
