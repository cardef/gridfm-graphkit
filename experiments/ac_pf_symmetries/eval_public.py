"""Piano X, X3 references: GENCO's public transfer checkpoints (gridfm/genco-pf-transfer) in our pipeline, on the
fixed zero-shot sets (multigrid.grid_splits seed 0, 1024 scenarios): the pretrained model (100 subgrids, masked
pretraining; it ships no normalizer stats) with a normalizer fitted on each target, as in the paper's zero-shot
evaluation; the case118 fine-tuned models (100 and 1,000 samples) with their own stats on the case118 sets.

python -m experiments.ac_pf_symmetries.eval_public [--grids ...] [--out results/pianox_pt/public]
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from gridfm_graphkit.io.param_handler import get_task, load_normalizer

from .multigrid import grid_splits
from .run import OUT, ROOT, accuracy, default_device, eval_loader, make_args

T = ROOT / "data" / "genco_transfer"
D = 180 / math.pi


def build(args, normalizer, path):
    task = get_task(args, [normalizer])
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = task.load_state_dict({k.replace("model._orig_mod.", "model."): v for k, v in sd.items()},
                                               strict=False)
    lost = [k for k in missing if k.startswith("model.")]
    assert not lost, f"model weights missing from {path}: {lost[:5]}"
    return task.model.eval(), list(unexpected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grids", nargs="+", default=["case118_ieee", "case118_tr", "case500_goc", "case2000_goc",
                                                  "case57_ieee", "case30_ieee", "case14_ieee"])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--out", type=Path, default=OUT / "pianox_pt" / "public")
    p.add_argument("--device", default=default_device())
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    args = make_args(SimpleNamespace(epochs=1, batch_size=64, patience=1, layers=12, hidden=48))
    for g in a.grids:
        args.data.networks = [g]
        runs = [("pretrained", T / "pretraining" / "best_model_state_dict.pt", None)]
        if g.startswith("case118"):
            runs += [(f"finetune{n}", T / "finetune" / f"case118_{n}" / "best_model_state_dict.pt",
                      T / "finetune" / f"case118_{n}" / "normalizer_stats.pt") for n in (100, 1000)]
        for name, ckpt, stats in runs:
            if stats is None:
                h = grid_splits(g, args, 0, a.n)  # normalizer fitted on the target's seed-0 train part
            else:
                nz = load_normalizer(args)
                nz.fit_from_dict(torch.load(stats, map_location="cpu", weights_only=False)["case118_ieee"])
                h = grid_splits(g, args, 0, a.n, normalizer=nz)
            model, unexpected = build(args, h["normalizer"], ckpt)
            acc = accuracy(model.to(a.device), eval_loader(h["all"], 64), a.device)
            base = float(h["normalizer"].baseMVA)
            (a.out / f"{name}_{g}.json").write_text(json.dumps(
                {"grid": g, "model": name, "ckpt": str(ckpt), "baseMVA": base, "n": len(h["all"]), "acc": acc,
                 "unexpected_keys": unexpected[:10]}, indent=1))
            print(f"{g} {name}: VA {acc['VA'] * D:.3f} offset {acc['VA_cm'] * D:.3f} rest {acc['VA_diff'] * D:.3f} "
                  f"dVA {acc['dVA'] * D:.3f} VM {acc['VM']:.4f} VM_PQ {acc['VM_PQ']:.4f} PBE {acc['PBE'] * base:.3g} "
                  f"(baseMVA {base:.1f}, unexpected keys {len(unexpected)})", flush=True)


if __name__ == "__main__":
    main()
