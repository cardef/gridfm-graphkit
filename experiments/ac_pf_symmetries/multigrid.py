"""R2 (Piano A §7.5): train on several grids at once (mixed batches), test zero-shot on held-out grids.

python -m experiments.ac_pf_symmetries.multigrid --arm canon m1fullcanon \
    --train case14_ieee case57_ieee case118_ieee --heldout case30_ieee --seeds 0 1 2 --results-dir results/abacus_r2

Both arms are canonicalized (S1+S3+S4 exact), so the held-out grid's normalizer cannot matter (E5b): it gets the
first training grid's. Each training grid keeps its own normalizer, fitted on its train split (framework
convention), and contributes the same number of scenarios, a fixed random subset (seed 0) split 80/10/10 by the
run seed; batches mix grids.
"""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, Compose

from gridfm_graphkit.datasets.globals import VM_H
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import get_task, get_task_transforms, load_normalizer
from gridfm_graphkit.training.callbacks import SaveBestModelStateDict

from .dcprior import DCInjectionLayers, DCPriorLayers
from .hodge import BranchAngleLayers
from .resnorm import BusResidualScale
from .run import OUT, ROOT, accuracy, default_device, eval_loader, make_args
from .symmetries import Canonicalize

ARMS = {
    "canon": None,
    "m1fullcanon": BranchAngleLayers,
    "m3canon": None,
    "canonresnorm": BusResidualScale,
    "dccanon": DCPriorLayers,
    "dczcanon": partial(DCPriorLayers, zero_init=True),
    "dcinjcanon": DCInjectionLayers,
    "canonvref": None,
    "dcinjvref": DCInjectionLayers,
    "dcinj1hop": DCInjectionLayers,
}
LOCAL = {"m3canon"}  # Piano A M3: the S3 frame per bus (local dimensionless inputs) instead of per graph
VREF = {"canonvref", "dcinjvref"}  # the task with the REF voltage given (UnmaskRefVoltage)
LAYERS = {"dcinj1hop": 1}  # GNS depth overrides: one layer = one hop and no physics feedback


class UnmaskRefVoltage(BaseTransform):
    """Give the model the REF bus's voltage magnitude, an operator setpoint in practice, which the framework's PF mask
    hides (with P and Q at REF unknown too, the PF is then one degree of freedom short): restore it from the labels and
    mark it known, so the model clamps it like the PV setpoints."""

    def forward(self, data):
        ref = data.mask_dict["REF"]
        data["bus"].x[ref, VM_H] = data["bus"].y[ref, VM_H]
        data.mask_dict["bus"][ref, VM_H] = False
        return data


def result_name(a):
    """result.json for a training run, result_eval.json for a re-evaluation, result_extrap_<grids>.json for a
    re-evaluation on other grids (size extrapolation, P47-P50)."""
    if a.extrap:
        return f"result_extrap_{'_'.join(a.extrap)}.json"
    return "result_eval.json" if a.eval_only else "result.json"


def grid_splits(case, args, seed, n, normalizer=None, vref=False):
    """n scenarios of `case` (a fixed random subset, seed 0), split 80/10/10 by the run seed; the normalizer is
    fitted on the train part unless one is given."""
    root = ROOT / "data" / case
    fit = normalizer is None
    if fit:
        normalizer = load_normalizer(args)
    transform = get_task_transforms(args)
    if vref:
        transform = Compose([transform, UnmaskRefVoltage()])
    ds = HeteroGridDatasetDisk(str(root), normalizer, transform=transform)
    ids = [int(i) for i in np.random.default_rng(0).permutation(len(ds))[:n]]
    keep = list(ids)
    random.Random(seed).shuffle(ids)
    k = len(ids) // 10
    test, val, train = ids[:k], ids[k : 2 * k], ids[2 * k :]
    if fit:
        normalizer.fit(str(root / "raw"), train)
    return {
        "train": Subset(ds, train),
        "val": Subset(ds, val),
        "test": Subset(ds, test),
        "all": Subset(ds, keep),
        "normalizer": normalizer,
    }


def run_one(arm, seed, a, device):
    L.seed_everything(seed, workers=True)
    args = make_args(a)
    if arm in LAYERS:
        args.model.num_layers = LAYERS[arm]
    args.data.networks = list(a.train)  # one normalizer per training grid (the task saves their stats by name)
    vref = arm in VREF
    grids = {c: grid_splits(c, args, seed, a.n_per_grid, vref=vref) for c in a.train}
    train_loader = DataLoader(
        ConcatDataset([g["train"] for g in grids.values()]),
        batch_size=a.batch_size,
        shuffle=True,
    )
    val_loader = eval_loader(ConcatDataset([g["val"] for g in grids.values()]), a.batch_size)
    task = get_task(args, [g["normalizer"] for g in grids.values()])
    inner = ARMS[arm](task.model) if ARMS[arm] else task.model
    if isinstance(inner, BusResidualScale):
        inner.fit_d_ref(train_loader)
    task.model = Canonicalize(inner, local=arm in LOCAL).fit_scale_ref(train_loader)

    run_dir = a.results_dir / f"heldout_{'_'.join(a.heldout)}" / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs_run = train_time = None
    if not a.eval_only:
        epochs_run, train_time = train(task, train_loader, val_loader, run_dir, a, device)
    task.load_state_dict(
        torch.load(run_dir / "model" / "best_model_state_dict.pt", map_location="cpu"),
    )
    model = task.model.to(device).eval()
    return evaluate(model, arm, seed, a, args, grids, run_dir, epochs_run, train_time, device, vref)


def train(task, train_loader, val_loader, run_dir, a, device):
    trainer = L.Trainer(
        max_epochs=a.epochs,
        accelerator=device,
        devices=1,
        logger=CSVLogger(save_dir=str(run_dir), name="logs"),
        callbacks=[
            EarlyStopping("Validation loss", patience=a.patience, mode="min"),
            SaveBestModelStateDict("Validation loss"),
        ],
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    t0 = time.perf_counter()
    trainer.fit(task, train_loader, val_loader)
    return trainer.current_epoch, time.perf_counter() - t0


def evaluate(model, arm, seed, a, args, grids, run_dir, epochs_run, train_time, device, vref):
    """In-distribution and zero-shot accuracy of a trained model; --eval-only writes result_eval.json, never result.json;
    with --extrap the zero-shot grids are those instead of the held-out one and in-distribution is skipped."""
    source = next(iter(grids.values()))["normalizer"]
    result = {
        "arm": arm,
        "seed": seed,
        "train": list(a.train),
        "heldout": list(a.heldout),
        "extrap": list(a.extrap or []),
        "n_per_grid": a.n_per_grid,
        "epochs_run": epochs_run,
        "train_time_s": train_time,
        "scale_ref": float(model.scale_ref),
        "device_name": torch.cuda.get_device_name() if device == "cuda" else device,
        "baseMVA": {c: float(g["normalizer"].baseMVA) for c, g in grids.items()},
        "in_dist": {} if a.extrap else {c: accuracy(model, eval_loader(g["test"], a.batch_size), device) for c, g in grids.items()},
        "zero_shot": {},
    }
    for c in a.extrap or a.heldout:
        h = grid_splits(c, args, seed, a.n_per_grid, normalizer=source, vref=vref)
        result["zero_shot"][c] = {
            "baseMVA": float(source.baseMVA),
            "acc": accuracy(model, eval_loader(h["all"], a.batch_size), device),
        }
    (run_dir / result_name(a)).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("arm", "seed", "epochs_run", "zero_shot")}))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--heldout", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-per-grid", type=int, default=1024)
    p.add_argument("--results-dir", type=Path, default=OUT / "abacus_r2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--device", default=default_device())
    p.add_argument("--eval-only", action="store_true", help="re-evaluate saved models with the current code")
    p.add_argument("--extrap", nargs="+", default=None,
                   help="with --eval-only: evaluate the saved models zero-shot on these grids instead of the held-out "
                        "one (size extrapolation, P47-P50); writes result_extrap_<grids>.json")
    a = p.parse_args()
    if a.extrap and not a.eval_only:
        p.error("--extrap requires --eval-only")
    for seed in a.seeds:
        for arm in a.arm:
            out = a.results_dir / f"heldout_{'_'.join(a.heldout)}" / f"{arm}_seed{seed}" / result_name(a)
            if out.exists():
                print(f"skip {arm} seed {seed}: {out} exists")
                continue
            run_one(arm, seed, a, a.device)


if __name__ == "__main__":
    main()
