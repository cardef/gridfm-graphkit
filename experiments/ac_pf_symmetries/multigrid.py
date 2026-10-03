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
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import get_task, get_task_transforms, load_normalizer
from gridfm_graphkit.training.callbacks import SaveBestModelStateDict

from .hodge import BranchAngleLayers
from .run import OUT, ROOT, accuracy, default_device, eval_loader, make_args
from .symmetries import Canonicalize

ARMS = {"canon": None, "m1fullcanon": BranchAngleLayers}


def grid_splits(case, args, seed, n, normalizer=None):
    """n scenarios of `case` (a fixed random subset, seed 0), split 80/10/10 by the run seed; the normalizer is
    fitted on the train part unless one is given."""
    root = ROOT / "data" / case
    fit = normalizer is None
    if fit:
        normalizer = load_normalizer(args)
    ds = HeteroGridDatasetDisk(str(root), normalizer, transform=get_task_transforms(args))
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
    args.data.networks = list(a.train)  # one normalizer per training grid (the task saves their stats by name)
    grids = {c: grid_splits(c, args, seed, a.n_per_grid) for c in a.train}
    train_loader = DataLoader(
        ConcatDataset([g["train"] for g in grids.values()]),
        batch_size=a.batch_size,
        shuffle=True,
    )
    val_loader = eval_loader(ConcatDataset([g["val"] for g in grids.values()]), a.batch_size)
    task = get_task(args, [g["normalizer"] for g in grids.values()])
    inner = ARMS[arm](task.model) if ARMS[arm] else task.model
    task.model = Canonicalize(inner).fit_scale_ref(train_loader)

    run_dir = a.results_dir / f"heldout_{'_'.join(a.heldout)}" / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
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
    epochs_run, train_time = trainer.current_epoch, time.perf_counter() - t0
    task.load_state_dict(
        torch.load(run_dir / "model" / "best_model_state_dict.pt", map_location="cpu"),
    )
    model = task.model.to(device).eval()

    source = next(iter(grids.values()))["normalizer"]
    result = {
        "arm": arm,
        "seed": seed,
        "train": list(a.train),
        "heldout": list(a.heldout),
        "n_per_grid": a.n_per_grid,
        "epochs_run": epochs_run,
        "train_time_s": train_time,
        "scale_ref": float(model.scale_ref),
        "device_name": torch.cuda.get_device_name() if device == "cuda" else device,
        "baseMVA": {c: float(g["normalizer"].baseMVA) for c, g in grids.items()},
        "in_dist": {c: accuracy(model, eval_loader(g["test"], a.batch_size), device) for c, g in grids.items()},
        "zero_shot": {},
    }
    for c in a.heldout:
        h = grid_splits(c, args, seed, a.n_per_grid, normalizer=source)
        result["zero_shot"][c] = {
            "baseMVA": float(source.baseMVA),
            "acc": accuracy(model, eval_loader(h["all"], a.batch_size), device),
        }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
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
    a = p.parse_args()
    for seed in a.seeds:
        for arm in a.arm:
            out = a.results_dir / f"heldout_{'_'.join(a.heldout)}" / f"{arm}_seed{seed}" / "result.json"
            if out.exists():
                print(f"skip {arm} seed {seed}: {out} exists")
                continue
            run_one(arm, seed, a, a.device)


if __name__ == "__main__":
    main()
