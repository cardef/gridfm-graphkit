"""Train M0 / M0+Aug / M-canon PF surrogates on one grid, audit S1/S3/S4 equivariance, test zero-shot.

python -m experiments.ac_pf_symmetries.run --variant m0 aug canon --seeds 0 1 2
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_task,
    get_task_transforms,
    load_normalizer,
)
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)
from gridfm_graphkit.tasks.pf_task import (
    _build_bus_target,
    _clamp_known_to_ground_truth,
)
from gridfm_graphkit.training.callbacks import SaveBestModelStateDict

from .symmetries import (
    E,
    Canonicalize,
    RandomSymmetryAugment,
    act,
    act_output,
    equivariance_error,
    pred_masks,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
BASE_CONFIG = ROOT / "examples/config/HGNS_PF_datakit_case14.yaml"
PHASES = [0.1, 0.5, 1.0, math.pi]
SCALES = [0.01, 0.1, 0.5, 2.0, 10.0, 100.0]


def make_args(a):
    cfg = yaml.safe_load(open(BASE_CONFIG))
    cfg["data"]["workers"] = 0
    cfg["training"]["epochs"] = a.epochs
    cfg["training"]["batch_size"] = a.batch_size
    cfg["callbacks"]["patience"] = a.patience
    cfg["model"]["num_layers"] = a.layers
    cfg["model"]["hidden_size"] = a.hidden
    return NestedNamespace(**cfg)


def build_grid(case, args, seed, augment=None):
    """Random 80/10/10 scenario split; normalizer fitted on the train split (framework convention)."""
    root = ROOT / "data" / case
    normalizer = load_normalizer(args)
    base = get_task_transforms(args)
    ds = HeteroGridDatasetDisk(str(root), normalizer, transform=base)
    ids = list(range(len(ds)))
    random.Random(seed).shuffle(ids)
    n = len(ids) // 10
    test, val, train = ids[:n], ids[n : 2 * n], ids[2 * n :]
    normalizer.fit(str(root / "raw"), train)
    train_ds = ds
    if augment is not None:
        train_ds = HeteroGridDatasetDisk(
            str(root),
            normalizer,
            transform=Compose([base, augment]),
        )
    return {
        "train": Subset(train_ds, train),
        "val": Subset(ds, val),
        "test": Subset(ds, test),
        "normalizer": normalizer,
    }


def eval_loader(subset, batch_size):
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def accuracy(model, loader, device, sym=None, param=None):
    """RMSE per predicted quantity and mean power-balance residual, optionally on the transformed set T_g."""
    acc = {q: torch.zeros(2, dtype=torch.float64) for q in ("VM", "VA", "PG", "QG")}
    pbe = torch.zeros(2, dtype=torch.float64)
    for data in loader:
        data = data.to(device)
        n = data["bus"].x.size(0)
        target, gen_to_bus, _ = _build_bus_target(data, n)
        if sym is not None:
            data = act(data, sym, param)
            target = act_output({"bus": target}, sym, param, data)["bus"]
        out = model(data)
        for q, (col, m) in pred_masks(data).items():
            diff = (out["bus"][m, col] - target[m, col]).cpu().double()
            acc[q] += torch.stack([(diff**2).sum(), m.sum().cpu().double()])
        ev = _clamp_known_to_ground_truth(out["bus"], target, data, gen_to_bus, n)
        ei, attr = data[E].edge_index, data[E].edge_attr
        pft, qft = ComputeBranchFlow()(ev, ei, attr)
        p_in, q_in = ComputeNodeInjection()(pft, qft, ei, n)
        rp, rq = ComputeNodeResiduals()(p_in, q_in, ev, data["bus"].x)
        pbe += torch.tensor(
            [float(torch.sqrt(rp**2 + rq**2).sum()), n],
            dtype=torch.float64,
        )
    res = {q: math.sqrt(a[0] / a[1]) for q, a in acc.items()}
    # residual is reported in the *original* base so it is comparable across k
    res["PBE"] = float(pbe[0] / pbe[1]) * (param if sym == "scale" else 1.0)
    return res


def audit(model, loader, device):
    rows = []
    for sym, params in (("phase", PHASES), ("scale", SCALES), ("flip", [0.5])):
        for p in params:
            rows.append(
                {
                    "sym": sym,
                    "param": p,
                    "ee": equivariance_error(model, loader, sym, p, device),
                    "acc": accuracy(model, loader, device, sym, p),
                },
            )
    return rows


def run_one(variant, seed, a, device):
    L.seed_everything(seed, workers=True)
    args = make_args(a)
    augment = {
        "aug": RandomSymmetryAugment(),
        "augmild": RandomSymmetryAugment(alpha_max=0.5, k_range=(0.5, 2.0)),
    }.get(variant)
    grid = build_grid(a.case, args, seed, augment)
    train_loader = DataLoader(grid["train"], batch_size=a.batch_size, shuffle=True)
    val_loader = eval_loader(grid["val"], a.batch_size)

    task = get_task(args, [grid["normalizer"]])
    if variant == "canon":
        task.model = Canonicalize(task.model).fit_scale_ref(train_loader)

    run_dir = OUT / a.case / f"{variant}_seed{seed}"
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
    train_time = time.perf_counter() - t0
    task.load_state_dict(
        torch.load(run_dir / "model" / "best_model_state_dict.pt", map_location="cpu"),
    )
    model = task.model.to(device).eval()

    test_loader = eval_loader(grid["test"], a.batch_size)
    result = {
        "variant": variant,
        "seed": seed,
        "case": a.case,
        "epochs_run": trainer.current_epoch,
        "train_time_s": train_time,
        "baseMVA": float(grid["normalizer"].baseMVA),
        "scale_ref": float(model.scale_ref) if variant == "canon" else None,
        "in_dist": accuracy(model, test_loader, device),
        "audit": audit(model, test_loader, device),
        "zero_shot": {},
    }
    for case in a.zero_shot:
        g = build_grid(case, args, seed)
        loader = eval_loader(g["test"], a.batch_size)
        result["zero_shot"][case] = {
            "baseMVA": float(g["normalizer"].baseMVA),
            "acc": accuracy(model, loader, device),
            "audit": audit(model, loader, device),
        }
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {k: result[k] for k in ("variant", "seed", "epochs_run", "in_dist")},
        ),
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        nargs="+",
        default=["m0", "aug", "augmild", "canon"],
        choices=["m0", "aug", "augmild", "canon"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--case", default="case14_ieee")
    p.add_argument("--zero-shot", nargs="*", default=["case30_ieee", "case57_ieee"])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    a = p.parse_args()
    for seed in a.seeds:
        for variant in a.variant:
            if (OUT / a.case / f"{variant}_seed{seed}" / "result.json").exists():
                print(f"skip {variant} seed {seed}: result.json exists")
                continue
            run_one(variant, seed, a, a.device)


if __name__ == "__main__":
    main()
