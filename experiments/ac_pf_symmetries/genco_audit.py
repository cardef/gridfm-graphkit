"""E1/E2/E5: audit of a *released* GENCO PF checkpoint, optionally wrapped post hoc in Canonicalize.

python -m experiments.ac_pf_symmetries.genco_audit \
    --config <ckpt>/case14_ieee_base_seed0.yaml --data-path data/genco \
    --model-path <ckpt>/best_model_state_dict.pt --normalizer-stats <ckpt>/normalizer_stats.pt \
    [--canon] [--n-test 2000] [--no-audit] [--splits-json <official splits json>] \
    [--target case57_ieee --target-stats source|refit] --out results/genco/case14_ieee_base_seed0.json

Data, split, normalizer stats and weights go through the framework's own path (same as `gridfm_graphkit
evaluate`), so the audited function is the released one. Audit and accuracy code are run.py's.
All accuracy numbers are in the *used* normalizer's per-unit (multiply PG/QG/PBE by `baseMVA` to compare
runs that use different normalizers).
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import yaml
from torch.utils.data import Subset

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.config_version import upgrade_config
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task

from .run import accuracy, audit, default_device, eval_loader
from .symmetries import Canonicalize, equivariance_error


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--normalizer-stats", required=True)
    p.add_argument(
        "--canon",
        action="store_true",
        help="wrap in Canonicalize (scale_ref fitted on --n-fit train graphs)",
    )
    p.add_argument(
        "--scale-ref",
        type=float,
        help="canon: constant from the source run's json (required with --target)",
    )
    p.add_argument("--n-test", type=int, default=2000, help="0 = full test split")
    p.add_argument("--n-fit", type=int, default=2000)
    p.add_argument(
        "--no-audit",
        action="store_true",
        help="accuracy only (gate G1 on the full test split)",
    )
    p.add_argument(
        "--splits-json",
        help="official {train,val,test} id lists (mlflow *_scenario_splits.json); full data only",
    )
    p.add_argument(
        "--target",
        help="evaluate the source model (config's network) on this network",
    )
    p.add_argument(
        "--target-stats",
        choices=["source", "refit"],
        default="source",
        help="source: keep the source grid's normalizer (fixed convention); refit: framework default, fitted on the target",
    )
    p.add_argument(
        "--scale-probe",
        type=float,
        nargs="*",
        default=[],
        help="extra S3 probes k; k = baseMVA_refit / baseMVA_source is exactly the refit normalizer's action",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=default_device())
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.canon and a.target and a.scale_ref is None:
        p.error(
            "--canon with --target needs the source grid's --scale-ref (fitting it on the target would cancel the point)",
        )

    cfg = upgrade_config(yaml.safe_load(open(a.config)))  # released YAMLs are v0
    cfg["data"]["workers"] = 0
    assert len(cfg["data"]["networks"]) == 1, "one network per audit"
    src = cfg["data"]["networks"][0]
    stats_path = a.normalizer_stats
    if a.target:
        cfg["data"]["networks"] = [a.target]
        if a.target_stats == "source":
            stats = torch.load(
                a.normalizer_stats,
                map_location="cpu",
                weights_only=True,
            )
            stats_path = str(Path(tempfile.mkdtemp()) / "normalizer_stats.pt")
            torch.save({a.target: stats[src]}, stats_path)
        else:
            stats_path = None
    net = cfg["data"]["networks"][0]
    ids = None
    if a.splits_json:
        ids = json.load(open(a.splits_json))
        folder = Path(tempfile.mkdtemp())
        for k in ("train", "val", "test"):
            torch.save(torch.tensor(ids[k]), folder / f"{k}.pt")
        cfg["data"]["split_from_existing_files"] = str(folder)
        cfg["data"]["split_by_load_scenario_idx"] = (
            False  # mutually exclusive in the datamodule
        )
    args = NestedNamespace(**cfg)

    dm = LitGridHeteroDataModule(args, a.data_path, normalizer_stats_path=stats_path)
    dm.setup("fit")
    if stats_path is not None:
        # the datamodule silently refits (warning only) if the network is missing from the stats file: fail loud
        saved = torch.load(stats_path, map_location="cpu", weights_only=True)[net]
        assert float(saved["baseMVA"]) == dm.data_normalizers[0].baseMVA, (
            "normalizer stats were not applied"
        )
    if ids is not None:
        assert len(dm.test_datasets[0]) == len(ids["test"]), (
            "official test split not reproduced"
        )
    task = get_task(args, dm.data_normalizers)
    sd = torch.load(a.model_path, map_location="cpu")
    # legacy torch.compile prefix; same remap as gridfm_graphkit.cli (not importable here: needs gridfm_datakit)
    task.load_state_dict(
        {k.replace("model._orig_mod.", "model."): v for k, v in sd.items()},
    )
    model = task.model
    if a.canon:
        model = Canonicalize(model)
        if a.scale_ref is not None:
            model.scale_ref.fill_(a.scale_ref)
        else:
            fit = Subset(
                dm.train_datasets[0],
                range(min(a.n_fit, len(dm.train_datasets[0]))),
            )
            model.fit_scale_ref(eval_loader(fit, a.batch_size))
    model = model.to(a.device).eval()

    test = dm.test_datasets[0]
    if a.n_test:
        test = Subset(test, range(min(a.n_test, len(test))))
    loader = eval_loader(test, a.batch_size)
    result = {
        "config": a.config,
        "model_path": a.model_path,
        "source": src,
        "network": net,
        "target_stats": a.target_stats if a.target else None,
        "official_split": bool(ids),
        "canon": a.canon,
        "scale_ref": float(model.scale_ref) if a.canon else None,
        "baseMVA": float(dm.data_normalizers[0].baseMVA),
        "n_test": len(test),
        "in_dist": accuracy(model, loader, a.device),
        "audit": None if a.no_audit else audit(model, loader, a.device),
    }
    result["scale_probes"] = [
        {
            "param": k,
            "ee": equivariance_error(model, loader, "scale", k, a.device),
            "acc": accuracy(model, loader, a.device, "scale", k),
        }
        for k in a.scale_probe
    ]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("network", "canon", "baseMVA", "n_test", "in_dist")
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
