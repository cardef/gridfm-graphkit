"""E0p-dup: exact duplicate scenarios in a public PF dataset, and test graphs whose exact twin is in the same
seed's train split (leakage) under the released runs' official splits.

python -m experiments.ac_pf_symmetries.split_audit --case case14_ieee [--root data/genco/full]
    [--runs data/genco_ckpt/mlflow/eval] --out results/genco/split_audit_case14_ieee.json

A scenario's key hashes every bus (Pd, Qd, Vm, Va), branch (status, r, x) and generator (status, p) value,
rounded to 1e-8, in a fixed row order; datakit writes every element of every scenario, so rows align.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def table(raw, name, cols, order):
    df = pd.read_parquet(raw / f"{name}.parquet", columns=["scenario", *order, *cols])
    df = df.sort_values(["scenario", *order], kind="stable")
    n = df.scenario.nunique()
    ids = df.scenario.to_numpy().reshape(n, -1)[:, 0]
    return ids, np.round(df[cols].to_numpy(dtype=float), 8).reshape(n, -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--root", type=Path, default=HERE.parents[1] / "data/genco/full")
    p.add_argument("--runs", type=Path, default=HERE.parents[1] / "data/genco_ckpt/mlflow/eval")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raw = a.root / a.case / "raw"
    sb, bus = table(raw, "bus_data", ["Pd", "Qd", "Vm", "Va"], ["bus"])
    sr, br = table(raw, "branch_data", ["br_status", "r", "x"], ["idx"])
    sg, gen = table(raw, "gen_data", ["in_service", "p_mw"], ["idx"])
    assert (sb == sr).all() and (sb == sg).all(), "tables disagree on the scenario set"
    key = pd.util.hash_pandas_object(
        pd.DataFrame(np.concatenate([bus, br, gen], 1)),
        index=False,
    ).to_numpy()
    lsi = (
        pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "load_scenario_idx"])
        .groupby("scenario")
        .load_scenario_idx.first()
        .reindex(sb)
        .to_numpy()
    )
    groups = pd.DataFrame({"k": key, "l": lsi}).groupby("k")
    res = {
        "scenarios": int(len(sb)),
        "distinct": int(len(np.unique(key))),
        "duplicate_groups_spanning_load_scenarios": int((groups.l.nunique() > 1).sum()),
        "largest_group": int(groups.size().max()),
        "splits": {},
    }
    pos = pd.Series(np.arange(len(sb)), index=sb)
    for f in sorted((a.runs / a.case).glob("*/seed*/*/artifacts/stats/*_scenario_splits.json")):
        size, seed = f.parts[-6], f.parts[-5]
        sp = json.loads(f.read_text())
        k = {s: key[pos[sp[s]].to_numpy()] for s in ("train", "val", "test")}
        twins = np.isin(k["test"], k["train"])
        res["splits"][f"{size}_{seed}"] = {
            "test": len(k["test"]),
            "test_distinct": int(len(np.unique(k["test"]))),
            "test_with_train_twin": int(twins.sum()),
            "load_scenarios_shared_train_test": len(
                set(lsi[pos[sp["train"]].to_numpy()]) & set(lsi[pos[sp["test"]].to_numpy()]),
            ),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
