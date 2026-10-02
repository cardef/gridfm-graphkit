"""Copy a subset of a datakit PF dataset's scenarios into a new raw folder, renumbered to 0..N-1.

Graphkit's preprocessing requires contiguous scenario ids, so the subset is renumbered by rank in the
sorted id list. Every source file is read on its own, every scenario must come from exactly one file, and
rows are re-ordered only by a stable sort on the new id, so each scenario keeps its source row order.
Output: <out>/raw/{bus,gen,branch}_data.parquet/scenario_partition=<new_id // 200>/part-0.parquet,
n_scenarios.txt, id_map.json (new id -> source id). Same procedure as HierarchicalFM W28's reduced copy.

random: N scenarios drawn uniformly (training data of the case14 campaign)
  python -m experiments.ac_pf_symmetries.subset_data random --src <raw> --out <dir> --n 2048 --seed 0
official: the scenarios of the released GENCO runs' splits (E1/E2/E5): every seed's full test split, the
first --n-train / --n-val ids of its train / val list; writes splits/<size>_seed<s>.json in the new ids
  python -m experiments.ac_pf_symmetries.subset_data official --src <raw> --runs <mlflow/eval/case> --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TABLES = ("bus_data", "gen_data", "branch_data")
PER_PARTITION = 200


def table_files(raw, table):
    p = raw / f"{table}.parquet"
    if p.is_file():
        return [p]
    parts = sorted(
        p.glob("scenario_partition=*"),
        key=lambda q: int(q.name.split("=")[1]),
    )
    return [f for part in parts for f in sorted(part.glob("*.parquet"))]


def source_ids(raw):
    files = table_files(raw, "bus_data")
    return np.unique(
        np.concatenate(
            [pq.read_table(f, columns=["scenario"]).column(0).to_numpy() for f in files],
        ),
    )


def copy_table(raw, table, ids):
    """All rows of the scenarios in `ids` (sorted, unique), renumbered to their rank in `ids`."""
    pieces, owner = [], {}
    for f in table_files(raw, table):
        t = pq.read_table(f)
        if "scenario_partition" in t.column_names:
            t = t.drop(["scenario_partition"])
        sc = t.column("scenario").to_numpy()
        keep = np.isin(sc, ids)
        if not keep.any():
            continue
        for s in np.unique(sc[keep]):
            assert owner.setdefault(int(s), str(f)) == str(f), (
                f"{table}: scenario {s} spans two files"
            )
        pieces.append(t.filter(pa.array(keep)))
    t = pa.concat_tables(pieces)
    old = t.column("scenario").to_numpy()
    new = np.searchsorted(ids, old)
    assert (ids[new] == old).all()
    order = np.argsort(new, kind="stable")
    t = t.take(pa.array(order))
    col = t.schema.get_field_index("scenario")
    return t.set_column(col, "scenario", pa.array(new[order], type=pa.int64()))


def write_subset(raw, ids, out):
    """Write the renumbered subset to out/raw; return {source id: new id}."""
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    dst = out / "raw"
    dst.mkdir(parents=True, exist_ok=False)
    for table in TABLES:
        t = copy_table(raw, table, ids)
        new = t.column("scenario").to_numpy()
        assert np.array_equal(np.unique(new), np.arange(len(ids))), (
            f"{table}: scenarios missing from the source"
        )
        if table == "bus_data":
            starts = np.flatnonzero(np.r_[True, new[1:] != new[:-1]])
            sizes = np.diff(np.r_[starts, len(new)])
            assert (sizes == sizes[0]).all(), "bus count varies across scenarios"
            bus = t.column("bus").to_numpy().reshape(-1, sizes[0])
            assert (bus == np.arange(sizes[0])).all(), "bus rows not 0..n-1"
        bounds = np.searchsorted(
            new,
            np.arange(0, len(ids) + PER_PARTITION, PER_PARTITION),
        )
        for q in range(len(bounds) - 1):
            lo, hi = int(bounds[q]), int(bounds[q + 1])
            if lo < hi:
                d = dst / f"{table}.parquet" / f"scenario_partition={q}"
                d.mkdir(parents=True)
                pq.write_table(t.slice(lo, hi - lo), d / "part-0.parquet")
        print(f"{table}: {t.num_rows} rows, {len(ids)} scenarios", flush=True)
    (dst / "n_scenarios.txt").write_text(str(len(ids)))
    (out / "id_map.json").write_text(
        json.dumps({"source": str(raw), "new_to_source": ids.tolist()}),
    )
    return {int(o): n for n, o in enumerate(ids)}


def official_splits(runs):
    """{(size, seed): {train, val, test}} from <runs>/<size>/seed<s>/<run id>/artifacts/stats/*_splits.json."""
    out = {}
    for f in sorted(runs.glob("*/seed*/*/artifacts/stats/*_scenario_splits.json")):
        size, seed = f.parts[-6], int(f.parts[-5].removeprefix("seed"))
        assert (size, seed) not in out, f"two split files for {size} seed {seed}"
        sp = json.loads(f.read_text())
        out[size, seed] = {k: [int(i) for i in sp[k]] for k in ("train", "val", "test")}
    assert out, f"no split files under {runs}"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["random", "official"])
    p.add_argument("--src", required=True, type=Path, help="source raw folder")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", type=int, help="random: number of scenarios")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", type=Path, help="official: mlflow/eval/<case> of the release")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-val", type=int, default=500)
    a = p.parse_args()

    if a.mode == "random":
        pool = source_ids(a.src)
        ids = np.random.default_rng(a.seed).choice(pool, size=a.n, replace=False)
        write_subset(a.src, ids, a.out)
        print(f"random {a.n} of {len(pool)} scenarios (seed {a.seed})")
        return

    splits = official_splits(a.runs)
    keep = {
        key: {
            "train": sp["train"][: a.n_train],
            "val": sp["val"][: a.n_val],
            "test": sp["test"],
        }
        for key, sp in splits.items()
    }
    union = sorted(set().union(*[set(v) for sp in keep.values() for v in sp.values()]))
    new_of = write_subset(a.src, union, a.out)
    d = a.out / "splits"
    d.mkdir()
    for (size, seed), sp in keep.items():
        renum = {k: [new_of[i] for i in v] for k, v in sp.items()}
        renum["official_sizes"] = {k: len(v) for k, v in splits[size, seed].items()}
        (d / f"{size}_seed{seed}.json").write_text(json.dumps(renum))
    same = {
        seed: len({json.dumps(sp) for (_, s), sp in splits.items() if s == seed})
        for seed in sorted({s for _, s in splits})
    }
    print(
        f"official: {len(union)} scenarios kept, {len(splits)} runs; "
        f"distinct splits per seed {same}",
    )


if __name__ == "__main__":
    main()
