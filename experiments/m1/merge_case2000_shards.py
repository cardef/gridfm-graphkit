"""Merge the 4 seed-sharded case2000_goc datakit outputs into data/case2000_goc/raw.

Provenance: cluster-scale R001 for case2000_goc was generated as 4 parallel
shards (seeds 0-3, 10240 requested scenarios each = the plan's 40960 total,
distribution unchanged) because single-node generation at 16 procs measured
~14 s/scenario -> ~7 days, past any sane --time. Shards are mergeable
because all perturbations are off: Y is identical across every scenario.

Renumbering: 'scenario' and 'load_scenario_idx' are offset per shard by the
cumulative (max+1) of prior shards so both stay globally unique; the loader
requires contiguous scenario ids [0, N) and split_by_load_scenario_idx must
not conflate different-seed load draws that happen to share an index.

Memory: processes one table at a time (not all 5 tables x 4 shards at
once) -- first attempt OOM'd at 32G. branch_data is the largest table
(~1.1G/shard compressed); freed between tables via `del` + gc.collect()
so peak RSS is ~1 table's worth, not all 5.
"""

import gc
import glob
import os
import os.path as osp
import shutil
import sys

import pandas as pd

BASE = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
SHARD_ROOT = osp.join(BASE, "data", "case2000_shards")
OUT_RAW = osp.join(BASE, "data", "case2000_goc", "raw")
TABLES = ["bus_data", "gen_data", "branch_data", "y_bus_data", "runtime_data"]

shard_raws = sorted(glob.glob(osp.join(SHARD_ROOT, "s*", "case2000_goc", "raw")))
if len(shard_raws) != 4:
    sys.exit(f"expected 4 shard raw dirs, found {len(shard_raws)}: {shard_raws}")

os.makedirs(OUT_RAW, exist_ok=True)

# Pass 1 (bus_data only): compute per-shard scenario/load offsets.
scen_offsets = []
load_offsets = []
scen_off = 0
load_off = 0
for raw in shard_raws:
    bus = pd.read_parquet(osp.join(raw, "bus_data.parquet"), columns=["scenario"])
    n_scen = int(bus["scenario"].max()) + 1
    assert bus["scenario"].nunique() == n_scen, f"{raw}: non-contiguous scenarios"
    scen_offsets.append(scen_off)
    scen_off += n_scen

    if osp.exists(osp.join(raw, "runtime_data.parquet")):
        pass  # load_scenario_idx lives on bus_data; re-read below with the column
    del bus
    print(f"{raw}: {n_scen} scenarios -> cumulative {scen_off}")

# load_scenario_idx offsets need the actual column (bus_data has it if datakit recorded it).
bus0 = pd.read_parquet(osp.join(shard_raws[0], "bus_data.parquet"), columns=["scenario"])
has_load_idx = False
for raw in shard_raws:
    cols = pd.read_parquet(osp.join(raw, "bus_data.parquet")).columns
    if "load_scenario_idx" in cols:
        has_load_idx = True
    break
del bus0

if has_load_idx:
    load_off = 0
    for raw in shard_raws:
        bus = pd.read_parquet(
            osp.join(raw, "bus_data.parquet"), columns=["load_scenario_idx"],
        )
        load_offsets.append(load_off)
        load_off += int(bus["load_scenario_idx"].max()) + 1
        del bus
else:
    load_offsets = [0, 0, 0, 0]

# Pass 2: one table at a time, offset + concat + write immediately, then free.
for t in TABLES:
    parts = []
    for raw, so, lo in zip(shard_raws, scen_offsets, load_offsets):
        p = osp.join(raw, f"{t}.parquet")
        if not osp.exists(p):
            sys.exit(f"missing {p}")
        df = pd.read_parquet(p)
        if "scenario" in df.columns:
            df["scenario"] += so
        if "load_scenario_idx" in df.columns:
            df["load_scenario_idx"] += lo
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    del parts
    merged.to_parquet(osp.join(OUT_RAW, f"{t}.parquet"))
    n_rows = len(merged)
    del merged
    gc.collect()
    print(f"{t}: {n_rows} rows written")

# Validation: the same invariants the dataset loader asserts.
bus = pd.read_parquet(osp.join(OUT_RAW, "bus_data.parquet"), columns=["scenario"])
n = bus["scenario"].nunique()
assert bus["scenario"].min() == 0 and bus["scenario"].max() == n - 1, "non-contiguous"
counts = bus.groupby("scenario").size()
assert counts.nunique() == 1, "inconsistent bus count across scenarios"
print(f"MERGE OK: {n} scenarios, {counts.iloc[0]} buses each -> {OUT_RAW}")

for aux in ["args.log", "scenarios_agg_load_profile.parquet"]:
    src = osp.join(shard_raws[0], aux)
    if osp.exists(src):
        shutil.copy(src, osp.join(OUT_RAW, f"shard0_{aux}"))
