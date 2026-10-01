"""E0: which symmetries does a datakit PF dataset let us test? (REF angle, phase shifters, admittance scale.)

python -m experiments.ac_pf_symmetries.data_audit --root data [--cases case14_ieee ...] [--max-partitions 20]

`root/<case>/raw/{bus,branch}_data.parquet` may be a flat file (local datakit) or a Hive directory of
`scenario_partition=*` (public HF datasets); for the latter only the first `--max-partitions` are read.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"


def read(raw, table, max_partitions):
    p = raw / f"{table}.parquet"
    if p.is_file():
        return pd.read_parquet(p)
    parts = sorted(
        p.glob("scenario_partition=*"),
        key=lambda q: int(q.name.split("=")[1]),
    )
    return pd.concat([pd.read_parquet(q) for q in parts[:max_partitions]])


def audit_case(raw, max_partitions):
    bus, br = (
        read(raw, "bus_data", max_partitions),
        read(raw, "branch_data", max_partitions),
    )
    ref = bus[bus.REF == 1]
    on = br[br.br_status == 1]
    per_graph_ref = ref.groupby("scenario").size()
    y = np.hypot(on.Yff_r, on.Yff_i).groupby(on.scenario).mean()
    return {
        "scenarios": int(bus.scenario.nunique()),
        "ref_per_graph_unique": sorted(set(per_graph_ref.tolist())),
        "ref_va_deg": [float(ref.Va.min()), float(ref.Va.max())],
        "shift_nonzero_branches": int((on["shift"].abs() > 1e-12).sum()),
        "shift_abs_max_deg": float(on["shift"].abs().max()),
        "tap_ne_1_branches": int(((on.tap - 1).abs() > 1e-12).sum()),
        "branches_out_of_service": int((br.br_status != 1).sum()),
        "mean_abs_Yff": float(y.mean()),
        "mean_abs_Yff_cv_across_scenarios": float(y.std() / y.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--cases", nargs="*")
    p.add_argument("--max-partitions", type=int, default=20)
    p.add_argument("--out", default=str(OUT / "data_audit.json"))
    a = p.parse_args()
    root = Path(a.root)
    cases = a.cases or sorted(d.name for d in root.iterdir() if (d / "raw").is_dir())
    res = {c: audit_case(root / c / "raw", a.max_partitions) for c in cases}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
