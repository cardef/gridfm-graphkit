"""Tables and the verdicts of results/abacus_r2/PREREGISTRATION.md, from the result files of multigrid.py.

python -m experiments.ac_pf_symmetries.r2_report [--results-dir results/abacus_r2]   # writes REPORT.md there

Rule from the pre-registration: with 3 seeds a difference is called only when the [min, max] ranges over seeds do
not overlap.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "abacus_r2"
DEG = 180.0 / math.pi
ARMS = ("canon", "m1fullcanon")
COLS = {"VM": (1.0, "VM (p.u.)"), "VA": (DEG, "VA (deg)"), "PG": (None, "PG (MW)"), "QG": (None, "QG (Mvar)"),
        "PBE": (None, "PBE (MVA)"), "VA_cm": (DEG, "VA offset (deg)"), "dVA": (DEG, "θ_f − θ_t (deg)")}


def phys(acc, base, q):
    scale = COLS[q][0]
    return acc[q] * (base if scale is None else scale)


def ms(a):
    return f"{a.mean():.3g} ± {a.std():.2g}" if len(a) > 1 else f"{a.mean():.3g}"


def separated(a, b):
    return bool(len(a) and len(b) and (a.max() < b.min() or b.max() < a.min()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=OUT)
    a = p.parse_args()
    runs = defaultdict(list)  # (fold dir, arm) -> results
    for f in sorted(a.results_dir.glob("heldout_*/*_seed*/result.json")):
        r = json.loads(f.read_text())
        runs[f.parts[-3], r["arm"]].append(r)
    folds = sorted({k[0] for k in runs})
    md = ["# R2: M0+Canon vs M1 on held-out grids", ""]
    md.append("Runs: " + ", ".join(f"{fd} {arm} {len(rs)}" for (fd, arm), rs in sorted(runs.items())) + "\n")
    p17, p18 = [], []
    for fd in folds:
        rs = {arm: runs.get((fd, arm), []) for arm in ARMS}
        if not all(rs.values()):
            continue
        any_r = rs["canon"][0]
        for kind, grids in (("zero-shot", any_r["heldout"]), ("in distribution", any_r["train"])):
            md += [f"### {fd}: {kind}", "", "| grid | arm | " + " | ".join(c[1] for c in COLS.values()) + " |",
                   "|---" * (len(COLS) + 2) + "|"]
            for g in grids:
                vals = {}
                for arm in ARMS:
                    cells = []
                    for q in COLS:
                        if kind == "zero-shot":
                            v = np.array([phys(r["zero_shot"][g]["acc"], r["zero_shot"][g]["baseMVA"], q) for r in rs[arm]])
                        else:
                            v = np.array([phys(r["in_dist"][g], r["baseMVA"][g], q) for r in rs[arm]])
                        vals[arm, q] = v
                        cells.append(ms(v))
                    md.append(f"| {g} | {arm} | " + " | ".join(cells) + " |")
                for q in ("VA", "VM") if kind == "zero-shot" else ("VA",):
                    a1, a0 = vals["m1fullcanon", q], vals["canon", q]
                    rec = (fd, g, q, separated(a1, a0), f"[{a1.min():.4g}, {a1.max():.4g}]", f"[{a0.min():.4g}, {a0.max():.4g}]")
                    (p17 if kind == "zero-shot" else p18).append(rec)
            md.append("")
    for name, recs in (("P17 (held-out grids)", p17), ("P18 (training grids)", p18)):
        if recs:
            ok = not any(x[3] for x in recs)
            md.append(f"**{name}: {'PASS' if ok else 'FAIL'}** (predicted no separation). " + "; ".join(
                f"{x[1]} {x[2]} m1 {x[4]} vs canon {x[5]}" + (" SEPARATED" if x[3] else "") for x in recs) + "\n")
    out = a.results_dir / "REPORT.md"
    out.write_text("\n".join(md))
    print(out.read_text())


if __name__ == "__main__":
    main()
