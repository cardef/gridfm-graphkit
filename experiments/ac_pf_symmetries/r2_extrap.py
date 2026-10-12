"""Tables and verdicts of the size-extrapolation block (P47-P50 of results/abacus_r2/PREREGISTRATION.md), from the
result_extrap_<grid>.json files that multigrid.py --eval-only --extrap writes into the registered run dirs, and the
references of BASELINES_extrap.txt (flat, DC) and CLASSICAL_extrap.txt (physics floors).

python -m experiments.ac_pf_symmetries.r2_extrap [--results-dir results/abacus_r2] [--grids case500_goc case2000_goc]
    # writes REPORT_extrap.md there

Rule from the pre-registration: 3 seeds per (fold, arm); a difference is called only when the [min, max] ranges do
not overlap; P47-P49 are scored fold by fold on case500_goc.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "abacus_r2"
DEG = 180.0 / math.pi
ARMS = ("canon", "m1fullcanon", "m3canon", "canonresnorm", "dccanon", "dczcanon", "dcinjcanon", "canonvref",
        "dcinjvref", "dcinj1hop")
FOLDS = ("heldout_case30_ieee", "heldout_case57_ieee", "heldout_case118_ieee")
COLS = {"VM": (1.0, "VM (p.u.)"), "VM_PQ": (1.0, "VM, PQ (p.u.)"), "VA": (DEG, "VA (deg)"), "VA_cm": (DEG, "offset (deg)"),
        "VA_diff": (DEG, "rest (deg)"), "dVA": (DEG, "θ_f − θ_t (deg)"), "PBE": (None, "PBE (MVA)"), "QG": (None, "QG (Mvar)")}


def phys(acc, base, q):
    scale = COLS[q][0]
    return acc.get(q, math.nan) * (base if scale is None else scale)


def rng(a):
    return f"[{a.min():.3g}, {a.max():.3g}]" if len(a) else "—"


def separated(a, b):
    return bool(len(a) and len(b) and (a.max() < b.min() or b.max() < a.min()))


def baselines(path):
    """{grid: {'flat': VA, 'DC': VA, 'offset': .., 'rest': .., 'dVA': .., 'VMflat': ..}} from r2_baselines.py output."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(\S+) max\|VA\| .* VA flat ([\d.]+) \| VA DC ([\d.]+) \| VA DC offset ([\d.]+) \| VA DC rest ([\d.]+) "
                     r"\| dVA flat [\d.]+ \| dVA DC ([\d.]+) \| VM flat ([\d.]+)", line)
        if m:
            g, flat, dc, cm, rest, dva, vmflat = m.groups()
            out[g] = dict(flat=float(flat), DC=float(dc), offset=float(cm), rest=float(rest), dVA=float(dva), VMflat=float(vmflat))
    return out


def floors(path):
    """{grid: {method: {'VA': .., 'VM': ..}}} from r2_classical.py output (nr inputs, fdpf, nr oracle, dc, dcloss, ph lines)."""
    out, g = defaultdict(dict), None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(case\S+) \| oracle Newton", line)
        if m:
            g = m.group(1)
            continue
        m = re.match(r"\s+(nr inputs|fdpf|nr oracle|dc|ph|dcloss)\s+VA ([\d.e+-]+) .* VM ([\d.e+-]+)", line)
        if m and g:
            out[g][m.group(1)] = {"VA": float(m.group(2)), "VM": float(m.group(3))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=OUT)
    p.add_argument("--grids", nargs="+", default=["case500_goc", "case2000_goc"])
    a = p.parse_args()
    bpath, cpath = a.results_dir / "BASELINES_extrap.txt", a.results_dir / "CLASSICAL_extrap.txt"
    base = baselines(bpath) if bpath.exists() else {}
    flo = floors(cpath) if cpath.exists() else {}
    runs = defaultdict(list)  # (fold, arm, grid) -> results
    for g in a.grids:
        for f in sorted(a.results_dir.glob(f"heldout_*/*_seed*/result_extrap_{g}.json")):
            r = json.loads(f.read_text())
            runs[f.parts[-3], r["arm"], g].append(r)
    md = ["# Size extrapolation of the trained arms (P47-P50)", ""]
    md.append("Runs: " + ", ".join(f"{k[0][8:]}/{k[1]}/{k[2]} {len(v)}" for k, v in sorted(runs.items())) + "\n")
    vals = {}
    for g in a.grids:
        ref, fl = base.get(g, {}), flo.get(g, {})
        md += [f"## {g}: zero-shot, 1024 scenarios", ""]
        if ref:
            md.append(f"References: flat VA {ref['flat']:.3g} deg, DC VA {ref['DC']:.3g} (offset {ref['offset']:.3g}, rest "
                      f"{ref['rest']:.3g}, θ_f − θ_t {ref['dVA']:.3g}), VM flat {ref['VMflat']:.3g}"
                      + (f"; inputs-only physics floor VA {fl['nr inputs']['VA']:.3g} VM {fl['nr inputs']['VM']:.3g}" if "nr inputs" in fl else "")
                      + (f"; fast-decoupled from inputs VA {fl['fdpf']['VA']:.3g}" if "fdpf" in fl else "") + "\n")
        md += ["| fold | arm | " + " | ".join(c[1] for c in COLS.values()) + " | VA / DC |", "|---" * (len(COLS) + 3) + "|"]
        for fd in FOLDS:
            for arm in ARMS:
                rs = runs.get((fd, arm, g))
                if not rs:
                    continue
                cells = []
                for q in COLS:
                    v = np.array([phys(r["zero_shot"][g]["acc"], r["zero_shot"][g]["baseMVA"], q) for r in rs])
                    vals[fd, arm, g, q] = v
                    cells.append(rng(v))
                va = vals[fd, arm, g, "VA"]
                ratio = f"{va.mean() / ref['DC']:.2f} [{va.min() / ref['DC']:.2f}, {va.max() / ref['DC']:.2f}]" if ref else "—"
                md.append(f"| {fd[8:]} | {arm} | " + " | ".join(cells) + f" | {ratio} |")
        md.append("")
    g = "case500_goc"
    ref = base.get(g)
    if ref and any(k[2] == g for k in runs):
        dc = ref["DC"]
        md += ["## Verdicts (case500_goc)", ""]
        rows = [(fd, vals[fd, "dccanon", g, "VA"]) for fd in FOLDS if (fd, "dccanon", g, "VA") in vals]
        ok = [bool(0.9 * dc <= v.min() and v.max() <= 1.1 * dc) for _, v in rows]
        md.append(f"**P47 (dccanon VA within [0.9, 1.1] x DC = [{0.9 * dc:.3g}, {1.1 * dc:.3g}] deg, every seed of every fold): "
                  f"{('PASS' if all(ok) else 'FAIL') if len(rows) == 3 else 'incomplete'}** "
                  + "; ".join(f"{fd[8:]} {rng(v)}" + (" ok" if o else "") for (fd, v), o in zip(rows, ok)) + "\n")
        rows = [(fd, vals[fd, "dcinjcanon", g, "VA"]) for fd in FOLDS if (fd, "dcinjcanon", g, "VA") in vals]
        ok = [bool(v.max() > dc and v.max() / v.min() > 2) for _, v in rows]
        md.append(f"**P48 (dcinjcanon: in every fold a seed above DC's {dc:.3g} deg and seed max/min > 2): "
                  f"{('PASS' if all(ok) else 'FAIL') if len(rows) == 3 else 'incomplete'}** "
                  + "; ".join(f"{fd[8:]} {rng(v)} spread {v.max() / v.min():.1f}x" + (" ok" if o else "") for (fd, v), o in zip(rows, ok)) + "\n")
        rows = [(fd, arm, vals[fd, arm, g, "VA"], vals[fd, arm, g, "VA_cm"]) for fd in FOLDS
                for arm in ("canon", "m1fullcanon", "m3canon", "canonvref") if (fd, arm, g, "VA") in vals]
        ok = [bool(v.min() >= 4 * dc and (cm**2).mean() > 0.5 * (v**2).mean()) for _, _, v, cm in rows]
        md.append(f"**P49 (canon-like arms VA >= 4 x DC = {4 * dc:.3g} deg every seed, offset the larger part of VA^2): "
                  f"{('PASS' if all(ok) else 'FAIL') if len(rows) == 12 else 'incomplete'}** "
                  + "; ".join(f"{fd[8:]}/{arm} {rng(v)} offset²/VA² {(cm**2).mean() / (v**2).mean():.2f}" + (" ok" if o else "")
                              for (fd, arm, v, cm), o in zip(rows, ok)) + "\n")
    out = a.results_dir / "REPORT_extrap.md"
    out.write_text("\n".join(md))
    print(out.read_text())


if __name__ == "__main__":
    main()
