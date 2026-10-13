"""Tables and verdicts of Piano X, X3 (X3-P1-X3-P6 of results/abacus_r2/PREREGISTRATION.md): GENCO with the injection
decoder trained on the 31 public subgrids, zero-shot on unseen grids.

Inputs: results/pianox_pt/heldout_case118_ieee/<arm>_seed<s>/result.json (zero-shot on case118_ieee at the end of
training) and result_extrap_*.json (the other test grids), each with the source and the target-fitted normalizer;
references from results/pianox/frontier_<case>.json and results/pianox_pt/frontier_pt_case118_tr.json (DC = the k = 0
start, fast-decoupled after 5 iterations with the REF voltage from the inputs and given), and GENCO's public transfer
checkpoints in results/pianox_pt/public/.

python -m experiments.ac_pf_symmetries.r2_pt   # writes results/pianox_pt/REPORT_pt.md
python -m experiments.ac_pf_symmetries.r2_pt --runs results/pianox_pt_a --out REPORT_pt_a.md   # X3a (gain 0.01)
python -m experiments.ac_pf_symmetries.r2_pt --runs results/pianox_pt_loop --out REPORT_pt_loop.md   # X4 (closed loop)
python -m experiments.ac_pf_symmetries.r2_pt --runs results/pianox_pt_loopg --out REPORT_pt_loopg.md   # X4b (guarded)

Also a training-stability table from each run's metrics.csv (X3a-P0: a collapse is an epoch after the fifth with
validation loss above 0.5; the collapsed state of X3 sat at 0.9-1.3, the healthy one at 0.13-0.19).
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
PT = RES / "pianox_pt"
DEG = 180 / math.pi
GRIDS = ("case118_ieee", "case118_tr", "case500_goc", "case2000_goc", "case57_ieee", "case30_ieee", "case14_ieee")
ARMS = ("plain", "dcinjcanon", "dcinjvref", "dcloopcanon", "dcloopvref", "dcloopgcanon", "dcloopgvref")
Q = ("VA", "VA_cm", "VA_diff", "dVA", "VM_PQ", "PBE")


def val(acc, base, q):
    v = acc.get(q, math.nan)
    return v * DEG if q in ("VA", "VA_cm", "VA_diff", "dVA") else v * base if q == "PBE" else v


def rng(a):
    a = np.asarray(a, float)
    return f"[{np.nanmin(a):.3g}, {np.nanmax(a):.3g}]" if a.size else "—"


def references():
    out = {}
    for g in GRIDS:
        f = PT / "frontier_pt_case118_tr.json" if g == "case118_tr" else RES / "pianox" / f"frontier_{g}.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())["metrics"]
        out[g] = {"DC": m["dc given k=0"]["VA"], "fdpf inputs k=5": m["fdpf inputs k=5"]["VA"],
                  "fdpf given k=5": m["fdpf given k=5"]["VA"]}
    return out


COLLAPSE, AFTER = 0.5, 5  # X3a-P0


def stability(root):
    """Per run: epochs logged, best validation loss and its epoch, max validation loss after epoch AFTER, collapsed."""
    out = {}
    for f in sorted((root / "heldout_case118_ieee").glob("*_seed*/logs/version_0/metrics.csv")):
        rows = [(int(r["epoch"]), float(r["Validation loss"])) for r in csv.DictReader(open(f)) if r["Validation loss"]]
        if not rows:
            continue
        best = min(rows, key=lambda x: x[1])
        late = [v for e, v in rows if e > AFTER]
        out[f.parents[2].name] = {"epochs": len(rows), "best": best[1], "best_epoch": best[0],
                                  "max_late": max(late) if late else math.nan,
                                  "collapsed": bool(late) and max(late) > COLLAPSE}
    return out


def load_runs(root):
    """{(arm, norm, grid, q): array over seeds}; norm is 'source' or 'refit'."""
    runs = defaultdict(dict)
    for f in sorted((root / "heldout_case118_ieee").glob("*_seed*/result*.json")):
        r = json.loads(f.read_text())
        for norm, key in (("source", "zero_shot"), ("refit", "zero_shot_refit")):
            for g, z in r.get(key, {}).items():
                for q in Q:
                    runs[r["arm"], norm, g, q][r["seed"]] = val(z["acc"], z["baseMVA"], q)
    return {k: np.array([v[s] for s in sorted(v)]) for k, v in runs.items()}


LOOP = {"pianox_pt_loop": ("X4", "dcloopcanon", "dcloopvref"), "pianox_pt_loopg": ("X4b", "dcloopgcanon", "dcloopgvref")}


def x4_verdicts(st, runs, ref, get, verdict, pre, canon, vref):
    """X4-P0-X4-P3 (PREREGISTRATION.md) for the plain (X4) or the guarded (X4b) closed loop; P2 compares with the six
    open-loop dcinjcanon seeds of X3 and X3a on case500_goc."""
    md = []
    loop = [k for k in st if k.startswith((canon + "_", vref + "_"))]
    complete = len(loop) == 6 and all(st[k]["epochs"] > AFTER for k in loop)
    md.append(f"**{pre}-P0 (no closed-loop seed above {COLLAPSE} in validation after epoch {AFTER}): "
              f"{verdict(complete and not any(st[k]['collapsed'] for k in loop), complete)}** "
              + "; ".join(f"{k} max {st[k]['max_late']:.3g}" for k in loop) + "\n")
    grids = ("case118_ieee", "case118_tr", "case500_goc", "case2000_goc")
    cells = [(g, get(canon, "source", g)) for g in grids]
    complete = all(len(v) == 3 for _, v in cells) and all(g in ref for g in grids)
    ok = complete and all((v < ref[g]["DC"]).all() for g, v in cells)
    md.append(f"**{pre}-P1 ({canon} below DC on 118, 118_tr, 500 and 2000, every seed): {verdict(ok, complete)}** "
              + "; ".join(f"{g} {rng(v)} vs DC {ref.get(g, {}).get('DC', math.nan):.3g}" for g, v in cells) + "\n")
    open_loop = np.concatenate([load_runs(r).get(("dcinjcanon", "source", "case500_goc", "VA"), np.array([]))
                                for r in (PT, RES / "pianox_pt_a")])
    v = get(canon, "source", "case500_goc")
    complete = len(v) == 3 and len(open_loop) == 6
    ok = complete and (v < open_loop.min()).all()
    md.append(f"**{pre}-P2 ({canon} on case500_goc below all six open-loop dcinjcanon seeds of X3 and X3a): "
              f"{verdict(ok, complete)}** {rng(v)} vs open-loop min {open_loop.min() if len(open_loop) else math.nan:.3g} "
              f"(open-loop seeds {', '.join(f'{x:.3g}' for x in sorted(open_loop))})\n")
    cells = [(g, a, get(a, "source", g), ref.get(g, {}).get("fdpf given k=5" if a == vref else "fdpf inputs k=5", math.nan))
             for g in ("case118_ieee", "case500_goc", "case2000_goc") for a in (canon, vref)]
    complete = all(len(c[2]) == 3 for c in cells)
    ok = complete and all((v >= r).all() for *_, v, r in cells)
    md.append(f"**{pre}-P3 (no closed-loop seed below fast-decoupled from the same inputs after 5 iterations on 118 / 500 / "
              f"2000): {verdict(ok, complete)}** " + "; ".join(
                  f"{g}/{a} min {np.nanmin(v) if len(v) else math.nan:.3g} vs {r:.3g}" for g, a, v, r in cells) + "\n")
    return md


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=PT, help="results root of the runs (X3: results/pianox_pt)")
    p.add_argument("--out", default="REPORT_pt.md", help="report file name, written in --runs")
    opts = p.parse_args()
    root = opts.runs if opts.runs.is_absolute() else Path(__file__).resolve().parent / opts.runs
    ref, runs = references(), load_runs(root)
    present = [arm for arm in ARMS if any(k[0] == arm for k in runs)]
    pub = {}
    for f in sorted((PT / "public").glob("*.json")):
        r = json.loads(f.read_text())
        pub[r["model"], r["grid"]] = {q: val(r["acc"], r["baseMVA"], q) for q in Q}
    md = [f"# X3: GENCO with the injection decoder on the 31 public subgrids, zero-shot ({root.name})", ""]
    md.append("Seeds per (arm, normalizer): " + ", ".join(
        f"{a}/{n} {len(runs.get((a, n, 'case118_ieee', 'VA'), []))}" for a in present for n in ("source", "refit")) + "\n")
    st = stability(root)
    md += ["## Training stability", "", "| run | epochs | best val. loss (epoch) | max val. loss after epoch 5 | collapsed |",
           "|---|---|---|---|---|"]
    md += [f"| {k} | {v['epochs']} | {v['best']:.3g} ({v['best_epoch']}) | {v['max_late']:.3g} | {v['collapsed']} |"
           for k, v in st.items()]
    md.append("")
    for g in GRIDS:
        md += [f"## {g}", ""]
        if g in ref:
            md.append("References (VA deg): " + ", ".join(f"{k} {v:.3g}" for k, v in ref[g].items()) + "\n")
        md += ["| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |", "|---" * 9 + "|"]
        for a in present:
            for n in ("source", "refit"):
                if (a, n, g, "VA") not in runs:
                    continue
                va = runs[a, n, g, "VA"]
                ratio = rng(va / ref[g]["DC"]) if g in ref else "—"
                md.append(f"| {a} | {n} | " + " | ".join(rng(runs[a, n, g, q]) for q in Q) + f" | {ratio} |")
        for (model, gg), v in sorted(pub.items()):
            if gg == g:
                ratio = f"{v['VA'] / ref[g]['DC']:.2f}" if g in ref else "—"
                md.append(f"| public {model} | {'target' if model == 'pretrained' else 'own stats'} | "
                          + " | ".join(f"{v[q]:.3g}" for q in Q) + f" | {ratio} |")
        md.append("")

    def get(a, n, g):
        return runs.get((a, n, g, "VA"), np.array([]))

    def verdict(ok, complete):
        return ("PASS" if ok else "FAIL") if complete else "incomplete"

    md += ["## Verdicts", ""]
    if root.name in LOOP:
        md += x4_verdicts(st, runs, ref, get, verdict, *LOOP[root.name])
        (root / opts.out).write_text("\n".join(md))
        print("\n".join(md))
        return
    pre = "X3a" if root.name == "pianox_pt_a" else "X3"  # the same predictions, scored on X3 or on X3a's runs
    inj = [k for k in st if k.startswith("dcinjcanon_")]
    complete = len(inj) == 3 and all(st[k]["epochs"] > AFTER for k in inj)
    if pre == "X3a":
        md.append(f"**X3a-P0 (no dcinjcanon seed collapses: validation loss at most {COLLAPSE} after epoch {AFTER}): "
                  f"{verdict(complete and not any(st[k]['collapsed'] for k in inj), complete)}** "
                  + "; ".join(f"{k} max {st[k]['max_late']:.3g}" for k in inj) + "\n")
    else:
        md.append(f"Stability (descriptive, X3 registered no stability prediction; X3a-P0's criterion): "
                  f"{sum(st[k]['collapsed'] for k in inj)} of {len(inj)} dcinjcanon seeds collapse; "
                  + "; ".join(f"{k} max {st[k]['max_late']:.3g}" for k in inj) + "\n")
    cells = [(g, get("dcinjcanon", "source", g)) for g in ("case118_ieee", "case118_tr")]
    complete = all(len(v) == 3 for _, v in cells) and all(g in ref for g, _ in cells)
    ok = complete and all((v < ref[g]["DC"]).all() for g, v in cells)
    md.append(f"**{pre}-P1 (dcinjcanon below DC on both case118 sets, every seed): {verdict(ok, complete)}** " + "; ".join(
        f"{g} {rng(v)} vs DC {ref.get(g, {}).get('DC', math.nan):.3g}" for g, v in cells) + "\n")
    v = get("dcinjcanon", "source", "case500_goc")
    complete = len(v) == 3 and "case500_goc" in ref
    md.append(f"**{pre}-P2 (dcinjcanon below DC on case500_goc, every seed): "
              f"{verdict(complete and (v < ref['case500_goc']['DC']).all(), complete)}** {rng(v)} vs DC "
              f"{ref.get('case500_goc', {}).get('DC', math.nan):.3g}\n")
    if "plain" in present:
        cells = [(g, n, get("plain", n, g)) for g in ("case118_ieee", "case500_goc") for n in ("source", "refit")]
        complete = all(len(v) == 3 for *_, v in cells)
        ok = complete and all((v > ref[g]["DC"]).all() for g, _, v in cells)
        md.append(f"**{pre}-P3 (plain above DC on case118_ieee and case500_goc, every seed, either normalizer): "
                  f"{verdict(ok, complete)}** " + "; ".join(f"{g}/{n} {rng(v)}" for g, n, v in cells) + "\n")
    v = get("dcinjcanon", "source", "case2000_goc")
    complete = len(v) == 3 and "case2000_goc" in ref
    md.append(f"**{pre}-P4 (on case2000_goc at least one dcinjcanon seed above DC): "
              f"{verdict(complete and (v > ref['case2000_goc']['DC']).any(), complete)}** {rng(v)} vs DC "
              f"{ref.get('case2000_goc', {}).get('DC', math.nan):.3g}\n")
    cells = []
    for g in ("case118_ieee", "case500_goc", "case2000_goc"):
        for a in present:
            for n in (("source",) if a != "plain" else ("source", "refit")):
                v = get(a, n, g)
                key = "fdpf given k=5" if a == "dcinjvref" else "fdpf inputs k=5"
                cells.append((g, a, n, v, ref.get(g, {}).get(key, math.nan)))
    complete = bool(cells) and all(len(c[3]) == 3 for c in cells)
    ok = complete and all((v >= r).all() for *_, v, r in cells)
    md.append(f"**{pre}-P5 (no learned arm below fast-decoupled from the same inputs after 5 iterations on 118 / 500 / 2000): "
              f"{verdict(ok, complete)}** " + "; ".join(f"{g}/{a}/{n} min {np.nanmin(v) if len(v) else math.nan:.3g} vs {r:.3g}"
                                                       for g, a, n, v, r in cells) + "\n")
    (root / opts.out).write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
