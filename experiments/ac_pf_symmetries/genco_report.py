"""Aggregate results/genco/ (written by cluster/genco_audit.sbatch) into results/genco/REPORT.md.

python -m experiments.ac_pf_symmetries.genco_report

Gates G1-G3 and the refit gate, E0p, E1 (EE of the released checkpoints), E2 (post-hoc Canonicalize),
E5a-c (GENCO_EXPERIMENTS.md, amendment 2026-10-01). Missing files are reported, never skipped silently.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "genco"
CKPT = HERE.parents[1] / "data" / "genco_ckpt"
SPLITS = HERE.parents[1] / "data" / "genco" / "r"
CASES = ["case14_ieee", "case30_ieee", "case57_ieee", "case118_ieee"]
SIZES = ["base", "small", "tiny"]
SEEDS = [0, 1, 42]
CH = ["VM", "VA", "PG", "QG"]
DEG = 180.0 / math.pi
missing = []


def tag(c, z, s):
    return f"{c}_{z}_s{s}"


def load(p):
    if not p.exists():
        missing.append(str(p.relative_to(OUT)))
        return None
    return json.loads(p.read_text())


def released(c, z, s):
    d = CKPT / c / z / f"seed{s}"
    m = {r["Metric"]: float(r["Value"]) for r in csv.DictReader(open(d / "metrics.csv"))}
    st = torch.load(d / "normalizer_stats.pt", map_location="cpu", weights_only=True)
    return m, float(st[c]["baseMVA"])


def phys(acc, base):
    """Physical units: VM p.u., VA deg, PG MW, QG Mvar, PBE MVA."""
    return {
        "VM": acc["VM"],
        "VA": acc["VA"] * DEG,
        "PG": acc["PG"] * base,
        "QG": acc["QG"] * base,
        "PBE": acc["PBE"] * base,
    }


def ms(xs, fmt="{:.3g}"):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if not len(a):
        return "n/a"
    sd = f"{a.std():.1%}" if "%" in fmt else f"{a.std():.2g}"  # spread in the mean's unit
    return fmt.format(a.mean()) + (f" ± {sd}" if len(a) > 1 else "")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def probe(r, sym, param):
    return next(
        (a for a in r["audit"] if a["sym"] == sym and abs(a["param"] - param) < 1e-9),
        None,
    )


def gates():
    rows, fails = [], 0
    for c in CASES:
        for z in SIZES:
            for s in SEEDS:
                t = tag(c, z, s)
                g, gr = load(OUT / "g1" / f"{t}.json"), load(OUT / "gate_refit" / f"{t}.json")
                if g is None:
                    continue
                m, base = released(c, z, s)
                sp = json.loads((SPLITS / c / "splits" / f"{z}_seed{s}.json").read_text())
                ours = g["in_dist"]["PBE"] * g["baseMVA"]
                rel = ours / m["PBE Mean"] - 1
                g1 = abs(rel) < 0.02
                g2 = g["n_test"] == sp["official_sizes"]["test"]
                fails += not (g1 and g2)
                refit = gr["baseMVA"] / base - 1 if gr else float("nan")
                rows.append(
                    [t, f"{g['n_test']} / {sp['official_sizes']['test']}", f"{ours:.4g}", f"{m['PBE Mean']:.4g}",
                     f"{rel:+.2%}", "pass" if g1 and g2 else "**FAIL**", f"{base:.4g}", f"{refit:+.2%}"],
                )
    head = ["checkpoint", "test graphs (ours / official)", "PBE ours (MVA)", "PBE released (MVA)", "rel",
            "G1+G2", "released baseMVA", "refit on 5000 train ids vs released"]
    return "### Gates (G3 = strict load, asserted by the script)\n\n" + table(head, rows) + (
        f"\n{fails} checkpoint(s) fail G1/G2.\n"
    )


def e0p():
    keys = ["scenarios", "ref_va_deg", "shift_nonzero_branches", "tap_ne_1_branches", "tap_gt_1_branches",
            "scenarios_with_branch_outage", "gens_out_of_service", "mean_abs_Yff_cv_across_scenarios"]
    rows = []
    for c in CASES:
        r = load(OUT / f"e0p_{c}.json")
        if r:
            rows.append([c] + [r[c].get(k, "n/a") for k in keys])
    return "### E0p: conventions in the public datasets (all scenarios)\n\n" + table(["grid"] + keys, rows)


PROBES = [("phase", math.pi, "VA"), ("phase", 0.1, "VA"), ("scale", 0.1, "VM"), ("scale", 10.0, "VM"),
          ("scale", 0.01, "VM"), ("scale", 100.0, "VM"), ("flip", 0.5, "VA"), ("fliptrafo", 1.0, "VM")]


def e1_e2():
    rows_ee, rows_acc, rows_canon = [], [], []
    for c in CASES:
        for z in SIZES:
            raw = [load(OUT / "e1" / f"{tag(c, z, s)}.json") for s in SEEDS]
            can = [load(OUT / "e2" / f"{tag(c, z, s)}.json") for s in SEEDS]
            raw = [r for r in raw if r]
            if not raw:
                continue
            ee = [ms([probe(r, sy, p)["ee"][q]["rmse"] / r["in_dist"][q] for r in raw]) for sy, p, q in PROBES]
            rows_ee.append([f"{c} {z}"] + ee)
            ph = [phys(r["in_dist"], r["baseMVA"]) for r in raw]
            rows_acc.append([f"{c} {z}"] + [ms([p[k] for p in ph]) for k in ("VM", "VA", "PG", "QG", "PBE")])
            pairs = [(r, k) for r, k in zip(raw, can) if k]
            if pairs:
                rel = [ms([k["in_dist"][q] / r["in_dist"][q] - 1 for r, k in pairs], "{:+.2%}")
                       for q in ("VM", "VA", "PG", "QG", "PBE")]
                worst = max(v["rel"] for _, k in pairs for a in k["audit"] for v in a["ee"].values())
                rows_canon.append([f"{c} {z}"] + rel + [f"{worst:.1e}"])
    hp = [f"{s} {p:g} ({q})" for s, p, q in PROBES]
    md = "### E1: EE_g / in-dist RMSE of the released checkpoints (mean ± std over seeds 0/1/42)\n\n"
    md += table(["checkpoint"] + hp, rows_ee)
    md += "\n### Released checkpoints, in-dist RMSE on the first 2000 official test graphs\n\n"
    md += table(["checkpoint", "VM (p.u.)", "VA (deg)", "PG (MW)", "QG (Mvar)", "PBE (MVA)"], rows_acc)
    md += "\n### E2: post-hoc Canonicalize, in-dist change relative to raw, and its worst EE (rel)\n\n"
    md += table(["checkpoint", "VM", "VA", "PG", "QG", "PBE", "max EE rel"], rows_canon)
    return md


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def partial_spearman(x, y, groups):
    """Spearman correlation of x and y after regressing both rank vectors on one-hot group indicators."""
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    cols = [np.ones(len(rx))]
    for g in groups:
        levels = sorted(set(g))
        cols += [np.array([v == lv for v in g], dtype=float) for lv in levels[1:]]
    X = np.stack(cols, 1)
    res = [v - X @ np.linalg.lstsq(X, v, rcond=None)[0] for v in (rx, ry)]
    raw = np.corrcoef(rx, ry)[0, 1]
    return raw, np.corrcoef(*res)[0, 1]


def e5():
    zs, checks_a, checks_b, h2 = defaultdict(list), [], [], []
    for c in CASES:
        for z in SIZES:
            for s in SEEDS:
                for t in CASES:
                    if t == c:
                        continue
                    stem = OUT / "e5" / f"{tag(c, z, s)}__{t}"
                    src, ref = load(Path(f"{stem}_source.json")), load(Path(f"{stem}_refit.json"))
                    cs, cr = load(Path(f"{stem}_canon_source.json")), load(Path(f"{stem}_canon_refit.json"))
                    if not (src and ref):
                        continue
                    bs, br = src["baseMVA"], ref["baseMVA"]
                    ps, pr = phys(src["in_dist"], bs), phys(ref["in_dist"], br)
                    pc = phys(cs["in_dist"], cs["baseMVA"]) if cs else None
                    zs[c, z, t].append((ps, pr, pc))
                    # E5a: the refit is S3 at k = br / bs, probed in the source run on the same graphs
                    pb = src["scale_probes"][0]
                    k = pb["param"]
                    assert abs(k - br / bs) < 1e-6 * k, (stem, k, br / bs)
                    pp = phys(pb["acc"], br)
                    pp["PBE"] = pb["acc"]["PBE"] * bs  # accuracy() reports T_k's PBE in the original base
                    for q in CH + ["PBE"]:
                        same = abs(pp[q] / pr[q] - 1)
                        bound = None
                        if q in CH:
                            unit = {"VM": 1, "VA": DEG, "PG": br, "QG": br}[q]
                            bound = pb["ee"][q]["rmse"] * unit
                        checks_a.append((stem.name, q, same, abs(pr[q] - ps[q]), bound))
                    if cs and cr:
                        # physical units: each run's per-unit is its own normalizer's
                        pcr = phys(cr["in_dist"], cr["baseMVA"])
                        checks_b.append(max(abs(pcr[q] / pc[q] - 1) for q in CH))
                    ee10 = probe(src, "scale", 10.0)["ee"]["VM"]["rmse"]
                    h2.append((ee10, ps["VM"], pc["VM"] if pc else None, c, t, z))
    md = "### E5: zero-shot on the other grids (2000 test graphs, mean ± std over seeds)\n\n"
    rows = []
    for (c, z, t), v in zs.items():
        for name, i in (("source", 0), ("refit", 1), ("canon", 2)):
            vals = [x[i] for x in v if x[i] is not None]
            rows.append([f"{c} {z}", t, name] + [ms([p[q] for p in vals]) for q in CH + ["PBE"]])
    md += table(["source", "target", "normalizer", "VM (p.u.)", "VA (deg)", "PG (MW)", "QG (Mvar)",
                 "PBE (MVA)"], rows)
    if checks_a:
        same = max(x[2] for x in checks_a)
        viol = [x for x in checks_a if x[4] is not None and x[3] > x[4] * (1 + 1e-4) + 1e-12]
        tight = np.median([x[3] / x[4] for x in checks_a if x[4]])
        md += (f"\nE5a: {len(checks_a)} (pair, channel) checks. Probe at k = refit/source base vs the refit run, "
               f"worst relative difference {same:.1e}; |ΔRMSE| > EE_S3(k): {len(viol)} violation(s); median "
               f"|ΔRMSE| / EE = {tight:.2f}.\n")
    if checks_b:
        md += f"\nE5b: canon under source vs refit normalizer, worst relative difference {max(checks_b):.1e} over {len(checks_b)} pairs.\n"
    if len(h2) > 3:
        x, y, _, c, t, z = zip(*h2)
        r, rp = partial_spearman(x, y, [c, t, z])
        md += (f"\nE5c (H2, exploratory): Spearman ρ(EE_S3,VM(k=10) on the target, zero-shot VM RMSE) over "
               f"{len(h2)} pairs = {r:.2f}; residualized on (source, target, size) = {rp:.2f}.\n")
    held = [h for h in h2 if h[3] != "case14_ieee" and h[2] is not None]
    if len(held) > 3:
        x, raw, can, c, t, z = map(np.array, zip(*held))
        g = [list(c), list(t), list(z)]
        _, p1 = partial_spearman(x, raw, g)
        _, p2 = partial_spearman(x, raw - can, g)
        _, p3 = partial_spearman(x, can, g)
        md += (f"\nE5d (amendment 22:00, {len(held)} held-out pairs, sources case30/57/118), ρ_partial with "
               f"EE_S3,VM(k=10): raw error {p1:.2f} (E5d-1: > 0.5), raw − canon {p2:.2f} (E5d-2: > 0.5), "
               f"canon error {p3:.2f} (E5d-3: below raw).\n")
    return md


def main():
    md = ["# GENCO released checkpoints: Piano A F0 audit (cluster run)", ""]
    md += [gates(), e0p(), e1_e2(), e5()]
    if missing:
        md.append(f"\n**Missing result files ({len(missing)}):** " + ", ".join(sorted(missing)[:40]) + "\n")
    out = OUT / "REPORT.md"
    out.write_text("\n".join(md))
    print(out.read_text())


if __name__ == "__main__":
    main()
