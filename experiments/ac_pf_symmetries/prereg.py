"""Verdicts on the predictions of results/abacus/PREREGISTRATION.md, computed from the result files only.

python -m experiments.ac_pf_symmetries.prereg [--results-dir results/abacus] [--case case14_ieee]

Rule from the pre-registration: with 3 seeds a difference is called only when the [min, max] ranges over seeds
do not overlap. "Within 2x" compares seed means.
"""

import argparse
import math
from pathlib import Path

import numpy as np

from .report import OUT, audit_rows, load

DEG = 180.0 / math.pi


def vals(runs, v, f):
    return np.array([f(r) for r in runs.get(v, [])], dtype=float)


def rng(a):
    return f"[{a.min():.4g}, {a.max():.4g}]" if len(a) else "n/a"


def below(a, b):
    """a entirely below b (ranges do not overlap)."""
    return len(a) > 0 and len(b) > 0 and a.max() < b.min()


def verdict(ok):
    return "PASS" if ok else "FAIL"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=OUT / "abacus")
    p.add_argument("--case", default="case14_ieee")
    a = p.parse_args()
    every = load(a.results_dir / a.case, "result.json")
    # the registered predictions P1-P9 are scored on seeds 0-2 only; later seeds feed P10 and the extensions
    runs = {v: [r for r in rs if r["seed"] in (0, 1, 2)] for v, rs in every.items()}
    print(f"runs per arm (all seeds): { {v: len(rs) for v, rs in every.items()} }\n")

    def ind(v, q, scale=1.0, src=None):
        return vals(src or runs, v, lambda r: r["in_dist"][q] * scale)

    # P1: canon and m0 overlap on every in-dist channel
    def overlap(src):
        out = {}
        for q in ("VM", "VA", "PG", "QG", "PBE"):
            c, m = ind("canon", q, src=src), ind("m0", q, src=src)
            out[q] = not (below(c, m) or below(m, c))
        return out

    over = overlap(runs)
    print(f"P1 {verdict(all(over.values()))}: ranges overlap per channel {over}")
    print(f"   extension, all seeds of canon/m0 (exploratory): {overlap(every)}")

    # P2: augcov VA within 2x of augmild and below 1 deg
    va = {v: ind(v, "VA", DEG) for v in ("augcov", "augmild", "aug", "m0", "augnophys", "m0nophys", "augphase")}
    ok = len(va["augcov"]) and va["augcov"].mean() < 2 * va["augmild"].mean() and va["augcov"].mean() < 1.0
    print(
        f"P2 {verdict(ok)}: VA deg augcov {rng(va['augcov'])} mean {va['augcov'].mean():.3g}, "
        f"augmild mean {va['augmild'].mean():.3g}, aug mean {va['aug'].mean():.3g}",
    )

    # P3: augnophys VA within 2x of m0nophys
    ok = len(va["augnophys"]) and va["augnophys"].mean() < 2 * va["m0nophys"].mean()
    print(
        f"P3 {verdict(ok)}: VA deg augnophys mean {va['augnophys'].mean():.3g} {rng(va['augnophys'])}, "
        f"m0nophys mean {va['m0nophys'].mean():.3g} {rng(va['m0nophys'])}",
    )

    # P4: augphase beats m0 on VA; reference routing: the common-offset ratio is the smallest
    beats = below(va["augphase"], va["m0"])
    ratios = {
        q: ind("augphase", q).mean() / ind("m0", q).mean() for q in ("VA", "VA_cm", "VA_diff", "dVA")
    }
    routing = ratios["VA_cm"] < ratios["VA_diff"] and ratios["VA_cm"] < ratios["dVA"]
    print(
        f"P4 gain {verdict(beats)} (VA deg augphase {rng(va['augphase'])} vs m0 {rng(va['m0'])}); "
        f"reference routing {verdict(routing)}: augphase/m0 ratios "
        + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items()),
    )

    # P5: canon VM zero-shot below m0 on both targets
    res = {}
    for t in ("case30_ieee", "case57_ieee"):
        c = vals(runs, "canon", lambda r: r["zero_shot"][t]["acc"]["VM"])
        m = vals(runs, "m0", lambda r: r["zero_shot"][t]["acc"]["VM"])
        res[t] = (below(c, m), rng(c), rng(m))
    print(
        f"P5 {verdict(all(x[0] for x in res.values()))}: zero-shot VM canon vs m0 "
        + "; ".join(f"{t} {x[1]} vs {x[2]}" for t, x in res.items()),
    )

    # P6: exactness
    def ratio(r, sym, param, q):
        row = audit_rows(r, sym, param)
        return row["ee"][q]["rmse"] / r["in_dist"][q]

    def canon_worst(frame_fix):
        return max(
            row["ee"][q]["rmse"]
            * (row["param"] if frame_fix and row["sym"] == "scale" and q in ("PG", "QG") else 1.0)
            / r["in_dist"][q]
            for r in runs.get("canon", [])
            for row in r["audit"]
            for q in ("VM", "VA", "PG", "QG")
        )

    m0 = runs.get("m0", [])
    s1 = min(ratio(r, "phase", math.pi, "VA") for r in m0)
    fl = max(max(ratio(r, "flip", 0.5, q) for q in ("VM", "VA")) for r in m0)
    ft = [ratio(r, "fliptrafo", 1.0, "VM") for r in m0]
    m0_ok = s1 > 10 and fl < 1e-3 and all(1e-3 < x < 1 for x in ft)
    # as registered, the PG/QG ratio under S3 mixed frames (EE in powers / k, RMSE in powers); report both
    print(
        f"P6 as registered {verdict(canon_worst(False) < 1e-3 and m0_ok)}, frame-consistent "
        f"{verdict(canon_worst(True) < 1e-3 and m0_ok)}: canon worst EE/RMSE {canon_worst(False):.1e} "
        f"(frame-consistent {canon_worst(True):.1e}); m0 EE_S1(pi)/RMSE min {s1:.3g}, "
        f"flip max {fl:.1e}, fliptrafo {[f'{x:.2g}' for x in ft]}",
    )

    # P7: H4, spread of zero-shot VM across the S3 representative a
    arms = [v for v in ("canon", "canonmix", "canonp95") if v in runs]
    if len(arms) == 3:
        out = {}
        for t in ("case30_ieee", "case57_ieee"):
            z = {v: vals(runs, v, lambda r: r["zero_shot"][t]["acc"]["VM"]) for v in arms}
            means = [z[v].mean() for v in arms]
            span, sd = max(means) - min(means), max(z[v].std() for v in arms)
            out[t] = (span > 2 * sd, span, sd, means)
        print(
            f"P7 {verdict(any(x[0] for x in out.values()))}: "
            + "; ".join(
                f"{t} span {x[1]:.3g} vs 2 sd {2 * x[2]:.3g} (means a=0/0.5/1: {', '.join(f'{m:.3g}' for m in x[3])})"
                for t, x in out.items()
            ),
        )
    else:
        print(f"P7 not run yet (arms present: {arms})")

    # P9: which axis of augcov breaks training (seed means)
    if "augcovphase" in runs and "augcovscale" in runs:
        ph, sc = ind("augcovphase", "VA", DEG), ind("augcovscale", "VA", DEG)
        print(
            f"P9 {verdict(sc.mean() > 1.0 and ph.mean() < 1.0)}: in-dist VA deg augcovscale {rng(sc)} mean "
            f"{sc.mean():.3g} (predicted > 1), augcovphase {rng(ph)} mean {ph.mean():.3g} (predicted < 1)",
        )

    # P10 (exploratory): sign of canon - m0 per seed; m0warm against both
    def by_seed(v):
        return {r["seed"]: r["in_dist"]["VA"] * DEG for r in every.get(v, [])}

    # P11-P13 (F1, M1): m1canon vs canon over seeds 0-4, every channel; registered rule = range separation
    # P11-P13 (head-swap M1) and P14-P16 (M1 in every layer): against canon over seeds 0-4, every channel;
    # registered rule = range separation in either direction
    for arm, (p_in, p_zs, p_ex) in {"m1canon": (11, 12, 13), "m1fullcanon": (14, 15, 16)}.items():
        if arm not in every:
            continue
        m1, cn = every[arm], every.get("canon", [])
        va1, vac = (np.array([r["in_dist"]["VA"] * DEG for r in x]) for x in (m1, cn))
        sep = below(va1, vac) or below(vac, va1)
        print(f"P{p_in} {verdict(not sep)}: in-dist VA deg {arm} {rng(va1)} vs canon {rng(vac)} (predicted overlap)")
        zs = {}
        for t in ("case30_ieee", "case57_ieee"):
            for q, sc in (("VA", DEG), ("VM", 1.0)):
                a1 = np.array([r["zero_shot"][t]["acc"][q] * sc for r in m1])
                a0 = np.array([r["zero_shot"][t]["acc"][q] * sc for r in cn])
                zs[t, q] = (below(a1, a0) or below(a0, a1), rng(a1), rng(a0))
        print(
            f"P{p_zs} {verdict(not any(x[0] for x in zs.values()))}: zero-shot {arm} vs canon "
            + "; ".join(f"{t[:6]} {q} {x[1]} vs {x[2]}" for (t, q), x in zs.items()),
        )
        worst = max(
            row["ee"][q]["rmse"]
            * (row["param"] if row["sym"] == "scale" and q in ("PG", "QG") else 1.0)
            / r["in_dist"][q]
            for r in m1
            for row in r["audit"]
            for q in ("VM", "VA", "PG", "QG")
        )
        print(f"P{p_ex} {verdict(worst < 1e-3)}: {arm} worst frame-consistent EE/RMSE {worst:.1e}")
        print(
            f"   R7 split, means: {arm} offset/rest/branch "
            + "/".join(f"{np.mean([r['in_dist'][q] * DEG for r in m1]):.3f}" for q in ("VA_cm", "VA_diff", "dVA"))
            + ", canon "
            + "/".join(f"{np.mean([r['in_dist'][q] * DEG for r in cn]):.3f}" for q in ("VA_cm", "VA_diff", "dVA")),
        )

    c, m, w = by_seed("canon"), by_seed("m0"), by_seed("m0warm")
    seeds = sorted(set(c) & set(m))
    signs = [c[s] < m[s] for s in seeds]
    print(
        f"P10 (exploratory): canon < m0 in {sum(signs)}/{len(seeds)} seeds; VA deg per seed "
        + ", ".join(f"s{s} canon {c[s]:.3f} m0 {m[s]:.3f}" + (f" m0warm {w[s]:.3f}" if s in w else "") for s in seeds),
    )


if __name__ == "__main__":
    main()
