"""Tables and verdicts of Piano K (K-P1-K-P5 of results/abacus_r2/PREREGISTRATION.md): topology transfer on case118.

Inputs under results/pianok: heldout_case118_nk1/<arm>_seed<s>/result.json (in-distribution on the official test
split, zero-shot on the N-1 set we generated) and result_extrap_*.json (zero-shot on the other N-k sets, from
multigrid.py --eval-only --extrap); genco_released/<size>_seed0_case118_nk<k>.json (the released checkpoints on the
same sets, genco_audit.py); BASELINES_nk.txt and CLASSICAL_nk.txt (r2_baselines.py / r2_classical.py on every set).

python -m experiments.ac_pf_symmetries.r2_k [--results-dir results/pianok]   # writes REPORT_k.md there

Rule from the pre-registration: 3 seeds; ratios across k are per seed and a criterion holds only if every seed meets it.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .r2_extrap import baselines, floors

OUT = Path(__file__).resolve().parent / "results" / "pianok"
DEG = 180.0 / math.pi
KS = (1, 2, 3, 5, 10, 20)
ARMS = ("plain", "dccanon", "dcinjcanon", "gp")
SIZES = ("base", "small", "tiny")
Q = {"VA": DEG, "VA_cm": DEG, "VA_diff": DEG, "dVA": DEG, "VM_PQ": 1.0, "VM": 1.0, "PBE": None}


def val(acc, base, q):
    s = Q[q]
    return acc.get(q, math.nan) * (base if s is None else s)


def rng(a):
    return f"[{np.nanmin(a):.3g}, {np.nanmax(a):.3g}]" if len(a) else "—"


def load_arms(d):
    """{(arm, k, q): array over seeds} from result.json (k = 1) and result_extrap_*.json (other k)."""
    out = defaultdict(dict)
    for f in sorted(d.glob("heldout_case118_nk1/*_seed*/result*.json")):
        r = json.loads(f.read_text())
        for g, z in r["zero_shot"].items():
            k = int(g.split("_nk")[1])
            for q in Q:
                out[r["arm"], k, q][r["seed"]] = val(z["acc"], z["baseMVA"], q)
        if r.get("in_dist"):
            for q in Q:
                out[r["arm"], 0, q][r["seed"]] = val(r["in_dist"]["case118_k"], r["baseMVA"]["case118_k"], q)
    return {key: np.array([v[s] for s in sorted(v)]) for key, v in out.items()}


def load_released(d):
    """{(size, k, q): value} from genco_audit.py outputs (seed 0 only)."""
    out = {}
    for f in sorted(d.glob("genco_released/*_seed0_case118_nk*.json")):
        r = json.loads(f.read_text())
        size, k = f.name.split("_seed0_")[0], int(f.name.split("_nk")[1].split(".")[0])
        for q in Q:
            out[size, k, q] = val(r["in_dist"], r["baseMVA"], q)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=OUT)
    a = p.parse_args()
    arms, rel = load_arms(a.results_dir), load_released(a.results_dir)
    base = baselines(a.results_dir / "BASELINES_nk.txt") if (a.results_dir / "BASELINES_nk.txt").exists() else {}
    flo = floors(a.results_dir / "CLASSICAL_nk.txt") if (a.results_dir / "CLASSICAL_nk.txt").exists() else {}
    dc = {k: base.get(f"case118_nk{k}", {}).get("DC", math.nan) for k in KS}
    md = ["# Piano K: topology transfer on case118 (K-P1-K-P5)", ""]
    md.append("References per k (deg): " + "; ".join(
        f"k={k}: flat {base.get(f'case118_nk{k}', {}).get('flat', math.nan):.3g}, DC {dc[k]:.3g} (offset "
        f"{base.get(f'case118_nk{k}', {}).get('offset', math.nan):.3g}), fdpf from inputs "
        f"{flo.get(f'case118_nk{k}', {}).get('fdpf', {}).get('VA', math.nan):.3g}, Newton oracle "
        f"{flo.get(f'case118_nk{k}', {}).get('nr oracle', {}).get('VA', math.nan):.2g}" for k in KS) + "\n")
    md += ["| arm | k | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |", "|---" * 9 + "|"]
    for arm in ARMS:
        for k in (0,) + KS:
            if (arm, k, "VA") not in arms:
                continue
            cells = [rng(arms[arm, k, q]) for q in ("VA", "VA_cm", "VA_diff", "dVA", "VM_PQ", "PBE")]
            ratio = rng(arms[arm, k, "VA"] / dc[k]) if k else "—"
            md.append(f"| {arm} | {'in-dist' if k == 0 else k} | " + " | ".join(cells) + f" | {ratio} |")
    for size in SIZES:
        for k in KS:
            if (size, k, "VA") not in rel:
                continue
            cells = [f"{rel[size, k, q]:.3g}" for q in ("VA", "VA_cm", "VA_diff", "dVA", "VM_PQ", "PBE")]
            md.append(f"| GENCO {size} | {k} | " + " | ".join(cells) + f" | {rel[size, k, 'VA'] / dc[k]:.2f} |")
    md.append("")

    def every(cond):
        return bool(len(cond) and all(cond))

    # K-P1: degradation with k is global (plain, released): offset(10)/offset(1) >= 2, dVA(10)/dVA(1) < 1.5
    lines = []
    ok = []
    for arm in ("plain",):
        if (arm, 10, "VA_cm") in arms and (arm, 1, "VA_cm") in arms:
            ro, rd = arms[arm, 10, "VA_cm"] / arms[arm, 1, "VA_cm"], arms[arm, 10, "dVA"] / arms[arm, 1, "dVA"]
            ok.append(every(ro >= 2) and every(rd < 1.5))
            lines.append(f"{arm} offset x{rng(ro)}, branch differences x{rng(rd)}")
    for size in SIZES:
        if (size, 10, "VA_cm") in rel and (size, 1, "VA_cm") in rel:
            ro, rd = rel[size, 10, "VA_cm"] / rel[size, 1, "VA_cm"], rel[size, 10, "dVA"] / rel[size, 1, "dVA"]
            ok.append(ro >= 2 and rd < 1.5)
            lines.append(f"GENCO {size} offset x{ro:.2f}, branch differences x{rd:.2f}")
    md.append(f"**K-P1 (degradation with k is global: offset(10)/offset(1) >= 2 and θ_f − θ_t grows < 1.5x, plain and released): "
              f"{('PASS' if all(ok) else 'FAIL') if len(ok) == 4 else 'incomplete'}** " + "; ".join(lines) + "\n")
    # K-P2: dccanon VA(10)/VA(1) <= 1.3 and (VA/DC)(10)/(VA/DC)(1) >= 0.8, every seed; plain's VA/DC ratio halves
    lines, ok = [], []
    if (("dccanon", 10, "VA") in arms) and (("dccanon", 1, "VA") in arms):
        r = arms["dccanon", 10, "VA"] / arms["dccanon", 1, "VA"]
        s = (arms["dccanon", 10, "VA"] / dc[10]) / (arms["dccanon", 1, "VA"] / dc[1])
        ok.append(every(r <= 1.3) and every(s >= 0.8))
        lines.append(f"dccanon VA x{rng(r)}, VA/DC x{rng(s)}")
    if (("plain", 10, "VA") in arms) and (("plain", 1, "VA") in arms):
        s = (arms["plain", 10, "VA"] / dc[10]) / (arms["plain", 1, "VA"] / dc[1])
        ok.append(every(s <= 0.5))
        lines.append(f"plain VA/DC x{rng(s)}")
    md.append(f"**K-P2 (the DC anchor removes the k-dependence: dccanon VA(10) <= 1.3 VA(1) and its VA/DC keeps >= 80%; plain's VA/DC halves): "
              f"{('PASS' if all(ok) else 'FAIL') if len(ok) == 2 else 'incomplete'}** " + "; ".join(lines) + "\n")
    # K-P3: dccanon <= 1.1 x DC at every k, every seed
    rows = [(k, arms["dccanon", k, "VA"] / dc[k]) for k in KS if ("dccanon", k, "VA") in arms]
    ok = [every(v <= 1.1) for _, v in rows]
    md.append(f"**K-P3 (dccanon <= 1.1 x DC at every k, every seed): {('PASS' if all(ok) else 'FAIL') if len(rows) == len(KS) else 'incomplete'}** "
              + "; ".join(f"k={k} x{rng(v)}" + (" ok" if o else "") for (k, v), o in zip(rows, ok)) + "\n")
    # K-P4: PH (regressor + fdpf) below 0.3 deg and 0.003 p.u. at every k, below every learned arm: rows 'ph' of CLASSICAL_nk_ph.txt
    ph = floors(a.results_dir / "CLASSICAL_nk_ph.txt") if (a.results_dir / "CLASSICAL_nk_ph.txt").exists() else {}
    rows = [(k, ph[f"case118_nk{k}"]["ph"]) for k in KS if f"case118_nk{k}" in ph and "ph" in ph[f"case118_nk{k}"]]
    ok = [v["VA"] < 0.3 and v["VM"] < 0.003 and all(v["VA"] < np.nanmin(arms[arm, k, "VA"]) for arm in ARMS if (arm, k, "VA") in arms)
          for k, v in rows]
    md.append(f"**K-P4 (PH: REF-voltage regressor + fast-decoupled below 0.3 deg and 0.003 p.u., below every learned arm, every k): "
              f"{('PASS' if all(ok) else 'FAIL') if len(rows) == len(KS) else 'incomplete'}** "
              + "; ".join(f"k={k} VA {v['VA']:.3g} VM {v['VM']:.2g}" + (" ok" if o else "") for (k, v), o in zip(rows, ok)) + "\n")
    # K-P5: gate: damped Newton oracle reproduces the labels at every k; released checkpoints on our k=1 within 10% of the official test
    rows = [(k, flo[f"case118_nk{k}"]["nr oracle"]["VA"]) for k in KS if f"case118_nk{k}" in flo and "nr oracle" in flo[f"case118_nk{k}"]]
    ok = [v <= 1e-5 for _, v in rows]
    official = {}
    for size in SIZES:
        f = a.results_dir.parent / "genco" / "g1" / f"case118_ieee_{size}_s0.json"
        if f.exists():
            official[size] = json.loads(f.read_text())["in_dist"]["VA"] * DEG
    g2 = [(size, rel[size, 1, "VA"], official[size]) for size in SIZES if (size, 1, "VA") in rel and size in official]
    ok2 = [abs(v - o) / o <= 0.10 for _, v, o in g2]
    md.append(f"**K-P5 (gate: Newton oracle VA <= 1e-5 deg at every k; released checkpoints on our k = 1 within 10% of their official-test VA): "
              f"{('PASS' if all(ok) and all(ok2) else 'FAIL') if len(rows) == len(KS) and len(g2) == 3 else 'incomplete'}** "
              + "; ".join(f"k={k} Newton {v:.2g}" for k, v in rows) + "; " + "; ".join(f"{s} {v:.3g} vs official {o:.3g}" for s, v, o in g2) + "\n")
    out = a.results_dir / "REPORT_k.md"
    out.write_text("\n".join(md))
    print(out.read_text())


if __name__ == "__main__":
    main()
