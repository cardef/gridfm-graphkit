"""Aggregate results/<case>/<variant>_seed*/result.json into poster tables and figures.

python -m experiments.ac_pf_symmetries.report --case case14_ieee
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
LABEL = {
    "m0": "M0 (baseline)",
    "aug": "M0+Aug wide (k 0.1-10, α ±π)",
    "augmild": "M0+Aug mild (k 0.5-2, α ±0.5)",
    "canon": "M-canon (exact S1+S3)",
}
DEG = 180.0 / math.pi


def load(case):
    runs = defaultdict(list)
    for f in sorted((OUT / case).glob("*_seed*/result.json")):
        r = json.load(open(f))
        runs[r["variant"]].append(r)
    return runs


def ms(values):
    a = np.asarray(values, dtype=float)
    return f"{a.mean():.3g} ± {a.std():.2g}" if len(a) > 1 else f"{a.mean():.3g}"


def physical(acc, base_mva):
    return {
        "VM (p.u.)": acc["VM"],
        "VA (deg)": acc["VA"] * DEG,
        "PG (MW)": acc["PG"] * base_mva,
        "QG (Mvar)": acc["QG"] * base_mva,
        "PBE (MVA)": acc["PBE"] * base_mva,
    }


def table_accuracy(runs, pick, title):
    cols = ["VM (p.u.)", "VA (deg)", "PG (MW)", "QG (Mvar)", "PBE (MVA)"]
    lines = [
        f"### {title}",
        "",
        "| model | " + " | ".join(cols) + " | epochs |",
        "|---" * (len(cols) + 2) + "|",
    ]
    for v, rs in runs.items():
        phys = [physical(*pick(r)) for r in rs]
        cells = [ms([p[c] for p in phys]) for c in cols]
        lines.append(
            f"| {LABEL[v]} | "
            + " | ".join(cells)
            + f" | {ms([r['epochs_run'] for r in rs])} |",
        )
    return "\n".join(lines) + "\n"


def audit_rows(r, sym, param):
    return next(
        a for a in r["audit"] if a["sym"] == sym and abs(a["param"] - param) < 1e-9
    )


def table_ee(runs, probes):
    lines = ["### Equivariance error EE_g (relative, on predicted entries)", ""]
    lines.append(
        "| model | " + " | ".join(f"{s} {p:g} ({q})" for s, p, q in probes) + " |",
    )
    lines.append("|---" * (len(probes) + 1) + "|")
    for v, rs in runs.items():
        cells = [
            ms([audit_rows(r, s, p)["ee"][q]["rel"] for r in rs]) for s, p, q in probes
        ]
        lines.append(f"| {LABEL[v]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def figure(runs, case):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for v, rs in runs.items():
        phases = sorted({a["param"] for a in rs[0]["audit"] if a["sym"] == "phase"})
        scales = sorted({a["param"] for a in rs[0]["audit"] if a["sym"] == "scale"})
        ee_ph = np.array(
            [
                [audit_rows(r, "phase", p)["ee"]["VA"]["rel"] for p in phases]
                for r in rs
            ],
        )
        ee_sc = np.array(
            [
                [audit_rows(r, "scale", k)["ee"]["VM"]["rel"] for k in scales]
                for r in rs
            ],
        )
        err_sc = np.array(
            [
                [audit_rows(r, "scale", k)["acc"]["VA"] * DEG for k in scales]
                for r in rs
            ],
        )
        for ax, x, y in (
            (axes[0], phases, ee_ph),
            (axes[1], scales, ee_sc),
            (axes[2], scales, err_sc),
        ):
            m, s = y.mean(0), y.std(0)
            ax.errorbar(
                x,
                np.maximum(m, 1e-9),
                yerr=s,
                marker="o",
                capsize=3,
                label=LABEL[v],
            )
    axes[0].set(
        xlabel="phase shift α (rad)",
        ylabel="EE_S1 on VA (relative)",
        yscale="log",
        title="S1: global phase",
    )
    axes[1].set(
        xlabel="MVA base factor k",
        ylabel="EE_S3 on VM (relative)",
        xscale="log",
        yscale="log",
        title="S3: scale",
    )
    axes[2].set(
        xlabel="MVA base factor k",
        ylabel="VA RMSE on T_k (deg)",
        xscale="log",
        yscale="log",
        title="Accuracy on transformed test set",
    )
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{case}: exact symmetries vs. augmentation vs. baseline (mean ± std over seeds)",
    )
    fig.tight_layout()
    path = OUT / case / "figures"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / "ee_audit.png", dpi=200)
    fig.savefig(path / "ee_audit.pdf")
    return path / "ee_audit.png"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="case14_ieee")
    a = p.parse_args()
    runs = load(a.case)
    if not runs:
        raise SystemExit(f"no results under {OUT / a.case}")
    md = [f"# AC-PF symmetry audit — trained on {a.case}", ""]
    md.append(
        f"Seeds per model: {{ {', '.join(f'{LABEL[v]}: {len(rs)}' for v, rs in runs.items())} }}\n",
    )
    md.append(
        table_accuracy(
            runs,
            lambda r: (r["in_dist"], r["baseMVA"]),
            "In-distribution test RMSE",
        ),
    )
    md.append(
        table_ee(
            runs,
            [
                ("phase", math.pi, "VA"),
                ("phase", 0.1, "VA"),
                ("scale", 10.0, "VM"),
                ("scale", 100.0, "VM"),
                ("scale", 0.01, "VM"),
                ("flip", 0.5, "VA"),
            ],
        ),
    )
    for case in runs[next(iter(runs))][0]["zero_shot"]:
        md.append(
            table_accuracy(
                runs,
                lambda r, c=case: (
                    r["zero_shot"][c]["acc"],
                    r["zero_shot"][c]["baseMVA"],
                ),
                f"Zero-shot test RMSE on {case} (never seen in training)",
            ),
        )
    fig = figure(runs, a.case)
    md.append(f"![EE audit]({fig.relative_to(OUT / a.case)})\n")
    out = OUT / a.case / "REPORT.md"
    out.write_text("\n".join(md))
    print(out.read_text())
    print(f"figure: {fig}")


if __name__ == "__main__":
    main()
