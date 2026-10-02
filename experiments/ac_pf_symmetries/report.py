"""Aggregate results/<case>/<variant>_seed*/result.json into poster tables and figures.

python -m experiments.ac_pf_symmetries.report --case case14_ieee [--result-file result_eval.json]
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
    "augphase": "M0+Aug phase only (α ±0.5)",
    "augscale": "M0+Aug scale only (k 0.5-2)",
    "augcov": "M0+Aug wide, loss in the sample frame (k 0.1-10, α ±π)",
    "augcovphase": "M0+Aug phase wide (α ±π), loss in the sample frame",
    "augcovscale": "M0+Aug scale wide (k 0.1-10), loss in the sample frame",
    "augnophys": "M0+Aug wide, physics loss weight 0",
    "m0nophys": "M0, physics loss weight 0",
    "m0warm": "M0 on canon's RNG path (one loader pass first)",
    "canon": "M-canon (canonicalized S1+S3+S4)",
    "canonmix": "M-canon, S3 frame P95^0.5 · mean|Y|^0.5",
    "canonp95": "M-canon, S3 frame P95 of input injections",
    "m1canon": "M1 + canon: branch differences, Hodge reconstruction",
}
DEG = 180.0 / math.pi


def load(root, result_file):
    runs = defaultdict(list)
    for f in sorted(root.glob(f"*_seed*/{result_file}")):
        r = json.load(open(f))
        runs[r["variant"]].append(r)
    return dict(sorted(runs.items(), key=lambda kv: list(LABEL).index(kv[0])))


def table_angles(runs):
    """R7: is a VA gain a common offset (reference routing) or also in the branch angles?"""
    cols = {
        "VA": "VA (deg)",
        "VA_cm": "VA common offset (deg)",
        "VA_diff": "VA minus offset (deg)",
        "dVA": "θ_f − θ_t (deg)",
    }
    lines = [
        "### In-distribution angle error split (R7): VA² = offset² + rest²",
        "",
        "| model | " + " | ".join(cols.values()) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for v, rs in runs.items():
        cells = [
            ms([r["in_dist"][q] * DEG if q in r["in_dist"] else None for r in rs])
            for q in cols
        ]
        lines.append(f"| {LABEL[v]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ms(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return "n/a"
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
    """The audit entry for (sym, param), or None (results older than the `fliptrafo` probe)."""
    return next(
        (a for a in r["audit"] if a["sym"] == sym and abs(a["param"] - param) < 1e-9),
        None,
    )


def ee_ratio(r, sym, param, q):
    """EE_g in units of the model's own in-distribution RMSE on the same channel (Piano A §5).

    Under S3 the PG/QG difference is measured in the transformed frame (powers / k); x k brings it back to the
    frame of the RMSE (without it, an exact model at float precision reads 1e-2 at k = 0.01).
    """
    a = audit_rows(r, sym, param)
    if a is None:
        return None
    frame = param if sym == "scale" and q in ("PG", "QG") else 1.0
    return a["ee"][q]["rmse"] * frame / r["in_dist"][q]


def table_ee(runs, probes):
    lines = [
        "### Equivariance error EE_g / in-distribution RMSE, same channel (> 1: symmetry breaking dominates)",
        "",
    ]
    lines.append(
        "| model | " + " | ".join(f"{s} {p:g} ({q})" for s, p, q in probes) + " |",
    )
    lines.append("|---" * (len(probes) + 1) + "|")
    for v, rs in runs.items():
        cells = [ms([ee_ratio(r, s, p, q) for r in rs]) for s, p, q in probes]
        lines.append(f"| {LABEL[v]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def figure(runs, case, root, suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for v, rs in runs.items():
        phases = sorted({a["param"] for a in rs[0]["audit"] if a["sym"] == "phase"})
        scales = sorted({a["param"] for a in rs[0]["audit"] if a["sym"] == "scale"})
        ee_ph = np.array([[ee_ratio(r, "phase", p, "VA") for p in phases] for r in rs])
        ee_sc = np.array([[ee_ratio(r, "scale", k, "VM") for k in scales] for r in rs])
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
        ylabel="EE_S1 on VA / in-dist RMSE",
        yscale="log",
        title="S1: global phase",
    )
    axes[1].set(
        xlabel="MVA base factor k",
        ylabel="EE_S3 on VM / in-dist RMSE",
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
    path = root / "figures"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"ee_audit{suffix}.png", dpi=200)
    fig.savefig(path / f"ee_audit{suffix}.pdf")
    return path / f"ee_audit{suffix}.png"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="case14_ieee")
    p.add_argument(
        "--result-file",
        default="result.json",
        help="result_eval.json: run.py --eval-only output",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=OUT,
        help="as in run.py (results/abacus for the cluster replicate)",
    )
    a = p.parse_args()
    root = a.results_dir / a.case
    runs = load(root, a.result_file)
    if not runs:
        raise SystemExit(f"no {a.result_file} under {root}")
    suffix = (
        ""
        if a.result_file == "result.json"
        else "_" + Path(a.result_file).stem.removeprefix("result_")
    )
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
    md.append(table_angles(runs))
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
                ("fliptrafo", 1.0, "VM"),
            ],
        ),
    )
    # results written before the zero-shot fix have no field: they refit the normalizer on the target
    norms = "/".join(
        sorted({r.get("zero_shot_norm", "refit") for rs in runs.values() for r in rs}),
    )
    for case in runs[next(iter(runs))][0]["zero_shot"]:
        md.append(
            table_accuracy(
                runs,
                lambda r, c=case: (
                    r["zero_shot"][c]["acc"],
                    r["zero_shot"][c]["baseMVA"],
                ),
                f"Zero-shot test RMSE on {case} (never seen in training; normalizer: {norms})",
            ),
        )
    fig = figure(runs, a.case, root, suffix)
    md.append(f"![EE audit]({fig.relative_to(root)})\n")
    out = root / f"REPORT{suffix}.md"
    out.write_text("\n".join(md))
    print(out.read_text())
    print(f"figure: {fig}")


if __name__ == "__main__":
    main()
