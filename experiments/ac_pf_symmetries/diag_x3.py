"""X3 diagnosis (outside the pre-registered pipeline): the injection-decoder arms collapse while training on
the 31 public subgrids (validation loss 0.15 -> >= 1.0, per-layer residual flat at ~7.4 from layer 4 on).

H1 precision: solve_anchored runs in float32, and kappa(L) * 6e-8 need not be small on the large subgrids.
H2 amplification: d theta / d c = L^-1, of norm 1/lambda_min, which grows with the grid's electrical size, so the
gradient reaching the injection head (and, through the physics feedback, the backbone) grows with the grid.

Per grid, one same-grid batch of 8 scenarios: n, lambda_min / lambda_max / kappa of the anchored block (float64, the
eps shift included), theta_DC float32 vs float64, and for the saved X3 checkpoints plain_seed0 and dcinjcanon_seed0
and dcinjcanon_seed2 (best on validation so far; seed 2 never left the bad basin) the loss, the gradient norm, and per layer max|c| and max|L^-1 c|.

usage: python -m experiments.ac_pf_symmetries.diag_x3 OUT.json GRID [GRID ...]   (on a GPU node)
env:   SYMFM_DATA_ROOT  node-local processed cache of every grid (built as in cluster/train_pt.sbatch)
Run of 2026-10-13 (job 63230, H100): results/pianox_pt/diag_x3.json.
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

GK = Path("/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/gridfm-graphkit")
sys.path.insert(0, str(GK))

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

import experiments.ac_pf_symmetries.multigrid as mg  # noqa: E402  (DATA_ROOT from SYMFM_DATA_ROOT)
from experiments.ac_pf_symmetries.dcprior import DCInjectionLayers, dc_context  # noqa: E402
from experiments.ac_pf_symmetries.hodge import solve_anchored  # noqa: E402
from experiments.ac_pf_symmetries.run import make_args  # noqa: E402
from experiments.ac_pf_symmetries.symmetries import Canonicalize, gns_of  # noqa: E402
from gridfm_graphkit.datasets.globals import VA_OUT  # noqa: E402
from gridfm_graphkit.io.param_handler import get_task  # noqa: E402

RUNS = GK / "experiments/ac_pf_symmetries/results/pianox_pt/heldout_case118_ieee"
PROBES = (("plain", 0), ("dcinjcanon", 0), ("dcinjcanon", 2))  # dcinjcanon_seed2 never left the bad basin
DEG = 180 / math.pi
DEV = torch.device("cuda")  # the CPU path hangs: MKL's batched float32 solve fails on 500-bus graphs (SLASWP error)


def anchored_block(ctx, k, eps=1e-6):
    """Graph k's anchored Laplacian (free block, eps * mean(w) shift as in solve_anchored), float64."""
    f, t, w, ref, batch = ctx["f"], ctx["t"], ctx["w"].double(), ctx["ref"], ctx["batch"]
    nodes = (batch == k).nonzero().squeeze(1)
    lo, n = int(nodes.min()), nodes.numel()
    sel = batch[f] == k
    fi, ti, wk = f[sel] - lo, t[sel] - lo, w[sel]
    lap = torch.zeros(n, n, dtype=torch.float64, device=w.device)
    for (i, j), s in (((fi, fi), 1.0), ((ti, ti), 1.0), ((fi, ti), -1.0), ((ti, fi), -1.0)):
        lap.index_put_((i, j), s * wk, accumulate=True)
    free = ~ref[nodes]
    return lap[free][:, free] + eps * w.mean() * torch.eye(int(free.sum()), dtype=torch.float64, device=w.device), n


def build(arm, seed, args, normalizer):
    task = get_task(args, [normalizer])
    if arm != "plain":
        task.model = Canonicalize(DCInjectionLayers(task.model))
    sd = torch.load(RUNS / f"{arm}_seed{seed}" / "model" / "best_model_state_dict.pt", map_location="cpu")
    missing, _ = task.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k.startswith("model.")], missing[:5]
    return task.to(DEV)


def probe(task, batch, inj):
    """Loss terms and gradient norm on one batch; for the injection arm, per layer max|c| and max|L^-1 c| (deg)."""
    rec = []
    hooks = []
    if inj:
        dec = gns_of(task.model).mlp_bus
        hooks.append(dec.mlp_bus.register_forward_hook(
            lambda m, i, o: rec.append({"c": float(o[:, VA_OUT].abs().max())})))
        hooks.append(dec.register_forward_hook(
            lambda m, i, o: rec[-1].update(shift_deg=float((o[:, VA_OUT] - m.ctx["theta_dc"]).abs().max() * DEG))))
    task.train()
    if hasattr(task.loss_fn, "set_epoch"):
        task.loss_fn.set_epoch(0)
    task.zero_grad()
    _, ld = task.shared_step(batch)
    ld["loss"].backward()
    for h in hooks:
        h.remove()
    g = torch.stack([p.grad.norm() for p in task.model.parameters() if p.grad is not None])
    return {
        "loss": {k: float(v) for k, v in ld.items() if torch.is_tensor(v) and v.numel() == 1},
        "grad_norm": float(g.norm()),
        "layers": rec,
    }


def main():
    out, grids = Path(sys.argv[1]), sys.argv[2:]
    a = SimpleNamespace(epochs=1, batch_size=8, patience=1, layers=12, hidden=48)
    args = make_args(a)
    res = {}
    for g in grids:
        s = mg.grid_splits(g, args, 0, 64)
        batch = next(iter(DataLoader(s["all"], batch_size=8))).to(DEV)
        ctx, p, theta_ref, conn = dc_context(batch)
        th32 = solve_anchored(p * conn, theta_ref=theta_ref, **ctx)
        th64 = solve_anchored((p * conn).double(), theta_ref=theta_ref.double(), **dict(ctx, w=ctx["w"].double()))
        err = (th32.double() - th64).abs()
        blk, n = anchored_block(ctx, 0)
        ev = torch.linalg.eigvalsh(blk)
        smooth = torch.linalg.solve(blk, torch.ones(blk.shape[0], dtype=torch.float64, device=DEV)).abs().max()
        r = {
            "n": n,
            "lambda_min": float(ev[0]),
            "lambda_max": float(ev[-1]),
            "kappa": float(ev[-1] / ev[0]),
            "inv_lambda_min": float(1 / ev[0]),
            "smooth_gain_deg": float(smooth * DEG),
            "w_mean": float(ctx["w"].mean()),
            "theta_max_deg": float((th64 - theta_ref.double()[ctx["batch"]]).abs().max() * DEG),
            "dc32_err_max_deg": float(err.max() * DEG),
            "dc32_err_med_deg": float(err.median() * DEG),
        }
        for arm, seed in PROBES:
            r[f"{arm}_seed{seed}"] = probe(build(arm, seed, args, s["normalizer"]), batch, arm != "plain")
        res[g] = r
        P, G, B = r["plain_seed0"], r["dcinjcanon_seed0"], r["dcinjcanon_seed2"]
        print(f"{g:34s} n={n:5d} 1/lmin={r['inv_lambda_min']:9.3g} kappa={r['kappa']:9.3g} "
              f"dc32err={r['dc32_err_max_deg']:8.2g}deg thmax={r['theta_max_deg']:6.1f} | "
              f"grad plain/inj0/inj2={P['grad_norm']:7.3g}/{G['grad_norm']:7.3g}/{B['grad_norm']:7.3g} | "
              f"loss={P['loss']['loss']:6.3g}/{G['loss']['loss']:6.3g}/{B['loss']['loss']:6.3g} | "
              f"L11 c={G['layers'][-1]['c']:6.3g}/{B['layers'][-1]['c']:6.3g} "
              f"shift={G['layers'][-1]['shift_deg']:7.3g}/{B['layers'][-1]['shift_deg']:7.3g}deg", flush=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
