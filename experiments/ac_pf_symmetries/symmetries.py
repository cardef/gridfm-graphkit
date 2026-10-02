"""Exact symmetries of the AC power-flow map as actions on GridFM hetero batches.

Actions operate on *normalized, masked* data (what the model sees) and on the
model output layout ``bus=[VM, VA, PG, QG]``, ``gen=[PG]``.

S1 phase : theta_i -> theta_i + alpha (per graph). Flows invariant.
S3 scale : new MVA base = k * old base: every power-like and admittance-like
           quantity divides by k, voltages unchanged.
S4 flip  : re-declare from/to of a branch. Each directed row stores its own
           (flow, Y_self, Y_mutual), so swapping the two rows already applies the
           transformer rule (tau, phi, z, b) -> (1/tau, -phi, tau^2 z, b/tau^2); the
           columns stored in the from->to convention on both rows transform too
           (tap -> 1/tap, angle limits negated and swapped). For lines it is a pure
           row permutation, exact by construction. AC branch flows swap
           (S_ft <-> S_tf), they do not negate: S_ft + S_tf are the losses.
S2 (local gauge) acts on any grid (it creates virtual phase shifts on lines), but on
data without phase shifters its constraint reduces to S1 (Piano A, Prop. 6): not implemented.
"""

import math

import torch
from torch import nn
from torch_geometric.transforms import BaseTransform

from gridfm_graphkit.datasets.globals import (
    ANG_MAX,
    ANG_MIN,
    BS,
    GS,
    MAX_PG,
    MAX_QG_H,
    MIN_PG,
    MIN_QG_H,
    P_E,
    PD_H,
    PG_H,
    PG_OUT,
    PG_OUT_GEN,
    Q_E,
    QD_H,
    QG_H,
    QG_OUT,
    RATE_A,
    TAP,
    VA_H,
    VA_OUT,
    VM_H,
    VM_OUT,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)

E = ("bus", "connects", "bus")
BUS_POWER = [PD_H, QD_H, QG_H, MIN_QG_H, MAX_QG_H, GS, BS]
BUS_Y_POWER = [PD_H, QD_H, QG_H]
GEN_POWER = [PG_H, MIN_PG, MAX_PG]
EDGE_POWER = [P_E, Q_E, YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I, RATE_A]
SYMMETRIES = ("phase", "scale", "flip", "fliptrafo")


def num_graphs(data):
    return getattr(data, "num_graphs", 1)


def node_batch(data, node_type):
    store = data[node_type]
    b = getattr(store, "batch", None)
    if b is None:
        return torch.zeros(store.x.size(0), dtype=torch.long, device=store.x.device)
    return b


def edge_batch(data):
    return node_batch(data, "bus")[data[E].edge_index[0]]


def _per_graph(param, data):
    p = torch.as_tensor(param, dtype=torch.float32, device=data["bus"].x.device)
    return p.expand(num_graphs(data)) if p.dim() == 0 else p


def act_phase(data, alpha):
    d = data.clone()
    a = _per_graph(alpha, d)[node_batch(d, "bus")]
    known = ~d.mask_dict["bus"][:, VA_H]
    d["bus"].x[:, VA_H] += torch.where(known, a, torch.zeros_like(a))
    d["bus"].y[:, VA_H] += a
    return d


def act_scale(data, k):
    d = data.clone()
    k = _per_graph(k, d)
    kb, kg, ke = k[node_batch(d, "bus")], k[node_batch(d, "gen")], k[edge_batch(d)]
    d["bus"].x[:, BUS_POWER] /= kb[:, None]
    d["bus"].y[:, BUS_Y_POWER] /= kb[:, None]
    d["gen"].x[:, GEN_POWER] /= kg[:, None]
    d["gen"].y[:, [PG_H]] /= kg[:, None]
    d[E].edge_attr[:, EDGE_POWER] /= ke[:, None]
    d[E].y /= ke[:, None]
    if hasattr(d, "baseMVA"):
        d.baseMVA = d.baseMVA * k
    return d


def line_pairs(data):
    """Return (partner_row, is_forward, is_line) for the bidirectional edge layout."""
    ei = data[E].edge_index
    eb = edge_batch(data)
    counts = torch.bincount(eb, minlength=num_graphs(data))
    assert (counts % 2 == 0).all(), "edge layout is not [forward..., reverse...]"
    half = counts // 2
    starts = torch.cumsum(counts, 0) - counts
    j = torch.arange(ei.size(1), device=ei.device)
    is_fwd = (j - starts[eb]) < half[eb]
    partner = torch.where(is_fwd, j + half[eb], j - half[eb])
    assert (ei[:, j] == ei.flip(0)[:, partner]).all(), "fwd/rev rows do not pair"
    is_line = data[E].edge_attr[:, TAP] == 1
    return partner, is_fwd, is_line


def _flip_rows(data, flip):
    """Re-declare from/to of the branches whose forward row is set in `flip` (see S4 above)."""
    d = data.clone()
    partner, _, _ = line_pairs(d)
    flip = flip | flip[partner]
    perm = torch.where(
        flip,
        partner,
        torch.arange(partner.numel(), device=partner.device),
    )
    d[E].edge_index = d[E].edge_index[:, perm]
    d[E].edge_attr = d[E].edge_attr[perm]
    d[E].y = d[E].y[perm]
    if (
        "branch" in d.mask_dict
    ):  # row-aligned; rebuilt, not mutated: clone() shares the dict
        d.mask_dict = {**d.mask_dict, "branch": d.mask_dict["branch"][perm]}
    a = d[E].edge_attr
    a[flip, TAP] = 1.0 / a[flip, TAP]
    a[flip, ANG_MIN], a[flip, ANG_MAX] = -a[flip, ANG_MAX], -a[flip, ANG_MIN]
    return d


def act_flip(data, p=0.5, seed=0, branches="lines"):
    """S4 on a random subset (probability p) of the lines, the transformers or all branches."""
    partner, is_fwd, is_line = line_pairs(data)
    kind = {"lines": is_line, "trafos": ~is_line, "all": torch.ones_like(is_line)}[
        branches
    ]
    gen = torch.Generator().manual_seed(seed)
    draw = (torch.rand(partner.numel(), generator=gen) < p).to(partner.device)
    return _flip_rows(data, draw & is_fwd & kind)


def orient_trafos(data):
    """S4 frame: declare every transformer with tap <= 1 (identity on case14: taps 0.93-0.98)."""
    _, is_fwd, is_line = line_pairs(data)
    return _flip_rows(data, is_fwd & ~is_line & (data[E].edge_attr[:, TAP] > 1))


def act(data, sym, param):
    if sym == "phase":
        return act_phase(data, param)
    if sym == "scale":
        return act_scale(data, param)
    if sym == "flip":
        return act_flip(data, p=param)
    if sym == "fliptrafo":
        return act_flip(data, p=param, branches="trafos")
    raise ValueError(sym)


def act_output(out, sym, param, data):
    """rho(g) applied to model outputs (or to a bus target of the same layout)."""
    o = {k: v.clone() for k, v in out.items()}
    if sym == "phase":
        o["bus"][:, VA_OUT] += _per_graph(param, data)[node_batch(data, "bus")]
    elif sym == "scale":
        k = _per_graph(param, data)
        o["bus"][:, [PG_OUT, QG_OUT]] /= k[node_batch(data, "bus")][:, None]
        if "gen" in o:
            o["gen"][:, [PG_OUT_GEN]] /= k[node_batch(data, "gen")][:, None]
    elif sym not in ("flip", "fliptrafo"):
        raise ValueError(sym)
    return o


def wrap_angle(x):
    """Angles live on the circle: map a difference to [-pi, pi)."""
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def pred_masks(data):
    """Bus output entries the PF model actually predicts: (column, mask)."""
    mb = data.mask_dict["bus"]
    pv, ref = data.mask_dict["PV"], data.mask_dict["REF"]
    return {
        "VM": (VM_OUT, mb[:, VM_H]),
        "VA": (VA_OUT, mb[:, VA_H]),
        "PG": (PG_OUT, ref),
        "QG": (QG_OUT, pv | ref),
    }


@torch.no_grad()
def equivariance_error(model, loader, sym, param, device="cpu"):
    """EE_g(f) = ||f(g u) - rho(g) f(u)||, per predicted quantity, over a loader.

    `rmse`: absolute, in the output's own unit (VA differences wrapped to [-pi, pi)); Piano A §5 reports it
    relative to the in-distribution RMSE of the same channel (report.py). `rel`: divided by ||rho(g) f(u)||
    (the *transformed* output): for S1 it saturates near 1 for a model that ignores alpha.
    """
    acc = {q: torch.zeros(3, dtype=torch.float64) for q in ("VM", "VA", "PG", "QG")}
    for data in loader:
        data = data.to(device)
        out = model(data)
        out_g = model(act(data, sym, param))
        expected = act_output(out, sym, param, data)
        for q, (col, m) in pred_masks(data).items():
            diff = (out_g["bus"][m, col] - expected["bus"][m, col]).cpu().double()
            if q == "VA":
                diff = wrap_angle(diff)
            ref = expected["bus"][m, col].cpu().double()
            acc[q] += torch.stack(
                [(diff**2).sum(), (ref**2).sum(), m.sum().cpu().double()],
            )
    return {
        q: {
            "rmse": math.sqrt(a[0] / a[2]),
            "rel": math.sqrt(a[0] / a[1]) if a[1] > 0 else float("nan"),
        }
        for q, a in acc.items()
    }


class Canonicalize(nn.Module):
    """Exact S1+S3+S4 equivariance by canonicalization around any bus/gen PF model.

    A frame per symmetry, computed from the inputs only (Piano A, Prop. 5), so the
    class of functions is exactly the equivariant one. Orientation: declare every
    transformer with tap <= 1 (outputs are bus quantities, invariant: nothing to
    undo). Phase: subtract each graph's reference angle from the inputs, add it back
    to the predicted angles. Scale: divide power/admittance inputs by a degree-1
    homogeneous statistic of the inputs, s_a = P95(|input injections|)^a *
    mean(|Yff|)^(1-a) per graph (Piano A H4: every a in [0, 1] is an exact frame;
    a = 0 is the PoC's), relative to a constant fitted on the training set so the
    in-distribution scale is unchanged on average; multiply predicted powers back.
    """

    def __init__(self, model, phase=True, scale=True, orient=True, scale_a=0.0):
        super().__init__()
        assert 0.0 <= scale_a <= 1.0
        self.model = model
        self.phase = phase
        self.scale = scale
        self.orient = orient
        self.scale_a = scale_a
        self.register_buffer("scale_ref", torch.ones(()))

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    @staticmethod
    def admittance_scale(data):
        eb = edge_batch(data)
        y = data[E].edge_attr[:, [YFF_TT_R, YFF_TT_I]].norm(dim=1)
        g = num_graphs(data)
        total = torch.zeros(g, device=y.device).index_add_(0, eb, y)
        return total / torch.bincount(eb, minlength=g).clamp(min=1)

    @staticmethod
    def injection_scale(data):
        """P95 of the nonzero |Pd|, |Qd| (every bus) and |Pg| (generators whose Pg is an input), per graph.

        The framework's normalizer takes the same quantile but also reads Qg and the slack Pg, which are PF
        outputs; this one reads inputs only.
        """
        bus = data["bus"].x[:, [PD_H, QD_H]].abs()
        known = ~data.mask_dict["gen"][:, PG_H]
        v = torch.cat([bus.reshape(-1), data["gen"].x[known, PG_H].abs()])
        b = torch.cat(
            [node_batch(data, "bus").repeat_interleave(2), node_batch(data, "gen")[known]],
        )
        v, b = v[v > 0], b[v > 0]
        return torch.stack(
            [torch.quantile(v[b == i], 0.95) for i in range(num_graphs(data))],
        )

    def graph_scale(self, data):
        a = self.scale_a
        if a == 0.0:
            return self.admittance_scale(data)
        if a == 1.0:
            return self.injection_scale(data)
        return self.injection_scale(data) ** a * self.admittance_scale(data) ** (1 - a)

    @staticmethod
    def ref_angle(data):
        ref = data.mask_dict["REF"]
        g = num_graphs(data)
        assert int(ref.sum()) == g, "expected exactly one REF bus per graph"
        theta = torch.zeros(g, device=ref.device)
        theta[node_batch(data, "bus")[ref]] = data["bus"].x[ref, VA_H]
        return theta

    @torch.no_grad()
    def fit_scale_ref(self, loader):
        scales = torch.cat([self.graph_scale(d) for d in loader])
        self.scale_ref.fill_(scales.mean())
        return self

    def forward(self, data, return_embeddings=False):
        d = orient_trafos(data) if self.orient else data
        if self.phase:
            theta = self.ref_angle(d)
            d = act_phase(d, -theta)
        if self.scale:
            s = self.graph_scale(d) / self.scale_ref
            d = act_scale(d, s)
        out = self.model(d, return_embeddings=return_embeddings)
        if return_embeddings:
            out, emb = out
        if self.scale:
            out = act_output(out, "scale", 1.0 / s, d)
        if self.phase:
            out = act_output(out, "phase", theta, d)
        return (out, emb) if return_embeddings else out


class RandomSymmetryAugment(BaseTransform):
    """Training-time data augmentation with random S1 phase and S3 scale actions.

    The loss then sees the augmented sample: power residuals and PG/QG errors scale
    like 1/k, so their weight drifts across samples (why the wide `aug` arm fails;
    `CovariantAugment` is the fair baseline).
    """

    def __init__(self, alpha_max=math.pi, k_range=(0.1, 10.0)):
        super().__init__()
        self.alpha_max = alpha_max
        self.log_k = (math.log(k_range[0]), math.log(k_range[1]))

    def forward(self, data):
        alpha = (torch.rand(()) * 2 - 1) * self.alpha_max
        k = torch.exp(self.log_k[0] + torch.rand(()) * (self.log_k[1] - self.log_k[0]))
        return act_scale(act_phase(data, alpha), k)


class CovariantAugment(nn.Module):
    """M0+Aug with every loss term in the sample's own frame (Piano A §6).

    Training only: act on the batch with g = (alpha ~ U(-alpha_max, alpha_max) per
    graph, one k ~ logU(k_range) per batch), run the model, undo rho(g) on its
    outputs and multiply its physics residuals by k (a power residual scales like
    1/k under S3). The task's losses therefore compare de-augmented predictions
    with the original labels and physics. k is shared by the batch because the
    model stores one residual scalar per layer (mean over the batch's buses).
    Identity at evaluation.
    """

    def __init__(self, model, alpha_max=math.pi, k_range=(0.1, 10.0)):
        super().__init__()
        self.model = model
        self.alpha_max = alpha_max
        self.log_k = (math.log(k_range[0]), math.log(k_range[1]))

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    def draw(self, n_graphs):
        alpha = (torch.rand(n_graphs) * 2 - 1) * self.alpha_max
        k = math.exp(
            self.log_k[0] + float(torch.rand(())) * (self.log_k[1] - self.log_k[0]),
        )
        return alpha, k

    def forward(self, data, return_embeddings=False):
        if not self.training:
            return self.model(data, return_embeddings=return_embeddings)
        alpha, k = self.draw(num_graphs(data))
        out = self.model(
            act_scale(act_phase(data, alpha), k),
            return_embeddings=return_embeddings,
        )
        if return_embeddings:
            out, emb = out
        res = self.model.layer_residuals
        for i in res:
            res[i] = res[i] * k
        out = act_output(act_output(out, "scale", 1.0 / k, data), "phase", -alpha, data)
        return (out, emb) if return_embeddings else out
