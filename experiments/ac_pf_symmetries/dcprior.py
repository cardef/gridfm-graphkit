"""DC prior (R2): every layer of GNS decodes its bus angles as the DC power-flow solution plus a learned correction.

theta_DC solves B theta = P on the non-REF buses, with B_ft = Im Y_ft on the forward rows, P = the Pg of the
generators whose Pg is an input minus Pd, and theta fixed at the REF bus: one anchored Laplacian solve per graph
(`solve_anchored`), from the inputs only, so it needs no gradient and is exact under S1 (anchored), S3 (B and P
scale together), S4 and S5. It is the DC of `r2_baselines.py`, the reference the zero-shot errors are compared
with. As in `BranchAngleLayers`, the decoder `mlp_bus` is wrapped, so the per-layer physics feedback acts on
DC + correction.

`DCInjectionLayers` reads the network's angle output as a per-bus injection correction c instead and turns it into
angles with the same anchored solve: theta = theta_DC + L^-1 c, so the learned part is local and its global effect
(the common shift relative to REF included) comes out of the exact solve. Buses not connected to their graph's REF
bus (an island after an outage) get no injection, so they stay at the REF angle instead of blowing up.
"""

import torch
from torch import nn

from gridfm_graphkit.datasets.globals import PD_H, PG_H, VA_H, VA_OUT, YFT_TF_I

from .hodge import solve_anchored
from .symmetries import E, line_pairs, node_batch, num_graphs


def ref_connected(f, t, ref):
    """Buses connected to their graph's REF bus through the branches f -> t."""
    reach = ref.clone()
    while True:
        new = reach.clone()
        new[t[reach[f]]] = True
        new[f[reach[t]]] = True
        if torch.equal(new, reach):
            return reach
        reach = new


def dc_context(data):
    """The anchored DC system of a batch: branches, weights Im Y_ft, REF, injections P and the REF-connected buses."""
    _, fwd, _ = line_pairs(data)
    ei, ea = data[E].edge_index, data[E].edge_attr
    gi, gb = data.edge_index_dict[("gen", "connected_to", "bus")]
    known = (~data.mask_dict["gen"][gi, PG_H]).to(ea.dtype)
    p = torch.zeros(data["bus"].x.size(0), dtype=ea.dtype, device=ea.device)
    p = p.index_add(0, gb, data["gen"].x[gi, PG_H] * known) - data["bus"].x[:, PD_H]
    ref, batch, n_graphs = data.mask_dict["REF"], node_batch(data, "bus"), num_graphs(data)
    theta_ref = ea.new_zeros(n_graphs)
    theta_ref[batch[ref]] = data["bus"].x[ref, VA_H]
    f, t = ei[0, fwd], ei[1, fwd]
    ctx = {"f": f, "t": t, "w": ea[fwd, YFT_TF_I], "ref": ref, "batch": batch, "n_graphs": n_graphs}
    return ctx, p, theta_ref, ref_connected(f, t, ref)


def dc_angles(data):
    """theta_DC per bus (module docstring)."""
    ctx, p, theta_ref, conn = dc_context(data)
    return solve_anchored(p * conn, theta_ref=theta_ref, **ctx)


def _zero_angle_row(mlp_bus):
    """Start the decoder's angle column at exactly zero (weights and bias of that output row)."""
    last = [m for m in mlp_bus.modules() if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        last.weight[VA_OUT].zero_()
        last.bias[VA_OUT].zero_()


class _DCDecoder(nn.Module):
    """GNS's bus decoder with theta_DC added to its angle column (bound per forward by DCPriorLayers)."""

    def __init__(self, mlp_bus):
        super().__init__()
        self.mlp_bus = mlp_bus
        self.theta_dc = None

    def forward(self, h):
        out = self.mlp_bus(h)
        shift = torch.zeros_like(out)
        shift[:, VA_OUT] = self.theta_dc
        return out + shift


class DCPriorLayers(nn.Module):
    """GNS whose every layer predicts the angles as theta_DC + correction (zero_init: the correction starts at 0)."""

    def __init__(self, model, zero_init=False):
        super().__init__()
        self.model = model
        if zero_init:
            _zero_angle_row(model.mlp_bus)
        model.mlp_bus = _DCDecoder(model.mlp_bus)

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    def forward(self, data, return_embeddings=False):
        dec = self.model.mlp_bus
        with torch.no_grad():
            dec.theta_dc = dc_angles(data)
        try:
            return self.model(data, return_embeddings=return_embeddings)
        finally:
            dec.theta_dc = None


class _DCInjectionDecoder(nn.Module):
    """GNS's bus decoder whose angle column is a per-bus injection correction c: angles theta_DC + L^-1 c."""

    def __init__(self, mlp_bus):
        super().__init__()
        self.mlp_bus = mlp_bus
        self.ctx = None  # bound per forward by DCInjectionLayers

    def forward(self, h):
        out = self.mlp_bus(h)
        c = self.ctx
        shift = solve_anchored(out[:, VA_OUT] * c["conn"], theta_ref=c["zero"], **c["system"])
        va = c["theta_dc"] + shift
        return torch.cat([out[:, :VA_OUT], va[:, None], out[:, VA_OUT + 1 :]], dim=1)


class DCInjectionLayers(nn.Module):
    """GNS whose every layer predicts the angles as theta_DC + L^-1 c, c a learned per-bus injection correction."""

    def __init__(self, model, zero_init=True):
        super().__init__()
        self.model = model
        if zero_init:
            _zero_angle_row(model.mlp_bus)
        model.mlp_bus = _DCInjectionDecoder(model.mlp_bus)

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    def forward(self, data, return_embeddings=False):
        dec = self.model.mlp_bus
        with torch.no_grad():
            system, p, theta_ref, conn = dc_context(data)
            theta_dc = solve_anchored(p * conn, theta_ref=theta_ref, **system)
        zero = theta_ref.new_zeros(theta_ref.shape)
        dec.ctx = {"system": system, "theta_dc": theta_dc, "conn": conn.to(p.dtype), "zero": zero}
        try:
            return self.model(data, return_embeddings=return_embeddings)
        finally:
            dec.ctx = None
