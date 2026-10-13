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

`gain` multiplies the network's output before the solve (X3a). The angles move by L^-1 times a change of c, and on the
public subgrids the smooth gain max|L^-1 1| reaches 6e4 deg per unit injection (median 6e3; 5e2 on case118), so one
Adam step of the decoder's own weights, which moves c by up to about lr * |h|_1 whatever the gradient's scale, can move
the angles by tens of degrees: gain = 0.01 makes those steps 100 times smaller. A plain attribute, not a buffer, so the
X3 checkpoints (gain 1) load unchanged.

`DCLoopLayers` closes the loop instead (Piano X): no learned angle. Every layer decodes theta_DC + L^-1 s, and after
the layer GENCO's own active-power residual (specified minus computed injection, the fast-decoupled mismatch dP) is
added to s: the P-theta half step of fast-decoupled with B' = L and V = 1, once per layer. The network's angle output
is unused; its voltage and generator outputs pass through, so the angles converge to the AC solution for the voltages
the network predicts, on any topology, and nothing learned is amplified by L^-1.
"""

import torch
from torch import nn

from gridfm_graphkit.datasets.globals import PD_H, PG_H, VA_H, VA_OUT, YFT_TF_I

from .hodge import solve_anchored
from .symmetries import E, gns_of, line_pairs, node_batch, num_graphs


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
        base = gns_of(model)
        if zero_init:
            _zero_angle_row(base.mlp_bus)
        base.mlp_bus = _DCDecoder(base.mlp_bus)

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    def forward(self, data, return_embeddings=False):
        dec = gns_of(self.model).mlp_bus
        with torch.no_grad():
            dec.theta_dc = dc_angles(data)
        try:
            return self.model(data, return_embeddings=return_embeddings)
        finally:
            dec.theta_dc = None


class _DCInjectionDecoder(nn.Module):
    """GNS's bus decoder whose angle column is a per-bus injection correction c: angles theta_DC + L^-1 c."""

    def __init__(self, mlp_bus, gain=1.0):
        super().__init__()
        self.mlp_bus = mlp_bus
        self.gain = gain
        self.ctx = None  # bound per forward by DCInjectionLayers

    def forward(self, h):
        out = self.mlp_bus(h)
        c = self.ctx
        shift = solve_anchored(self.gain * out[:, VA_OUT] * c["conn"], theta_ref=c["zero"], **c["system"])
        va = c["theta_dc"] + shift
        return torch.cat([out[:, :VA_OUT], va[:, None], out[:, VA_OUT + 1 :]], dim=1)


class DCInjectionLayers(nn.Module):
    """GNS whose every layer predicts the angles as theta_DC + L^-1 c, c a learned per-bus injection correction."""

    def __init__(self, model, zero_init=True, gain=1.0):
        super().__init__()
        self.model = model
        base = gns_of(model)
        if zero_init:
            _zero_angle_row(base.mlp_bus)
        base.mlp_bus = _DCInjectionDecoder(base.mlp_bus, gain)

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    def forward(self, data, return_embeddings=False):
        dec = gns_of(self.model).mlp_bus
        with torch.no_grad():
            system, p, theta_ref, conn = dc_context(data)
            theta_dc = solve_anchored(p * conn, theta_ref=theta_ref, **system)
        zero = theta_ref.new_zeros(theta_ref.shape)
        dec.ctx = {"system": system, "theta_dc": theta_dc, "conn": conn.to(p.dtype), "zero": zero}
        try:
            return self.model(data, return_embeddings=return_embeddings)
        finally:
            dec.ctx = None


class _DCLoopDecoder(nn.Module):
    """GNS's bus decoder with the angle column replaced by theta_DC + L^-1 s, s the injection that the P-theta steps
    have accumulated so far (bound per forward by DCLoopLayers); the network's own angle output is unused."""

    def __init__(self, mlp_bus):
        super().__init__()
        self.mlp_bus = mlp_bus
        self.ctx = None

    def forward(self, h):
        out = self.mlp_bus(h)
        c = self.ctx
        va = c["theta_dc"] + solve_anchored(c["s"], theta_ref=c["zero"], **c["system"])
        return torch.cat([out[:, :VA_OUT], va[:, None], out[:, VA_OUT + 1 :]], dim=1)


class DCLoopLayers(nn.Module):
    """GNS with the P-theta step of fast-decoupled in every layer: s += GENCO's active residual, on the REF-connected
    buses (module docstring).

    guard=True (X4b) adds the safeguard of damped Newton, per graph: a step after which the norm of the active residual
    grows is undone and retried from the last accepted point with half the gain. With wrong voltages (early training)
    the P-theta problem can have no solution and the plain loop then accumulates mismatch without bound (X4: angles of
    thousands of radians); the guarded loop keeps s bounded and, where the steps keep reducing the mismatch, is the
    plain loop."""

    def __init__(self, model, guard=False):
        super().__init__()
        self.model = model
        self.guard = guard
        base = gns_of(model)
        base.mlp_bus = _DCLoopDecoder(base.mlp_bus)

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    def forward(self, data, return_embeddings=False):
        base = gns_of(self.model)
        with torch.no_grad():
            system, p, theta_ref, conn = dc_context(data)
            theta_dc = solve_anchored(p * conn, theta_ref=theta_ref, **system)
        ctx = {"system": system, "theta_dc": theta_dc, "conn": conn.to(p.dtype), "zero": theta_ref.new_zeros(theta_ref.shape),
               "s": torch.zeros_like(p)}

        batch, n_graphs = system["batch"], system["n_graphs"]
        free = ctx["conn"] * (~system["ref"]).to(p.dtype)
        guard = {"gamma": p.new_ones(n_graphs), "norm": None, "r": torch.zeros_like(p), "last": torch.zeros_like(p)}

        def step(module, inputs, outputs):  # outputs = (residual_P, residual_Q) of the layer just decoded
            r = outputs[0] * ctx["conn"]
            if not self.guard:
                ctx["s"] = ctx["s"] + r
                return
            norm = p.new_zeros(n_graphs).index_add(0, batch, (r * free) ** 2)
            accept = torch.ones_like(norm, dtype=torch.bool) if guard["norm"] is None else norm <= guard["norm"]
            keep = accept[batch]
            gamma = torch.where(accept, guard["gamma"], guard["gamma"] / 2)
            base_r = torch.where(keep, r, guard["r"])  # step from the last accepted point
            s = torch.where(keep, ctx["s"], ctx["s"] - guard["last"])  # undo a step that made things worse
            new = gamma[batch] * base_r
            ctx["s"] = s + new
            guard.update(gamma=gamma, r=base_r, last=new,
                         norm=norm if guard["norm"] is None else torch.where(accept, norm, guard["norm"]))

        base.mlp_bus.ctx = ctx
        hook = base.node_residuals_layer.register_forward_hook(step)
        try:
            return self.model(data, return_embeddings=return_embeddings)
        finally:
            hook.remove()
            base.mlp_bus.ctx = None
