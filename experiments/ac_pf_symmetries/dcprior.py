"""DC prior (R2): every layer of GNS decodes its bus angles as the DC power-flow solution plus a learned correction.

theta_DC solves B theta = P on the non-REF buses, with B_ft = Im Y_ft on the forward rows, P = the Pg of the
generators whose Pg is an input minus Pd, and theta fixed at the REF bus: one anchored Laplacian solve per graph
(`solve_anchored`), from the inputs only, so it needs no gradient and is exact under S1 (anchored), S3 (B and P
scale together), S4 and S5. It is the DC of `r2_baselines.py`, the reference the zero-shot errors are compared
with. As in `BranchAngleLayers`, the decoder `mlp_bus` is wrapped, so the per-layer physics feedback acts on
DC + correction.
"""

import torch
from torch import nn

from gridfm_graphkit.datasets.globals import PD_H, PG_H, VA_H, VA_OUT, YFT_TF_I

from .hodge import solve_anchored
from .symmetries import E, line_pairs, node_batch, num_graphs


def dc_angles(data):
    """theta_DC per bus (module docstring)."""
    _, fwd, _ = line_pairs(data)
    ei, ea = data[E].edge_index, data[E].edge_attr
    gi, gb = data.edge_index_dict[("gen", "connected_to", "bus")]
    known = (~data.mask_dict["gen"][gi, PG_H]).to(ea.dtype)
    p = torch.zeros(data["bus"].x.size(0), dtype=ea.dtype, device=ea.device)
    p = p.index_add(0, gb, data["gen"].x[gi, PG_H] * known) - data["bus"].x[:, PD_H]
    ref, batch, n_graphs = data.mask_dict["REF"], node_batch(data, "bus"), num_graphs(data)
    theta_ref = ea.new_zeros(n_graphs)
    theta_ref[batch[ref]] = data["bus"].x[ref, VA_H]
    return solve_anchored(p, ei[0, fwd], ei[1, fwd], ea[fwd, YFT_TF_I], ref, theta_ref, batch, n_graphs)


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
    """GNS whose every layer predicts the angles as theta_DC + correction."""

    def __init__(self, model):
        super().__init__()
        self.model = model
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
