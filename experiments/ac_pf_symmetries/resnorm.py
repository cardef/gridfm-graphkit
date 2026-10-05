"""Ablation of M3 (R2): canon's per-graph frame for the inputs, M3's per-bus normalization of the residual.

Inside M3 (`Canonicalize(local=True)`) GNS sees bus i's power-balance residual divided by s_i = D_i / D_ref, both
where it feeds it back into every layer (`physics_mlp`) and in its physics loss (`layer_residuals`). Wrapped in
`Canonicalize()`, `BusResidualScale` gives GNS exactly that residual (D_i is degree-1 homogeneous, so the per-graph
frame cancels) while every input keeps canon's frame: a difference from canon is the residual normalization, a
difference from M3 the local inputs.
"""

import torch
from torch import nn

from .symmetries import Canonicalize


class _ScaledResiduals(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.scale = None  # per bus, bound for one forward by BusResidualScale

    def forward(self, P_in, Q_in, bus_data_pred, bus_data_orig):
        res_p, res_q = self.inner(P_in, Q_in, bus_data_pred, bus_data_orig)
        return res_p / self.scale, res_q / self.scale


class BusResidualScale(nn.Module):
    """Divide every bus residual GNS computes by D_i / D_ref, D_i from the inputs it sees (`node_scale`)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        model.node_residuals_layer = _ScaledResiduals(model.node_residuals_layer)
        self.register_buffer("d_ref", torch.ones(()))

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    @torch.no_grad()
    def fit_d_ref(self, loader):
        """M3's constant: the mean D_i over the training buses, before any frame."""
        self.d_ref.fill_(torch.cat([Canonicalize.node_scale(d) for d in loader]).mean())
        return self

    def forward(self, data, return_embeddings=False):
        layer = self.model.node_residuals_layer
        layer.scale = Canonicalize.node_scale(data) / self.d_ref
        try:
            return self.model(data, return_embeddings=return_embeddings)
        finally:
            layer.scale = None
