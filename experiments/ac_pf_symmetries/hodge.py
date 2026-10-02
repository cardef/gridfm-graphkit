"""M1 (Piano A §6): bus angles reconstructed from predicted branch angle differences.

theta = argmin_theta sum_e w_e (theta_f - theta_t - delta_e)^2 with theta_REF given, i.e. the weighted Hodge
projection theta = L_w^+ B^T W delta + const with the constant fixed at the REF angle (the least-squares solution is
unique up to a constant, so anchoring and projecting are the same thing). Weights w_e = |Y_ft|: with them the
projection is the DC-power-flow one, and under S3 every weight scales by 1/k, which leaves theta unchanged.
"""

import torch
from torch import nn
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.globals import VA_H, VM_OUT, YFT_TF_I, YFT_TF_R

from .symmetries import E, line_pairs, node_batch, num_graphs


def reconstruct_angles(delta, f, t, w, ref, theta_ref, batch, n_graphs, eps=1e-6):
    """Bus angles from branch differences `delta` (branch e goes f[e] -> t[e]), anchored at the REF angle.

    Per graph: the weighted Laplacian with the REF row and column replaced by the identity, solved against
    B^T W delta (REF entry 0), plus theta_ref. Every graph of the batch must have the same bus count (one grid per
    batch). eps * mean(w) on the non-REF diagonal keeps a bus islanded by an outage at the REF angle instead of
    making the solve singular; on a connected graph it moves theta by O(eps).
    """
    n_all = batch.numel()
    n = n_all // n_graphs
    assert n * n_graphs == n_all and bool((torch.bincount(batch, minlength=n_graphs) == n).all()), (
        "one grid per batch"
    )
    local = torch.arange(n_all, device=batch.device) - batch * n
    g, fi, ti = batch[f], local[f], local[t]
    lap = delta.new_zeros(n_graphs, n, n)
    for (i, j), s in (((fi, fi), 1.0), ((ti, ti), 1.0), ((fi, ti), -1.0), ((ti, fi), -1.0)):
        lap.index_put_((g, i, j), s * w, accumulate=True)
    rhs = delta.new_zeros(n_graphs, n)
    rhs.index_put_((g, fi), w * delta, accumulate=True)
    rhs.index_put_((g, ti), -w * delta, accumulate=True)
    free = torch.ones(n_graphs, n, dtype=torch.bool, device=batch.device)
    free[batch[ref], local[ref]] = False
    lap = lap * (free[:, :, None] & free[:, None, :]) + torch.diag_embed(
        (~free).to(lap.dtype) + eps * w.mean() * free.to(lap.dtype),
    )
    theta0 = torch.linalg.solve(lap, (rhs * free).unsqueeze(-1)).squeeze(-1)
    return (theta0 + theta_ref[:, None]).reshape(-1)


class BranchAngleHead(nn.Module):
    """Replace a GNS model's bus angles by the REF-anchored reconstruction from branch differences (M1).

    delta for branch f -> t is h(f, t) - h(t, f), one MLP on [h_from, h_to, edge row] of the backbone's final bus
    embeddings evaluated on the two directed rows: antisymmetric, so re-declaring a line flips delta with the
    incidence and leaves theta unchanged (S4 on lines; on transformers the tap column is stored from -> to on both
    rows, as for M0, and `Canonicalize`'s orientation frame is what makes it exact). PG at the REF and QG at PV/REF
    are recomputed from the new state with
    the model's own physics decoder, and the last layer residual is replaced by the residual of that state, so the
    physics loss keeps its 12 terms and weights. Not S1-exact on its own (the backbone reads the REF angle input):
    wrap it in `Canonicalize` for that.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        d, h = model.hidden_dim * model.heads, model.hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d + model.edge_dim, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(),
            nn.Linear(h, 1),
        )

    @property
    def layer_residuals(self):
        return self.model.layer_residuals

    def forward(self, data, return_embeddings=False):
        m = self.model
        out, emb = m(data, return_embeddings=True)
        ei, ea = data[E].edge_index, data[E].edge_attr
        x = emb["bus"]
        h = self.edge_mlp(torch.cat([x[ei[0]], x[ei[1]], ea], 1)).squeeze(-1)
        partner, fwd, _ = line_pairs(data)
        delta = (h - h[partner])[fwd]
        f, t = ei[:, fwd]
        w = ea[fwd][:, [YFT_TF_R, YFT_TF_I]].norm(dim=1)
        ref, batch, n_graphs = data.mask_dict["REF"], node_batch(data, "bus"), num_graphs(data)
        theta_ref = x.new_zeros(n_graphs)
        theta_ref[batch[ref]] = data["bus"].x[ref, VA_H]
        theta = reconstruct_angles(delta, f, t, w, ref, theta_ref, batch, n_graphs)
        state = torch.stack([out["bus"][:, VM_OUT], theta], 1)
        n = state.size(0)
        pft, qft = m.branch_flow_layer(state, ei, ea)
        p_in, q_in = m.node_injection_layer(pft, qft, ei, n)
        _, gen_to_bus = data.edge_index_dict[("gen", "connected_to", "bus")]
        agg = scatter_add(out["gen"].squeeze(-1), gen_to_bus, dim=0, dim_size=n)
        bus = m.physics_decoder(p_in, q_in, state, data["bus"].x, agg, data.mask_dict)
        rp, rq = m.node_residuals_layer(p_in, q_in, bus, data["bus"].x)
        m.layer_residuals[max(m.layer_residuals)] = torch.stack([rp, rq], -1).norm(dim=-1).mean()
        out = {"bus": bus, "gen": out["gen"]}
        return (out, emb) if return_embeddings else out
