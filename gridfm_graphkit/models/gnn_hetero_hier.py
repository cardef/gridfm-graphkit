# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Kron-Schur 2-level hierarchical GNS (encoder-process-decoder U-Net).

Fine and coarse processors are instances of the same HeteroConv/TransformerConv
layer class as :class:`GNS_heterogeneous`; pool/unpool are the fixed
electrically exact operators attached by ``AddHierarchy``:

  fine stack (L_f layers, physics-residual loop as in the flat baseline)
    -> latent restriction: h_c = MLP_in([h_bus[boundary] ; cbus phys feats])
    -> coarse stack (L_c layers on thresholded Y_red edges)
    -> coarse decoder D_c -> (VM_c, VA_c)  [trained with CoarseVoltageMSE]
    -> physical prolongation: V_hat = v_aff + P_sp V_B (complex, 2-channel)
    -> latent merge: h_bus += MLP_out([P_sp h_c (2 ch) ; V_hat])
  fine stack (L_f' layers, physics-residual loop)
    -> existing per-layer decode + physics decoder (identical to baseline)

The physical channel {W (precomputed), v_aff, P_sp, Y_red} carries the
exactness claims; the latent channel {MLP_in, coarse stack, MLP_out} claims
none. The residual merge lets the fine path override the linear prior.

R021 (``model.unpool: helm2``, default ``affine``): the affine unpool
``V_hat = P_sp V_B + v_aff`` is the order-1 truncation (at flat voltage) of
the boundary-conditioned HELM series (idea-stage/HELM_UNPOOL_NOTE.md); the
helm2 unpool replaces ``v_aff`` with the order-2 tail ``c_1 + c_2``:

    c_0 = P_sp V_B                      (existing scatter)
    w_0 = 1/conj(c_0)
    c_1 = Yii^-1 (conj(S_I) w_0)        (batched dense LU solve)
    w_1 = -w_0 conj(c_1) w_0
    c_2 = Yii^-1 (conj(S_I) w_1)        (second solve, same factors)

``conj(S_I)`` is recovered exactly as ``Yii v_aff`` (no cache change). The
static operators (Yii, interior index map) come from the per-grid HELM
runtime file written by ``build_grid_hierarchy`` and are loaded lazily in
the model's own process, keyed by the ``helm_runtime_path`` string that
``AddHierarchy`` attaches per sample (worker/DDP-safe). The divergence
canary median ``|c_2|/|c_1|`` of the last forward is kept in
``self.helm_canary`` (kill criterion (b) of the pre-registered R021 arm).
"""

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv
from gridfm_graphkit.utils.scatter import scatter_add

from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.io.param_handler import get_physics_decoder
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)
from gridfm_graphkit.datasets.globals import (
    VM_H,
    VA_H,
    PG_H,
)

FINE_RELATIONS = (
    ("bus", "connects", "bus"),
    ("gen", "connected_to", "bus"),
    ("bus", "connected_to", "gen"),
)
COARSE_RELATION = ("cbus", "connects", "cbus")
SEEDS_RELATION = ("bus", "seeds", "cbus")
PROLONG_RELATION = ("cbus", "prolong", "bus")

# |c_0| floor for the reciprocal-series weights w = 1/conj(c_0): the coarse
# decoder can emit near-zero voltages early in training, and 1/|c_0| would
# blow up gradients. Grid voltages are ~1 p.u. (measured min |c_0| with true
# V_b: 0.909 across all grids, HELM pilot); below the floor the weight is
# damped smoothly to 0 instead of exploding.
HELM_C0_MIN = 0.5


def helm2_tail(lu, pivots, yii, c0_int, v_aff_int):
    """Order-1+2 HELM tail for a same-grid group of samples.

    Args: LU factors + pivots of dense Yii [n_i, n_i] (complex), Yii itself,
    ``c0_int``/``v_aff_int`` complex [B, n_i] (boundary-conditioned germ from
    the prolongation scatter; affine term = Yii^-1 conj(S_I)).
    Returns (tail [B, n_i] complex = c1 + c2, canary |c2|/|c1| flat tensor).
    """
    cs = yii @ v_aff_int.transpose(0, 1)  # conj(S_I), [n_i, B]
    a2 = c0_int.abs().square().clamp_min(HELM_C0_MIN**2)
    w0 = (c0_int / a2).transpose(0, 1)  # 1/conj(c0), floored
    c1 = torch.linalg.lu_solve(lu, pivots, cs * w0)
    w1 = -w0 * c1.conj() * w0
    c2 = torch.linalg.lu_solve(lu, pivots, cs * w1)
    canary = (c2.abs() / c1.abs().clamp_min(1e-12)).flatten()
    return (c1 + c2).transpose(0, 1), canary


@MODELS_REGISTRY.register("GNS_hetero_hier")
class GNS_hetero_hier(nn.Module):
    """2-level Kron-Schur hierarchical GNS for PF (see module docstring)."""

    def __init__(self, args) -> None:
        super().__init__()
        m = args.model
        self.hidden_dim = m.hidden_size
        self.heads = m.attention_head
        self.num_layers_fine = getattr(m, "num_layers_fine", 4)
        self.num_layers_coarse = getattr(m, "num_layers_coarse", 4)
        self.num_layers_fine_post = getattr(m, "num_layers_fine_post", 4)
        self.dropout = getattr(m, "dropout", 0.0)
        self.cbus_in_dim = getattr(m, "input_cbus_dim", 8)
        self.unpool = getattr(m, "unpool", "affine")
        if self.unpool not in ("affine", "helm2"):
            raise ValueError(f"model.unpool must be affine|helm2, got {self.unpool}")
        self._helm_rt = {}  # runtime-file path -> static HELM operators
        self.helm_canary = None  # median |c2|/|c1| of the last helm2 forward
        self.task = args.task.task_name
        if self.task != "PowerFlow":
            raise NotImplementedError("GNS_hetero_hier supports PowerFlow only (M0)")

        D = self.hidden_dim * self.heads

        def proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )

        self.input_proj_bus = proj(m.input_bus_dim)
        self.input_proj_gen = proj(m.input_gen_dim)
        self.input_proj_edge = proj(m.edge_dim)
        self.input_proj_cedge = proj(2)  # (G_red, B_red)

        self.physics_mlp = nn.Sequential(
            nn.Linear(2, D),
            nn.LeakyReLU(),
        )

        def fine_layer(i, first):
            in_dim = self.hidden_dim if first else D
            conv_dict = {
                ("bus", "connects", "bus"): TransformerConv(
                    in_dim,
                    self.hidden_dim,
                    heads=self.heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("gen", "connected_to", "bus"): TransformerConv(
                    in_dim,
                    self.hidden_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("bus", "connected_to", "gen"): TransformerConv(
                    in_dim,
                    self.hidden_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
            }
            return HeteroConv(conv_dict, aggr="sum")

        n_fine = self.num_layers_fine + self.num_layers_fine_post
        self.fine_layers = nn.ModuleList(
            [fine_layer(i, first=(i == 0)) for i in range(n_fine)],
        )
        self.norms_bus = nn.ModuleList(nn.LayerNorm(D) for _ in range(n_fine))
        self.norms_gen = nn.ModuleList(nn.LayerNorm(D) for _ in range(n_fine))

        # coarse processor: same layer class on the reduced network
        self.coarse_layers = nn.ModuleList(
            HeteroConv(
                {
                    COARSE_RELATION: TransformerConv(
                        D,
                        self.hidden_dim,
                        heads=self.heads,
                        edge_dim=self.hidden_dim,
                        dropout=self.dropout,
                        beta=True,
                    ),
                },
                aggr="sum",
            )
            for _ in range(self.num_layers_coarse)
        )
        self.norms_cbus = nn.ModuleList(
            nn.LayerNorm(D) for _ in range(self.num_layers_coarse)
        )

        # cross-level MLPs (the only new trainable pieces besides D_c)
        self.mlp_in = nn.Sequential(
            nn.Linear(D + self.cbus_in_dim, D),
            nn.LeakyReLU(),
            nn.LayerNorm(D),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(2 * D + 2, D),
            nn.LeakyReLU(),
            nn.LayerNorm(D),
        )
        # coarse decoder D_c -> (VM_c, VA_c)
        self.mlp_cbus = nn.Sequential(
            nn.Linear(D, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 2),
        )

        self.mlp_bus = nn.Sequential(
            nn.Linear(D, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, m.output_bus_dim),
        )
        self.mlp_gen = nn.Sequential(
            nn.Linear(D, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, m.output_gen_dim),
        )

        self.activation = nn.LeakyReLU()
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()
        self.physics_decoder = get_physics_decoder(args)
        self.layer_residuals = {}

    def _helm_runtime(self, path, device):
        """Load (once per process) the per-grid static HELM operators."""
        rt = self._helm_rt.get(path)
        if rt is None:
            if not path:
                raise RuntimeError(
                    "unpool='helm2' requires the HELM runtime file next to the "
                    "hierarchy cache; rebuild it (experiments/m0/"
                    "r006_p_recon_gate.py or build_grid_hierarchy).",
                )
            d = torch.load(path, weights_only=True)
            n_i = d["interior_idx"].numel()
            yii = torch.zeros(n_i, n_i, dtype=torch.complex64)
            idx = d["yii_edge_index"]
            yii[idx[0], idx[1]] = torch.complex(
                d["yii_edge_attr"][:, 0],
                d["yii_edge_attr"][:, 1],
            )
            lu, pivots = torch.linalg.lu_factor(yii)
            rt = {
                "interior_idx": d["interior_idx"],
                "yii": yii,
                "lu": lu,
                "pivots": pivots,
            }
            self._helm_rt[path] = rt
        if rt["yii"].device != device:
            rt = {k: v.to(device) for k, v in rt.items()}
            self._helm_rt[path] = rt
        return rt

    def _helm2_unpool(self, batch, c0, v_aff_c, num_bus):
        """HELM2 unpool over a (possibly multi-grid) batch, grouped by grid."""
        paths = batch.helm_runtime_path
        if isinstance(paths, str):
            paths = [paths]
        bus_store = batch["bus"]
        ptr = getattr(bus_store, "ptr", None)
        if ptr is None:
            ptr = torch.tensor([0, num_bus], device=c0.device)
        tail = torch.zeros_like(c0)
        canaries = []
        for path in dict.fromkeys(paths):  # unique, order-stable
            rt = self._helm_runtime(path, c0.device)
            sample_ids = torch.tensor(
                [i for i, p in enumerate(paths) if p == path],
                device=c0.device,
            )
            gidx = ptr[sample_ids].unsqueeze(1) + rt["interior_idx"].unsqueeze(0)
            tail_g, canary = helm2_tail(
                rt["lu"],
                rt["pivots"],
                rt["yii"],
                c0[gidx],
                v_aff_c[gidx],
            )
            tail = tail.index_put((gidx,), tail_g)
            canaries.append(canary)
        self.helm_canary = torch.cat(canaries).median().detach()
        return c0 + tail

    def _fine_block(self, layer_range, h_bus, h_gen, ctx):
        """Run fine HGNS layers with the baseline's decode/physics loop."""
        (
            edge_index_dict,
            edge_attr_proj,
            x_dict,
            mask_dict,
            bus_mask,
            gen_mask,
            bus_fixed,
            gen_fixed,
            bus_edge_index,
            bus_edge_attr,
            gen_to_bus_index,
            num_bus,
        ) = ctx
        output_temp, gen_temp = None, None
        for i in layer_range:
            out = self.fine_layers[i](
                {"bus": h_bus, "gen": h_gen},
                edge_index_dict,
                edge_attr_proj,
            )
            out_bus = self.activation(self.norms_bus[i](out["bus"]))
            out_gen = self.activation(self.norms_gen[i](out["gen"]))
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            bus_temp = self.mlp_bus(h_bus)
            gen_temp = self.mlp_gen(h_gen)
            bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
            gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)

            Pft, Qft = self.branch_flow_layer(bus_temp, bus_edge_index, bus_edge_attr)
            P_in, Q_in = self.node_injection_layer(
                Pft,
                Qft,
                bus_edge_index,
                num_bus,
            )
            agg_bus = scatter_add(
                gen_temp.squeeze(),
                gen_to_bus_index,
                dim=0,
                dim_size=num_bus,
            )
            output_temp = self.physics_decoder(
                P_in,
                Q_in,
                bus_temp,
                x_dict["bus"],
                agg_bus,
                mask_dict,
            )
            residual_P, residual_Q = self.node_residuals_layer(
                P_in,
                Q_in,
                output_temp,
                x_dict["bus"],
            )
            bus_residuals = torch.stack([residual_P, residual_Q], dim=-1)
            self.layer_residuals[i] = torch.linalg.norm(
                bus_residuals,
                dim=-1,
            ).mean()
            h_bus = h_bus + self.physics_mlp(bus_residuals)
        return h_bus, h_gen, output_temp, gen_temp

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        mask_dict = batch.mask_dict
        self.layer_residuals = {}

        h_bus = self.input_proj_bus(x_dict["bus"])
        h_gen = self.input_proj_gen(x_dict["gen"])
        num_bus = x_dict["bus"].size(0)

        edge_attr_proj = {}
        for key in FINE_RELATIONS:
            attr = edge_attr_dict.get(key)
            edge_attr_proj[key] = (
                self.input_proj_edge(attr) if attr is not None else None
            )
        fine_edge_index = {k: edge_index_dict[k] for k in FINE_RELATIONS}

        bus_mask = mask_dict["bus"][:, VM_H : VA_H + 1]
        gen_mask = mask_dict["gen"][:, : (PG_H + 1)]
        bus_fixed = x_dict["bus"][:, VM_H : VA_H + 1]
        gen_fixed = x_dict["gen"][:, : (PG_H + 1)]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]

        ctx = (
            fine_edge_index,
            edge_attr_proj,
            x_dict,
            mask_dict,
            bus_mask,
            gen_mask,
            bus_fixed,
            gen_fixed,
            bus_edge_index,
            bus_edge_attr,
            gen_to_bus_index,
            num_bus,
        )

        # --- fine stack 1 ---
        h_bus, h_gen, _, _ = self._fine_block(
            range(self.num_layers_fine),
            h_bus,
            h_gen,
            ctx,
        )

        # --- latent restriction (selection S via seeds relation) ---
        seed_bus, seed_cbus = edge_index_dict[SEEDS_RELATION]
        num_cbus = x_dict["cbus"].size(0)
        h_seed = h_bus.new_zeros(num_cbus, h_bus.size(1))
        h_seed[seed_cbus] = h_bus[seed_bus]
        h_c = self.mlp_in(torch.cat([h_seed, x_dict["cbus"]], dim=1))

        # --- coarse processor on thresholded Y_red ---
        c_edge_index = edge_index_dict[COARSE_RELATION]
        c_edge_attr = self.input_proj_cedge(edge_attr_dict[COARSE_RELATION])
        for i, conv in enumerate(self.coarse_layers):
            out = conv(
                {"cbus": h_c},
                {COARSE_RELATION: c_edge_index},
                {COARSE_RELATION: c_edge_attr},
            )
            h_c = h_c + self.activation(self.norms_cbus[i](out["cbus"]))

        # --- coarse decoder ---
        vmva_c = self.mlp_cbus(h_c)  # [N_c, 2] -> (VM_c, VA_c)

        # --- physical prolongation: c_0 = P_sp V_B (scatter) ---
        prol_src, prol_dst = edge_index_dict[PROLONG_RELATION]
        p_attr = edge_attr_dict[PROLONG_RELATION]  # [nnz, 2] (Re, Im)
        vm_c, va_c = vmva_c[:, 0], vmva_c[:, 1]
        vb_r, vb_i = vm_c * torch.cos(va_c), vm_c * torch.sin(va_c)
        pr, pi = p_attr[:, 0], p_attr[:, 1]
        msg_r = pr * vb_r[prol_src] - pi * vb_i[prol_src]
        msg_i = pr * vb_i[prol_src] + pi * vb_r[prol_src]
        c0_r = scatter_add(msg_r, prol_dst, dim=0, dim_size=num_bus)
        c0_i = scatter_add(msg_i, prol_dst, dim=0, dim_size=num_bus)
        v_aff = batch["bus"].v_aff
        if self.unpool == "affine":
            # V_hat = c_0 + v_aff (order-1 HELM at flat voltage)
            v_hat = v_aff.clone()
            v_hat[:, 0] += c0_r
            v_hat[:, 1] += c0_i
        else:
            # V_hat = c_0 + c_1 + c_2 (R021 helm2)
            v_c = self._helm2_unpool(
                batch,
                torch.complex(c0_r, c0_i),
                torch.complex(v_aff[:, 0], v_aff[:, 1]),
                num_bus,
            )
            v_hat = torch.stack([v_c.real, v_c.imag], dim=1)

        # --- latent merge through the same sparsified P (both channels) ---
        m_r = scatter_add(
            pr.unsqueeze(1) * h_c[prol_src],
            prol_dst,
            dim=0,
            dim_size=num_bus,
        )
        m_i = scatter_add(
            pi.unsqueeze(1) * h_c[prol_src],
            prol_dst,
            dim=0,
            dim_size=num_bus,
        )
        h_bus = h_bus + self.mlp_out(torch.cat([m_r, m_i, v_hat], dim=1))

        # --- fine stack 2 + final decode (baseline-identical) ---
        n_fine = self.num_layers_fine + self.num_layers_fine_post
        h_bus, h_gen, output_temp, gen_temp = self._fine_block(
            range(self.num_layers_fine, n_fine),
            h_bus,
            h_gen,
            ctx,
        )

        return {"bus": output_temp, "gen": gen_temp, "cbus": vmva_c}
