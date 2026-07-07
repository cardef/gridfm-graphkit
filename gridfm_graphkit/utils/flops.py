# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Analytic per-relation forward-FLOP accounting (pre-registered at M0).

Counts one forward pass for a single graph, multiply-accumulate = 2 FLOPs.
Formulas follow the installed torch_geometric TransformerConv
(lin_query/key/value [+ lin_edge], q.k attention, (v+e)*alpha message,
lin_skip + lin_beta with beta=True, concat heads) and the model code in
``gnn_heterogeneous_gns.py`` / ``gnn_hetero_hier.py`` (per-layer decode
heads + physics residual loop included).

Used at M1 to define matched-FLOP (flat depth, KS) pairs: FLOPs within
[0.9, 1.1] at equal hidden_dim, per the frozen experiment plan.
"""

from dataclasses import dataclass


@dataclass
class GraphSizes:
    """Per-sample element counts (directed edge counts, post-transform)."""

    n_bus: int
    n_gen: int
    e_busbus: int  # directed bus-bus edges (2 per branch)
    n_cbus: int = 0  # hierarchy only
    e_coarse: int = 0  # directed thresholded Y_red off-diagonal edges
    nnz_prolong: int = 0  # retained P entries incl. boundary identity
    n_int: int = 0  # interior buses (HELM2 unpool solves, R021)


def linear(n, d_in, d_out):
    return 2 * n * d_in * d_out


def mlp2(n, d_in, d_hidden, d_out):
    """Two-layer MLP as used by the decode heads / input projections."""
    return linear(n, d_in, d_hidden) + linear(n, d_hidden, d_out)


def transformer_conv(n_src, n_dst, e, d_in, heads, out, with_edge, d_edge):
    """One TransformerConv relation, beta=True, concat heads."""
    hc = heads * out
    f = 0
    f += linear(n_dst, d_in, hc)  # lin_query
    f += linear(n_src, d_in, hc)  # lin_key
    f += linear(n_src, d_in, hc)  # lin_value
    if with_edge:
        f += linear(e, d_edge, hc)  # lin_edge (per edge)
        f += 2 * e * hc  # key+edge, value+edge adds
    f += 2 * e * hc  # q.k dot per edge
    f += 6 * e * heads  # softmax (exp/sum/div, per head)
    f += 2 * e * hc  # alpha * value + scatter-add
    f += linear(n_dst, d_in, hc)  # lin_skip
    f += linear(n_dst, 3 * hc, 1) + 3 * n_dst * hc  # lin_beta + blend
    return f


def _fine_layer(s: GraphSizes, d_in, h, heads, cfg):
    """One HGNS HeteroConv fine layer + decode heads + physics loop."""
    d = h * heads
    f = 0
    f += transformer_conv(
        s.n_bus,
        s.n_bus,
        s.e_busbus,
        d_in,
        heads,
        h,
        True,
        h,
    )
    f += transformer_conv(s.n_gen, s.n_bus, s.n_gen, d_in, heads, h, False, 0)
    f += transformer_conv(s.n_bus, s.n_gen, s.n_gen, d_in, heads, h, False, 0)
    f += 10 * (s.n_bus + s.n_gen) * d  # norms + activation + residual
    # per-layer decode heads
    f += mlp2(s.n_bus, d, h, cfg["output_bus_dim"])
    f += mlp2(s.n_gen, d, h, cfg["output_gen_dim"])
    # physics: branch flows (~40 flops/edge incl. trig), injections,
    # decoder + residuals (~30/bus), physics_mlp
    f += 40 * s.e_busbus + 30 * s.n_bus
    f += linear(s.n_bus, 2, d)
    return f


def _input_projections(s: GraphSizes, cfg, h):
    f = mlp2(s.n_bus, cfg["input_bus_dim"], h, h)
    f += mlp2(s.n_gen, cfg["input_gen_dim"], h, h)
    f += mlp2(s.e_busbus, cfg["edge_dim"], h, h)
    return f


def gns_heterogeneous_flops(s: GraphSizes, cfg) -> int:
    """Forward FLOPs of GNS_heterogeneous (flat baseline)."""
    h, heads = cfg["hidden_size"], cfg["attention_head"]
    d = h * heads
    f = _input_projections(s, cfg, h)
    for i in range(cfg["num_layers"]):
        f += _fine_layer(s, h if i == 0 else d, h, heads, cfg)
    return f


def gns_hetero_hier_flops(s: GraphSizes, cfg) -> int:
    """Forward FLOPs of GNS_hetero_hier (Kron-Schur 2-level)."""
    h, heads = cfg["hidden_size"], cfg["attention_head"]
    d = h * heads
    f = _input_projections(s, cfg, h)
    f += mlp2(s.e_coarse, 2, h, h)  # coarse edge projection
    n_fine = cfg["num_layers_fine"] + cfg["num_layers_fine_post"]
    for i in range(n_fine):
        f += _fine_layer(s, h if i == 0 else d, h, heads, cfg)
    # restriction + MLP_in
    f += linear(s.n_cbus, d + cfg.get("input_cbus_dim", 8), d)
    # coarse stack
    for _ in range(cfg["num_layers_coarse"]):
        f += transformer_conv(
            s.n_cbus,
            s.n_cbus,
            s.e_coarse,
            d,
            heads,
            h,
            True,
            h,
        )
        f += 10 * s.n_cbus * d
    # coarse decoder D_c
    f += mlp2(s.n_cbus, d, h, 2)
    # physical prolongation (complex mul + scatter) + trig
    f += 10 * s.nnz_prolong + 20 * s.n_cbus
    if cfg.get("unpool", "affine") == "helm2":
        # R021 unpool tail: conj(S_I) recovery (dense complex matmul) + two
        # batched LU solves (fwd+bwd triangular substitution each) + the
        # elementwise series weights; complex MAC = 8 real FLOPs
        f += 8 * s.n_int * s.n_int  # cS = Yii v_aff
        f += 2 * 8 * s.n_int * s.n_int  # lu_solve for c1, c2
        f += 40 * s.n_int  # w0/w1/tail/canary elementwise
    # latent merge: two weighted scatters of h_c + MLP_out
    f += 6 * s.nnz_prolong * d
    f += linear(s.n_bus, 2 * d + 2, d)
    return f


def matched_flat_depth(s: GraphSizes, hier_cfg, depths=(8, 16, 32, 48)):
    """Pick the flat depth whose FLOPs are closest to the KS model's.

    Returns (best_depth, ratio flat/KS, per-depth table). The M1 matched-FLOP
    criterion accepts ratios in [0.9, 1.1]; if no swept depth lands inside,
    the caller adjusts width (both reported).
    """
    target = gns_hetero_hier_flops(s, hier_cfg)
    flat_cfg = dict(hier_cfg)
    table = {}
    for depth in depths:
        flat_cfg["num_layers"] = depth
        table[depth] = gns_heterogeneous_flops(s, flat_cfg)
    best = min(table, key=lambda k: abs(table[k] / target - 1))
    return best, table[best] / target, {"ks_flops": target, "flat": table}
