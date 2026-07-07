# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Native-PyTorch replacements for the torch_scatter / torch_sparse subset
used in this repo (E1 migration — see refine-logs/ENGINEERING_PLAN.md).

Semantics mirror torch_scatter for the call patterns that exist here
(1-D index along dim=0, verified empirically and in
tests/test_native_scatter.py):

- missing ``dim_size`` -> ``index.max() + 1`` (0 for empty index),
- empty segments -> 0 for add/mean/max (torch_scatter initializes with
  zeros),
- ``scatter(..., out=...)`` accumulates into ``out`` in place,
- ``scatter_max`` returns ``(values, argmax)`` with ``src.size(0)`` as the
  empty-segment argmax sentinel; tie-breaking between equal maxima is NOT
  guaranteed to match torch_scatter (no call site consumes argmax).

Only dim=0 is implemented — every call site in the repo uses dim=0; other
dims raise loudly.
"""

import torch


def _size(index, dim_size):
    if dim_size is not None:
        return int(dim_size)
    return int(index.max()) + 1 if index.numel() else 0


def _check_dim(dim):
    if dim != 0:
        raise NotImplementedError(
            "native scatter ops in gridfm_graphkit support dim=0 only",
        )


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    """torch_scatter.scatter_add for 1-D index along dim 0."""
    _check_dim(dim)
    if out is None:
        out = src.new_zeros((_size(index, dim_size), *src.shape[1:]))
    return out.index_add_(0, index, src)


def scatter_mean(src, index, dim=0, dim_size=None):
    """torch_scatter.scatter_mean (empty segments -> 0)."""
    _check_dim(dim)
    n = _size(index, dim_size)
    sums = src.new_zeros((n, *src.shape[1:])).index_add_(0, index, src)
    counts = src.new_zeros(n).index_add_(0, index, src.new_ones(index.shape))
    counts = counts.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))
    return sums / counts


def scatter_max(src, index, dim=0, dim_size=None):
    """torch_scatter.scatter_max values; empty segments -> 0.

    Returns ``(values, None)``: no call site in this repo consumes the
    argmax, and computing it needs an int64 scatter_reduce that MPS does
    not support (the very dispatch gap the old torch_scatter MPS
    workaround papered over). Any future use of the second element fails
    loudly on None.
    """
    _check_dim(dim)
    n = _size(index, dim_size)
    idx_exp = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    values = src.new_zeros((n, *src.shape[1:]))
    values.scatter_reduce_(0, idx_exp, src, reduce="amax", include_self=False)
    return values, None


def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
    """torch_scatter.scatter for the reduces used in this repo."""
    _check_dim(dim)
    if reduce in ("sum", "add"):
        return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
    if reduce == "mean":
        if out is not None:
            raise NotImplementedError("scatter(reduce='mean') with out=")
        return scatter_mean(src, index, dim=dim, dim_size=dim_size)
    if reduce == "max":
        if out is not None:
            raise NotImplementedError("scatter(reduce='max') with out=")
        return scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    raise NotImplementedError(f"scatter reduce={reduce!r}")


def coalesce_sum(index, value, m, n):
    """torch_sparse.coalesce(op='add'): sort row-major, sum duplicates.

    Supports value of shape [nnz] or [nnz, K] (hybrid sparse-dense).
    """
    sp = torch.sparse_coo_tensor(index, value, size=(m, n, *value.shape[1:]))
    sp = sp.coalesce()
    return sp.indices(), sp.values()


def dense_from_edge_index(edge_index, edge_weight, num_nodes):
    """Dense [N, N] adjacency with duplicate-edge accumulation
    (SparseTensor.from_edge_index(...).to_dense() equivalent;
    edge_weight=None -> ones)."""
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1),
            dtype=torch.float,
            device=edge_index.device,
        )
    adj = edge_weight.new_zeros(num_nodes, num_nodes)
    adj.index_put_((edge_index[0], edge_index[1]), edge_weight, accumulate=True)
    return adj


def dense_to_coo(mat):
    """(row, col, val) of entries nonzero in any trailing channel
    (SparseTensor.from_dense(mat).coo() equivalent, row-major order)."""
    nz = mat if mat.dim() == 2 else mat.abs().sum(dim=tuple(range(2, mat.dim())))
    idx = (nz != 0).nonzero(as_tuple=False)
    row, col = idx[:, 0], idx[:, 1]
    return row, col, mat[row, col]
