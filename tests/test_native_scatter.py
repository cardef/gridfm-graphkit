# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""E1 migration gate: native scatter/sparse ops vs torch_scatter/torch_sparse.

Fixed-value assertions run always (they keep meaning once the wheels are
gone from the environment); parity assertions run only where the legacy
packages are importable.
"""

import pytest
import torch

from gridfm_graphkit.utils.scatter import (
    coalesce_sum,
    dense_from_edge_index,
    dense_to_coo,
    scatter,
    scatter_add,
    scatter_max,
    scatter_mean,
)

try:
    import torch_scatter

    HAS_TS = True
except ImportError:
    HAS_TS = False

try:
    import torch_sparse

    HAS_TSP = True
except ImportError:
    HAS_TSP = False


def test_fixed_values():
    src = torch.tensor([1.0, -2.0, 3.0, -4.0])
    idx = torch.tensor([0, 0, 2, 2])
    assert torch.equal(
        scatter_add(src, idx, dim=0, dim_size=4),
        torch.tensor([-1.0, 0.0, -1.0, 0.0]),
    )
    assert torch.equal(
        scatter_mean(src, idx, dim=0, dim_size=4),
        torch.tensor([-0.5, 0.0, -0.5, 0.0]),
    )
    vals, arg = scatter_max(src, idx, dim=0, dim_size=4)
    assert torch.equal(vals, torch.tensor([1.0, 0.0, 3.0, 0.0]))
    assert arg is None  # argmax deliberately not computed (MPS int64 gap)
    # out= accumulates in place
    out = torch.ones(3)
    r = scatter(src, idx, dim=0, out=out, reduce="add")
    assert r is out and torch.equal(out, torch.tensor([0.0, 1.0, 0.0]))
    # missing dim_size -> max+1; empty index -> size 0
    assert scatter_add(src, idx, dim=0).shape == (3,)
    empty = scatter_add(torch.zeros(0, 5), torch.zeros(0, dtype=torch.long), dim=0)
    assert empty.shape == (0, 5)
    with pytest.raises(NotImplementedError):
        scatter_add(src, idx, dim=1)


def test_coalesce_fixed():
    idx = torch.tensor([[1, 0, 1], [0, 2, 0]])
    val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    out_idx, out_val = coalesce_sum(idx, val, 2, 3)
    assert torch.equal(out_idx, torch.tensor([[0, 1], [2, 0]]))
    assert torch.equal(out_val, torch.tensor([[3.0, 4.0], [11.0, 22.0]]))


def test_dense_helpers_fixed():
    ei = torch.tensor([[0, 0, 1], [1, 1, 2]])  # duplicate (0,1)
    adj = dense_from_edge_index(ei, torch.tensor([1.0, 2.0, 5.0]), 3)
    assert adj[0, 1] == 3.0 and adj[1, 2] == 5.0
    mat = torch.zeros(3, 3, 2)
    mat[0, 1] = torch.tensor([1.0, 0.0])
    mat[2, 0] = torch.tensor([0.0, -2.0])
    row, col, val = dense_to_coo(mat)
    assert torch.equal(row, torch.tensor([0, 2]))
    assert torch.equal(col, torch.tensor([1, 0]))
    assert torch.equal(val, mat[row, col])


@pytest.mark.skipif(not HAS_TS, reason="torch_scatter not installed")
@pytest.mark.parametrize("shape", [(50,), (50, 7), (50, 4, 3)])
def test_parity_torch_scatter(shape):
    g = torch.Generator().manual_seed(0)
    src = torch.randn(shape, generator=g)
    idx = torch.randint(0, 12, (shape[0],), generator=g)  # segments 10/11 empty
    for dim_size in (12, None):
        assert torch.allclose(
            scatter_add(src, idx, dim=0, dim_size=dim_size),
            torch_scatter.scatter_add(src, idx, dim=0, dim_size=dim_size),
        )
        assert torch.allclose(
            scatter_mean(src, idx, dim=0, dim_size=dim_size),
            torch_scatter.scatter_mean(src, idx, dim=0, dim_size=dim_size),
        )
        v_n, _ = scatter_max(src, idx, dim=0, dim_size=dim_size)
        v_t, _ = torch_scatter.scatter_max(src, idx, dim=0, dim_size=dim_size)
        assert torch.allclose(v_n, v_t)
    # out= accumulation parity
    out_n = torch.randn((12, *shape[1:]), generator=g)
    out_t = out_n.clone()
    scatter(src, idx, dim=0, out=out_n, reduce="add")
    torch_scatter.scatter(src, idx, dim=0, out=out_t, reduce="add")
    assert torch.allclose(out_n, out_t)


@pytest.mark.skipif(not HAS_TSP, reason="torch_sparse not installed")
def test_parity_torch_sparse():
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 9, (2, 40), generator=g)  # guaranteed duplicates
    val = torch.randn(40, 6, generator=g)
    i_n, v_n = coalesce_sum(idx, val, 9, 9)
    i_t, v_t = torch_sparse.coalesce(idx, val, 9, 9, op="add")
    assert torch.equal(i_n, i_t)
    assert torch.allclose(v_n, v_t)

    # dense round-trip parity vs SparseTensor on duplicate-FREE graphs.
    # On multigraphs the legacy path was internally inconsistent
    # (to_dense() overwrites parallel edges while sum(dim=1) adds them);
    # dense_from_edge_index deliberately accumulates so deg == adj.sum(1).
    from torch_sparse import SparseTensor

    ei = torch.unique(torch.randint(0, 15, (2, 60), generator=g), dim=1)
    ew = torch.rand(ei.size(1), generator=g)
    adj_n = dense_from_edge_index(ei, ew, 15)
    adj_t = SparseTensor.from_edge_index(ei, ew, sparse_sizes=(15, 15)).to_dense()
    assert torch.allclose(adj_n, adj_t)
    # edge_weight=None -> ones
    assert torch.allclose(
        dense_from_edge_index(ei, None, 15),
        SparseTensor.from_edge_index(ei, None, sparse_sizes=(15, 15)).to_dense(),
    )
    # multigraph consistency (the fix): parallel edges accumulate
    ei_dup = torch.tensor([[0, 0], [1, 1]])
    adj_dup = dense_from_edge_index(ei_dup, None, 2)
    assert adj_dup[0, 1] == 2.0 and torch.equal(
        adj_dup.sum(1),
        SparseTensor.from_edge_index(ei_dup, None, sparse_sizes=(2, 2)).sum(dim=1),
    )

    mat = torch.randn(10, 10, 4, generator=g)
    mat[torch.rand(10, 10, generator=g) < 0.6] = 0.0
    r_n, c_n, v2_n = dense_to_coo(mat)
    r_t, c_t, v2_t = SparseTensor.from_dense(mat, has_value=True).coo()
    assert torch.equal(r_n, r_t) and torch.equal(c_n, c_t)
    assert torch.allclose(v2_n, v2_t)
