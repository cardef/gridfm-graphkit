# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.fm_scaling.select_geometry import evaluate_candidates
from gridfm_graphkit.fm_scaling.contracts import (
    ComplexCOO,
    ContractError,
    GeometryBudget,
    GridTopology,
)
from gridfm_graphkit.fm_scaling.geometry import (
    KronGeometryBuilder,
    QuotientGeometryBuilder,
    projected_sparse_message_flops,
    select_geometry_candidate,
)
from gridfm_graphkit.fm_scaling.partition import DeterministicPartitioner
from gridfm_graphkit.fm_scaling.registry import (
    GeometryRegistry,
    load_geometry_bundle,
    save_geometry_bundle,
)


def fixed_backend(adjacency, num_parts, seed):
    del adjacency, seed
    if num_parts != 3:
        raise AssertionError(num_parts)
    return [0, 0, 1, 1, 2, 2]


def synthetic_topology(order=None):
    order = list(range(6)) if order is None else list(order)
    stable_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    y_stable = np.zeros((6, 6), dtype=np.complex128)
    for source, target in stable_edges:
        admittance = 1.0 - 2.0j
        y_stable[source, source] += admittance
        y_stable[target, target] += admittance
        y_stable[source, target] -= admittance
        y_stable[target, source] -= admittance
    y_stable += np.eye(6) * (0.2 + 0.1j)

    stable_to_original = {bus_id: index for index, bus_id in enumerate(order)}
    y = y_stable[np.ix_(order, order)]
    row, col = np.nonzero(y)
    fine_edges = tuple(
        (stable_to_original[source], stable_to_original[target])
        for source, target in stable_edges
    )
    return GridTopology(
        topology_key="synthetic-6",
        bus_ids=tuple(order),
        fine_edges=fine_edges,
        y_bus=ComplexCOO(
            shape=(6, 6),
            row=tuple(int(value) for value in row),
            col=tuple(int(value) for value in col),
            value=tuple(complex(y[r, c]) for r, c in zip(row, col)),
        ),
        base_mva=100.0,
        provenance_group="synthetic",
    )


def budget(kappa=20.0):
    return GeometryBudget(
        rho=0.5,
        k_p=3,
        k_c=2,
        kappa=kappa,
        metis_seed=17,
        max_condition=1e8,
        max_harmonic_residual=1.0,
        max_dense_bytes=10_000_000,
        max_build_seconds=10.0,
    )


def builders():
    partitioner = DeterministicPartitioner(fixed_backend)
    return KronGeometryBuilder(partitioner), QuotientGeometryBuilder(partitioner)


def _selection_candidate(
    policy_hash,
    residual,
    cross_nnz,
    coarse_nnz,
    coarse_nodes,
):
    measurement = {
        "residual": residual,
        "cross_nnz": cross_nnz,
        "coarse_nnz": coarse_nnz,
        "coarse_nodes": coarse_nodes,
    }
    measurement["projected_sparse_message_flops"] = projected_sparse_message_flops(
        measurement,
        _flop_model(),
    )
    return {
        "policy_hash": policy_hash,
        "status": "PASS",
        "measurements": [measurement],
    }


def _flop_model():
    return {"per_cross_nnz": 4, "per_coarse_nnz": 2, "per_coarse_node": 1}


def test_geometry_selection_uses_five_percent_residual_band_then_sparse_flops():
    best_but_expensive = _selection_candidate("a", 1.0, 100, 50, 20)
    within_band_and_cheaper = _selection_candidate("b", 1.04, 10, 5, 20)
    outside_band = _selection_candidate("c", 1.051, 1, 1, 2)

    selected, best, limit = select_geometry_candidate(
        [best_but_expensive, within_band_and_cheaper, outside_band],
        _flop_model(),
    )

    assert selected["policy_hash"] == "b"
    assert best == pytest.approx(1.0)
    assert limit == pytest.approx(1.05)


def test_geometry_selection_rejects_inconsistent_projected_flops():
    candidate = _selection_candidate("a", 0.1, 3, 2, 2)
    candidate["measurements"][0]["projected_sparse_message_flops"] += 1

    with pytest.raises(ContractError, match="inconsistent projected FLOPs"):
        select_geometry_candidate([candidate], _flop_model())


def test_geometry_selection_requires_explicit_projected_flop_model():
    candidate = _selection_candidate("a", 0.1, 3, 2, 2)

    with pytest.raises(ContractError, match="projected FLOP model requires exactly"):
        select_geometry_candidate([candidate], {})


def test_geometry_selection_rejects_unversioned_candidate_input(tmp_path):
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text("schema_version: wrong\ncandidates: []\n")

    with pytest.raises(ContractError, match="wrong schema"):
        evaluate_candidates(tmp_path / "missing.yaml", tmp_path, candidates)


def _row_sums(operator):
    sums = np.zeros(operator.output_size)
    for row, weight in zip(operator.row, operator.weight):
        sums[row] += weight
    return sums


def test_kron_and_quotient_share_partition_and_conservative_transport():
    kron_builder, quotient_builder = builders()
    topology = synthetic_topology()
    kron = kron_builder.build(topology, budget())
    quotient = quotient_builder.build(topology, budget())

    assert kron.partition == quotient.partition
    assert kron.coarse_graph.num_nodes == quotient.coarse_graph.num_nodes == 3
    assert np.allclose(_row_sums(kron.prolong), 1.0)
    assert np.allclose(_row_sums(kron.restrict), 1.0)
    assert np.allclose(_row_sums(quotient.prolong), 1.0)
    assert np.allclose(_row_sums(quotient.restrict), 1.0)
    assert kron.harmonic_residual < 1e-10
    assert set(quotient.prolong.coefficient) == {1 + 0j}

    coefficient = np.zeros(
        (kron.prolong.output_size, kron.prolong.input_size),
        dtype=np.complex128,
    )
    prolong = np.zeros_like(coefficient.real)
    restrict = np.zeros((coefficient.shape[1], coefficient.shape[0]))
    for row, col, value, weight in zip(
        kron.prolong.row,
        kron.prolong.col,
        kron.prolong.coefficient,
        kron.prolong.weight,
    ):
        coefficient[row, col] = value
        prolong[row, col] = weight
    for row, col, weight in zip(
        kron.restrict.row,
        kron.restrict.col,
        kron.restrict.weight,
    ):
        restrict[row, col] = weight
    magnitude = np.abs(coefficient)
    d_i = magnitude.sum(axis=1)
    d_b = magnitude.sum(axis=0)
    assert np.allclose(np.diag(d_b) @ restrict, prolong.T @ np.diag(d_i))
    assert np.allclose(np.diag(d_b) @ restrict, magnitude.T)


def test_partition_is_stable_id_permutation_invariant():
    partitioner = DeterministicPartitioner(fixed_backend)
    original = synthetic_topology()
    permuted = synthetic_topology(order=[3, 0, 5, 2, 1, 4])

    first = partitioner.partition(original, rho=0.5, seed=17)
    second = partitioner.partition(permuted, rho=0.5, seed=17)
    first_by_id = {
        original.bus_ids[i]: first.cell_of_bus[i] for i in range(len(first.cell_of_bus))
    }
    second_by_id = {
        permuted.bus_ids[i]: second.cell_of_bus[i]
        for i in range(len(second.cell_of_bus))
    }
    assert first_by_id == second_by_id
    assert [original.bus_ids[i] for i in first.anchors] == [
        permuted.bus_ids[i] for i in second.anchors
    ]


def test_real_pymetis_backend_is_contiguous_and_deterministic():
    partitioner = DeterministicPartitioner()
    topology = synthetic_topology()
    first = partitioner.partition(topology, rho=0.5, seed=17)
    second = partitioner.partition(topology, rho=0.5, seed=17)
    assert first == second
    assert first.algorithm == "metis-contiguous-repair-v1"
    for cell in range(3):
        members = {
            index for index, value in enumerate(first.cell_of_bus) if value == cell
        }
        frontier = {next(iter(members))}
        reached = set()
        while frontier:
            node = frontier.pop()
            if node in reached:
                continue
            reached.add(node)
            for source, target in topology.fine_edges:
                if source == node and target in members:
                    frontier.add(target)
                if target == node and source in members:
                    frontier.add(source)
        assert reached == members


def test_partition_repairs_empty_backend_cells_deterministically():
    def missing_cell_backend(adjacency, num_parts, seed):
        del adjacency, seed
        assert num_parts == 3
        return [0, 0, 0, 0, 1, 1]

    partitioner = DeterministicPartitioner(missing_cell_backend)
    topology = synthetic_topology()
    first = partitioner.partition(topology, rho=0.5, seed=17)
    second = partitioner.partition(topology, rho=0.5, seed=17)

    assert first == second
    assert first.cell_of_bus == (0, 0, 0, 1, 2, 2)
    assert len(first.anchors) == 3


def test_partition_repairs_disconnected_backend_cells():
    def disconnected_backend(adjacency, num_parts, seed):
        del adjacency, seed
        assert num_parts == 3
        return [0, 1, 0, 1, 2, 2]

    partition = DeterministicPartitioner(disconnected_backend).partition(
        synthetic_topology(),
        rho=0.5,
        seed=17,
    )

    assert partition.cell_of_bus == (0, 0, 1, 1, 2, 2)


def test_partition_uses_two_cells_for_small_positive_rho():
    def two_cell_backend(adjacency, num_parts, seed):
        del adjacency, seed
        assert num_parts == 2
        return [0, 0, 0, 1, 1, 1]

    partition = DeterministicPartitioner(two_cell_backend).partition(
        synthetic_topology(),
        rho=0.01,
        seed=17,
    )

    assert partition.num_cells == 2


def test_geometry_bundle_round_trip_and_device_cache(tmp_path):
    kron_builder, quotient_builder = builders()
    topology = synthetic_topology()
    geometries = [
        kron_builder.build(topology, budget()),
        quotient_builder.build(topology, budget()),
    ]
    path = tmp_path / "geometry.pt"
    digest = save_geometry_bundle(path, geometries)
    restored, restored_digest = load_geometry_bundle(path)

    assert restored_digest == digest
    assert {item.geometry_hash for item in restored} == {
        item.geometry_hash for item in geometries
    }
    registry = GeometryRegistry(restored)
    first = registry.for_device(
        "kron",
        "synthetic-6",
        torch.device("cpu"),
        torch.float32,
    )
    second = registry.for_device(
        "kron",
        "synthetic-6",
        torch.device("cpu"),
        torch.float32,
    )
    assert first is second
    assert first.coarse_attribute.shape[1] == 4


def test_geometry_identity_excludes_measured_build_diagnostics():
    kron_builder, _ = builders()
    geometry = kron_builder.build(synthetic_topology(), budget())
    changed = replace(
        geometry,
        harmonic_residual=geometry.harmonic_residual + 0.01,
        condition_number=geometry.condition_number + 1,
        provenance=replace(
            geometry.provenance,
            build_seconds=geometry.provenance.build_seconds + 100,
            dense_bytes=geometry.provenance.dense_bytes + 100,
        ),
    )
    assert changed.geometry_hash == geometry.geometry_hash


def test_common_sparsity_cap_fails_closed():
    kron_builder, quotient_builder = builders()
    topology = synthetic_topology()
    with pytest.raises(ContractError, match="exceeds"):
        kron_builder.build(topology, budget(kappa=0.1))
    with pytest.raises(ContractError, match="exceeds"):
        quotient_builder.build(topology, budget(kappa=0.1))
