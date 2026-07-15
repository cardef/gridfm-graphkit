# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Kron--Schur and same-partition quotient geometry builders."""

from __future__ import annotations

import math
import time
from typing import Protocol

import numpy as np

from gridfm_graphkit.fm_scaling.contracts import (
    ContractError,
    GeometryBudget,
    GeometryProvenance,
    GridTopology,
    HierarchyGeometry,
    SparseGraph,
    SparseOperator,
)
from gridfm_graphkit.fm_scaling.partition import DeterministicPartitioner


class GeometryBuilder(Protocol):
    def build(
        self,
        topology: GridTopology,
        budget: GeometryBudget,
    ) -> HierarchyGeometry:
        pass


def _dense_y_bus(topology: GridTopology) -> np.ndarray:
    matrix = np.zeros(topology.y_bus.shape, dtype=np.complex128)
    matrix[
        np.asarray(topology.y_bus.row, dtype=np.int64),
        np.asarray(topology.y_bus.col, dtype=np.int64),
    ] = np.asarray(topology.y_bus.value, dtype=np.complex128)
    return matrix


def _topk_indices(values: np.ndarray, k: int, tie_ids: list[int]) -> list[int]:
    nonzero = [index for index, value in enumerate(values) if abs(value) > 0]
    return sorted(nonzero, key=lambda index: (-abs(values[index]), tie_ids[index]))[:k]


def _sparsify_harmonic_map(
    dense: np.ndarray,
    k: int,
    interior_ids: list[int],
    anchor_ids: list[int],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    selected: set[tuple[int, int]] = set()
    for row in range(dense.shape[0]):
        for col in _topk_indices(dense[row], k, anchor_ids):
            selected.add((row, col))
    for col in range(dense.shape[1]):
        rows = _topk_indices(dense[:, col], 1, interior_ids)
        if not rows:
            raise ContractError(f"harmonic-map column {col} has no nonzero entry")
        selected.add((rows[0], col))

    if len({row for row, _ in selected}) != dense.shape[0]:
        raise ContractError("harmonic sparsifier failed row coverage")
    if len({col for _, col in selected}) != dense.shape[1]:
        raise ContractError("harmonic sparsifier failed column coverage")

    ordered = sorted(selected)
    sparse = np.zeros_like(dense)
    for row, col in ordered:
        sparse[row, col] = dense[row, col]
    return ordered, sparse


def _sparsify_coarse_graph(
    dense: np.ndarray,
    k: int,
    anchor_ids: list[int],
) -> list[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    for row in range(dense.shape[0]):
        candidates = dense[row].copy()
        candidates[row] = 0
        for col in _topk_indices(candidates, k, anchor_ids):
            selected.add((row, col))
    if not selected:
        raise ContractError("coarse graph has no off-diagonal edge")
    return sorted(selected)


def _transport_operators(
    coefficients: np.ndarray,
    coordinates: list[tuple[int, int]],
) -> tuple[SparseOperator, SparseOperator]:
    magnitude = np.abs(coefficients)
    row_mass = magnitude.sum(axis=1)
    col_mass = magnitude.sum(axis=0)
    if np.any(row_mass <= 0) or np.any(col_mass <= 0):
        raise ContractError("transport support lacks row or column coverage")

    rows = tuple(row for row, _ in coordinates)
    cols = tuple(col for _, col in coordinates)
    coeff = tuple(complex(coefficients[row, col]) for row, col in coordinates)
    prolong_weight = tuple(
        float(magnitude[row, col] / row_mass[row]) for row, col in coordinates
    )
    restrict_weight = tuple(
        float(magnitude[row, col] / col_mass[col]) for row, col in coordinates
    )
    prolong = SparseOperator(
        output_size=coefficients.shape[0],
        input_size=coefficients.shape[1],
        row=rows,
        col=cols,
        coefficient=coeff,
        weight=prolong_weight,
    )
    restrict = SparseOperator(
        output_size=coefficients.shape[1],
        input_size=coefficients.shape[0],
        row=cols,
        col=rows,
        coefficient=coeff,
        weight=restrict_weight,
    )
    return restrict, prolong


def _check_resource_gates(
    *,
    budget: GeometryBudget,
    condition: float,
    residual: float,
    dense_bytes: int,
    build_seconds: float,
) -> None:
    failures = []
    if not math.isfinite(condition) or condition > budget.max_condition:
        failures.append(f"condition={condition:.6g}>{budget.max_condition:.6g}")
    if not math.isfinite(residual) or residual > budget.max_harmonic_residual:
        failures.append(
            f"harmonic_residual={residual:.6g}>{budget.max_harmonic_residual:.6g}",
        )
    if dense_bytes > budget.max_dense_bytes:
        failures.append(f"dense_bytes={dense_bytes}>{budget.max_dense_bytes}")
    if build_seconds > budget.max_build_seconds:
        failures.append(
            f"build_seconds={build_seconds:.6g}>{budget.max_build_seconds:.6g}",
        )
    if failures:
        raise ContractError("geometry resource gate failed: " + "; ".join(failures))


class KronGeometryBuilder:
    """Build sparse runtime geometry from dense Kron construction intermediates."""

    def __init__(self, partitioner: DeterministicPartitioner | None = None):
        self.partitioner = partitioner or DeterministicPartitioner()

    def build(
        self,
        topology: GridTopology,
        budget: GeometryBudget,
    ) -> HierarchyGeometry:
        started = time.perf_counter()
        partition = self.partitioner.partition(
            topology,
            rho=budget.rho,
            seed=budget.metis_seed,
        )
        anchors = list(partition.anchors)
        anchor_set = set(anchors)
        interior = [i for i in range(len(topology.bus_ids)) if i not in anchor_set]
        if not interior:
            raise ContractError("Kron construction requires an interior")

        y_bus = _dense_y_bus(topology)
        y_bb = y_bus[np.ix_(anchors, anchors)]
        y_bi = y_bus[np.ix_(anchors, interior)]
        y_ib = y_bus[np.ix_(interior, anchors)]
        y_ii = y_bus[np.ix_(interior, interior)]
        condition = float(np.linalg.cond(y_ii))
        try:
            harmonic = -np.linalg.solve(y_ii, y_ib)
            y_red = y_bb - y_bi @ np.linalg.solve(y_ii, y_ib)
        except np.linalg.LinAlgError as error:
            raise ContractError("Y_II solve failed") from error

        coordinates, sparse_harmonic = _sparsify_harmonic_map(
            harmonic,
            budget.k_p,
            [topology.bus_ids[i] for i in interior],
            [topology.bus_ids[i] for i in anchors],
        )
        coarse_coordinates = _sparsify_coarse_graph(
            y_red,
            budget.k_c,
            [topology.bus_ids[i] for i in anchors],
        )
        runtime_nnz = len(coordinates) + len(coarse_coordinates)
        cap = math.floor(budget.kappa * topology.undirected_edge_count)
        if runtime_nnz > cap:
            raise ContractError(
                f"runtime nnz {runtime_nnz} exceeds cap {cap}; coverage is mandatory",
            )

        denominator = max(float(np.linalg.norm(y_ib)), np.finfo(float).eps)
        residual = float(np.linalg.norm(y_ii @ sparse_harmonic + y_ib) / denominator)
        dense_bytes = sum(
            matrix.nbytes for matrix in (y_bus, y_bb, y_bi, y_ib, y_ii, harmonic, y_red)
        )
        build_seconds = time.perf_counter() - started
        _check_resource_gates(
            budget=budget,
            condition=condition,
            residual=residual,
            dense_bytes=dense_bytes,
            build_seconds=build_seconds,
        )

        restrict, prolong = _transport_operators(
            sparse_harmonic,
            coordinates,
        )
        coarse = SparseGraph(
            num_nodes=len(anchors),
            source=tuple(row for row, _ in coarse_coordinates),
            target=tuple(col for _, col in coarse_coordinates),
            coefficient=tuple(
                complex(y_red[row, col]) for row, col in coarse_coordinates
            ),
        )
        provenance = GeometryProvenance(
            topology_hash=topology.topology_hash,
            policy_hash=budget.policy_hash,
            builder="kron",
            schema_version=topology.schema_version,
            build_seconds=build_seconds,
            dense_bytes=dense_bytes,
        )
        return HierarchyGeometry(
            topology_key=topology.topology_key,
            kind="kron",
            partition=partition,
            interior=tuple(interior),
            restrict=restrict,
            prolong=prolong,
            coarse_graph=coarse,
            harmonic_residual=residual,
            condition_number=condition,
            provenance=provenance,
        )


class QuotientGeometryBuilder:
    """Build the same-partition assignment and complex cut-sum control."""

    def __init__(self, partitioner: DeterministicPartitioner | None = None):
        self.partitioner = partitioner or DeterministicPartitioner()

    def build(
        self,
        topology: GridTopology,
        budget: GeometryBudget,
    ) -> HierarchyGeometry:
        started = time.perf_counter()
        partition = self.partitioner.partition(
            topology,
            rho=budget.rho,
            seed=budget.metis_seed,
        )
        anchor_set = set(partition.anchors)
        interior = [i for i in range(len(topology.bus_ids)) if i not in anchor_set]
        interior_position = {bus: index for index, bus in enumerate(interior)}

        assignment = np.zeros((len(interior), partition.num_cells), dtype=np.complex128)
        coordinates = []
        for bus in interior:
            row = interior_position[bus]
            cell = partition.cell_of_bus[bus]
            assignment[row, cell] = 1 + 0j
            coordinates.append((row, cell))

        # Empty-interior cells still have their anchor state, but the common
        # conservative restriction requires each coarse column to receive at
        # least one interior message. Fail rather than silently changing policy.
        restrict, prolong = _transport_operators(assignment, coordinates)

        cut_sums: dict[tuple[int, int], complex] = {}
        for row, col, value in zip(
            topology.y_bus.row,
            topology.y_bus.col,
            topology.y_bus.value,
        ):
            if row == col:
                continue
            source_cell = partition.cell_of_bus[row]
            target_cell = partition.cell_of_bus[col]
            if source_cell == target_cell:
                continue
            key = (source_cell, target_cell)
            cut_sums[key] = cut_sums.get(key, 0j) + value
        cut_sums = {key: value for key, value in cut_sums.items() if abs(value) > 0}
        coarse_coordinates = sorted(cut_sums)
        if not coarse_coordinates:
            raise ContractError("quotient construction produced no coarse edge")

        runtime_nnz = len(coordinates) + len(coarse_coordinates)
        cap = math.floor(budget.kappa * topology.undirected_edge_count)
        if runtime_nnz > cap:
            raise ContractError(
                f"quotient runtime nnz {runtime_nnz} exceeds common cap {cap}",
            )
        build_seconds = time.perf_counter() - started
        dense_bytes = assignment.nbytes
        _check_resource_gates(
            budget=budget,
            condition=1.0,
            residual=0.0,
            dense_bytes=dense_bytes,
            build_seconds=build_seconds,
        )

        coarse = SparseGraph(
            num_nodes=partition.num_cells,
            source=tuple(row for row, _ in coarse_coordinates),
            target=tuple(col for _, col in coarse_coordinates),
            coefficient=tuple(cut_sums[key] for key in coarse_coordinates),
        )
        provenance = GeometryProvenance(
            topology_hash=topology.topology_hash,
            policy_hash=budget.policy_hash,
            builder="quotient",
            schema_version=topology.schema_version,
            build_seconds=build_seconds,
            dense_bytes=dense_bytes,
        )
        return HierarchyGeometry(
            topology_key=topology.topology_key,
            kind="quotient",
            partition=partition,
            interior=tuple(interior),
            restrict=restrict,
            prolong=prolong,
            coarse_graph=coarse,
            harmonic_residual=0.0,
            condition_number=1.0,
            provenance=provenance,
        )
