# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable contracts for the confirmatory communication-only path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


SCHEMA_VERSION = "fm-scaling-v1"
GeometryKind = Literal["kron", "quotient"]


class ContractError(ValueError):
    """Raised when an artifact violates the frozen experiment contract."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    """Hash a dataclass or JSON-compatible value with canonical ordering."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ComplexCOO:
    """Portable complex sparse matrix represented without mutable arrays."""

    shape: tuple[int, int]
    row: tuple[int, ...]
    col: tuple[int, ...]
    value: tuple[complex, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ContractError(f"invalid sparse shape {self.shape}")
        if not (len(self.row) == len(self.col) == len(self.value)):
            raise ContractError("COO row, col, and value lengths differ")
        if any(index < 0 or index >= self.shape[0] for index in self.row):
            raise ContractError("COO row index out of range")
        if any(index < 0 or index >= self.shape[1] for index in self.col):
            raise ContractError("COO column index out of range")
        if len(set(zip(self.row, self.col))) != len(self.row):
            raise ContractError("COO coordinates must be unique")


@dataclass(frozen=True)
class GridTopology:
    """Topology-only geometry input; scenario and solver fields do not exist."""

    topology_key: str
    bus_ids: tuple[int, ...]
    fine_edges: tuple[tuple[int, int], ...]
    y_bus: ComplexCOO
    base_mva: float
    provenance_group: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        n_bus = len(self.bus_ids)
        if not self.topology_key or not self.provenance_group:
            raise ContractError("topology_key and provenance_group are required")
        if n_bus < 2 or len(set(self.bus_ids)) != n_bus:
            raise ContractError("bus IDs must be unique and contain at least two buses")
        if self.y_bus.shape != (n_bus, n_bus):
            raise ContractError("Y-bus shape does not match bus count")
        if not self.base_mva > 0:
            raise ContractError("case-declared baseMVA must be positive")
        for source, target in self.fine_edges:
            if source == target or min(source, target) < 0:
                raise ContractError("fine edges must be non-self edges")
            if max(source, target) >= n_bus:
                raise ContractError("fine edge index out of range")

    @property
    def topology_hash(self) -> str:
        return stable_hash(self)

    @property
    def undirected_edge_count(self) -> int:
        return len({tuple(sorted(edge)) for edge in self.fine_edges})


@dataclass(frozen=True)
class Partition:
    """Cell assignment in the topology's original tensor order."""

    cell_of_bus: tuple[int, ...]
    anchors: tuple[int, ...]
    seed: int
    algorithm: str = "metis-contiguous"

    def __post_init__(self) -> None:
        if not self.cell_of_bus or not self.anchors:
            raise ContractError("partition cannot be empty")
        cells = set(self.cell_of_bus)
        if cells != set(range(len(self.anchors))):
            raise ContractError(
                "partition cell labels must be canonical and contiguous",
            )
        if len(set(self.anchors)) != len(self.anchors):
            raise ContractError("partition anchors must be unique")
        if any(index < 0 or index >= len(self.cell_of_bus) for index in self.anchors):
            raise ContractError("anchor index out of range")
        for cell, anchor in enumerate(self.anchors):
            if self.cell_of_bus[anchor] != cell:
                raise ContractError("anchor must belong to its canonical cell")

    @property
    def num_cells(self) -> int:
        return len(self.anchors)


@dataclass(frozen=True)
class GeometryBudget:
    """One preregistered source-only geometry policy."""

    rho: float
    k_p: int
    k_c: int
    kappa: float
    metis_seed: int
    max_condition: float
    max_harmonic_residual: float
    max_dense_bytes: int
    max_build_seconds: float

    def __post_init__(self) -> None:
        if not 0 < self.rho < 1:
            raise ContractError("rho must lie strictly between zero and one")
        if min(self.k_p, self.k_c) < 1 or self.kappa <= 0:
            raise ContractError("sparsity parameters must be positive")
        if (
            min(
                self.max_condition,
                self.max_harmonic_residual,
                self.max_dense_bytes,
                self.max_build_seconds,
            )
            <= 0
        ):
            raise ContractError("resource and validity limits must be positive")

    @property
    def policy_hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True)
class SparseOperator:
    """Sparse map with complex attributes and real conservative weights."""

    output_size: int
    input_size: int
    row: tuple[int, ...]
    col: tuple[int, ...]
    coefficient: tuple[complex, ...]
    weight: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.row)
        if not (size == len(self.col) == len(self.coefficient) == len(self.weight)):
            raise ContractError("sparse operator fields have unequal lengths")
        if self.output_size < 1 or self.input_size < 1:
            raise ContractError("sparse operator dimensions must be positive")
        if any(i < 0 or i >= self.output_size for i in self.row):
            raise ContractError("operator row out of range")
        if any(i < 0 or i >= self.input_size for i in self.col):
            raise ContractError("operator column out of range")
        if any(weight <= 0 for weight in self.weight):
            raise ContractError("transport weights must be strictly positive")

    @property
    def nnz(self) -> int:
        return len(self.row)


@dataclass(frozen=True)
class SparseGraph:
    """Directed coarse graph with a common complex edge schema."""

    num_nodes: int
    source: tuple[int, ...]
    target: tuple[int, ...]
    coefficient: tuple[complex, ...]

    def __post_init__(self) -> None:
        if not (len(self.source) == len(self.target) == len(self.coefficient)):
            raise ContractError("coarse graph fields have unequal lengths")
        if self.num_nodes < 1:
            raise ContractError("coarse graph must contain a node")
        if any(i < 0 or i >= self.num_nodes for i in self.source + self.target):
            raise ContractError("coarse edge index out of range")
        if any(a == b for a, b in zip(self.source, self.target)):
            raise ContractError("coarse graph excludes self edges")

    @property
    def nnz(self) -> int:
        return len(self.source)


@dataclass(frozen=True)
class GeometryProvenance:
    topology_hash: str
    policy_hash: str
    builder: GeometryKind
    schema_version: str
    build_seconds: float
    dense_bytes: int


@dataclass(frozen=True)
class HierarchyGeometry:
    """Runtime geometry consumed by the hierarchy communication core."""

    topology_key: str
    kind: GeometryKind
    partition: Partition
    interior: tuple[int, ...]
    restrict: SparseOperator
    prolong: SparseOperator
    coarse_graph: SparseGraph
    harmonic_residual: float
    condition_number: float
    provenance: GeometryProvenance

    def __post_init__(self) -> None:
        if self.partition.num_cells != self.coarse_graph.num_nodes:
            raise ContractError("partition/coarse-node count mismatch")
        if self.restrict.output_size != self.partition.num_cells:
            raise ContractError("restriction output does not match coarse count")
        if self.prolong.input_size != self.partition.num_cells:
            raise ContractError("prolongation input does not match coarse count")
        if self.restrict.input_size != len(self.interior):
            raise ContractError("restriction input does not match interior count")
        if self.prolong.output_size != len(self.interior):
            raise ContractError("prolongation output does not match interior count")
        if set(self.interior) & set(self.partition.anchors):
            raise ContractError("anchors and interior must be disjoint")
        if len(self.interior) + len(self.partition.anchors) != len(
            self.partition.cell_of_bus,
        ):
            raise ContractError("anchors and interior must cover every bus")
        if self.harmonic_residual < 0 or self.condition_number < 0:
            raise ContractError("geometry diagnostics cannot be negative")

    @property
    def geometry_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": self.provenance.schema_version,
                "topology_hash": self.provenance.topology_hash,
                "policy_hash": self.provenance.policy_hash,
                "builder": self.provenance.builder,
                "topology_key": self.topology_key,
                "kind": self.kind,
                "partition": self.partition,
                "interior": self.interior,
                "restrict": self.restrict,
                "prolong": self.prolong,
                "coarse_graph": self.coarse_graph,
            },
        )

    @property
    def runtime_nnz(self) -> int:
        return self.prolong.nnz + self.coarse_graph.nnz
