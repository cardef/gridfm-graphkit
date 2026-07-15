# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed geometry persistence and device registry."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from gridfm_graphkit.fm_scaling.contracts import (
    ContractError,
    GeometryProvenance,
    HierarchyGeometry,
    Partition,
    SparseGraph,
    SparseOperator,
)


def _complex_tensor(values: tuple[complex, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.complex128)


def _operator_payload(operator: SparseOperator) -> dict:
    return {
        "output_size": operator.output_size,
        "input_size": operator.input_size,
        "row": torch.tensor(operator.row, dtype=torch.int64),
        "col": torch.tensor(operator.col, dtype=torch.int64),
        "coefficient": _complex_tensor(operator.coefficient),
        "weight": torch.tensor(operator.weight, dtype=torch.float64),
    }


def _operator_from_payload(payload: dict) -> SparseOperator:
    return SparseOperator(
        output_size=int(payload["output_size"]),
        input_size=int(payload["input_size"]),
        row=tuple(int(value) for value in payload["row"].tolist()),
        col=tuple(int(value) for value in payload["col"].tolist()),
        coefficient=tuple(complex(value) for value in payload["coefficient"].tolist()),
        weight=tuple(float(value) for value in payload["weight"].tolist()),
    )


def geometry_to_payload(geometry: HierarchyGeometry) -> dict:
    return {
        "geometry_hash": geometry.geometry_hash,
        "topology_key": geometry.topology_key,
        "kind": geometry.kind,
        "partition": {
            "cell_of_bus": torch.tensor(
                geometry.partition.cell_of_bus,
                dtype=torch.int64,
            ),
            "anchors": torch.tensor(geometry.partition.anchors, dtype=torch.int64),
            "seed": geometry.partition.seed,
            "algorithm": geometry.partition.algorithm,
        },
        "interior": torch.tensor(geometry.interior, dtype=torch.int64),
        "restrict": _operator_payload(geometry.restrict),
        "prolong": _operator_payload(geometry.prolong),
        "coarse_graph": {
            "num_nodes": geometry.coarse_graph.num_nodes,
            "source": torch.tensor(
                geometry.coarse_graph.source,
                dtype=torch.int64,
            ),
            "target": torch.tensor(
                geometry.coarse_graph.target,
                dtype=torch.int64,
            ),
            "coefficient": _complex_tensor(geometry.coarse_graph.coefficient),
        },
        "harmonic_residual": geometry.harmonic_residual,
        "condition_number": geometry.condition_number,
        "provenance": {
            "topology_hash": geometry.provenance.topology_hash,
            "policy_hash": geometry.provenance.policy_hash,
            "builder": geometry.provenance.builder,
            "schema_version": geometry.provenance.schema_version,
            "build_seconds": geometry.provenance.build_seconds,
            "dense_bytes": geometry.provenance.dense_bytes,
        },
    }


def geometry_from_payload(payload: dict) -> HierarchyGeometry:
    partition_payload = payload["partition"]
    partition = Partition(
        cell_of_bus=tuple(
            int(value) for value in partition_payload["cell_of_bus"].tolist()
        ),
        anchors=tuple(int(value) for value in partition_payload["anchors"].tolist()),
        seed=int(partition_payload["seed"]),
        algorithm=str(partition_payload["algorithm"]),
    )
    graph_payload = payload["coarse_graph"]
    graph = SparseGraph(
        num_nodes=int(graph_payload["num_nodes"]),
        source=tuple(int(value) for value in graph_payload["source"].tolist()),
        target=tuple(int(value) for value in graph_payload["target"].tolist()),
        coefficient=tuple(
            complex(value) for value in graph_payload["coefficient"].tolist()
        ),
    )
    provenance = GeometryProvenance(**payload["provenance"])
    geometry = HierarchyGeometry(
        topology_key=str(payload["topology_key"]),
        kind=str(payload["kind"]),
        partition=partition,
        interior=tuple(int(value) for value in payload["interior"].tolist()),
        restrict=_operator_from_payload(payload["restrict"]),
        prolong=_operator_from_payload(payload["prolong"]),
        coarse_graph=graph,
        harmonic_residual=float(payload["harmonic_residual"]),
        condition_number=float(payload["condition_number"]),
        provenance=provenance,
    )
    if geometry.geometry_hash != payload["geometry_hash"]:
        raise ContractError("geometry payload hash mismatch")
    return geometry


def save_geometry_bundle(
    path: Path,
    geometries: list[HierarchyGeometry],
) -> str:
    """Write an explicit, weights-only-loadable geometry bundle."""
    entries = {}
    for geometry in geometries:
        key = f"{geometry.kind}:{geometry.topology_key}"
        if key in entries:
            raise ContractError(f"duplicate geometry {key}")
        entries[key] = geometry_to_payload(geometry)
    payload = {"format": "fm-scaling-geometry-v1", "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_geometry_bundle(path: Path) -> tuple[list[HierarchyGeometry], str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "fm-scaling-geometry-v1":
        raise ContractError("unsupported geometry bundle format")
    geometries = [
        geometry_from_payload(payload["entries"][key])
        for key in sorted(payload["entries"])
    ]
    return geometries, digest


@dataclass(frozen=True)
class DeviceOperator:
    row: torch.Tensor
    col: torch.Tensor
    coefficient: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class DeviceGeometry:
    topology_key: str
    kind: str
    geometry_hash: str
    anchors: torch.Tensor
    interior: torch.Tensor
    restrict: DeviceOperator
    prolong: DeviceOperator
    coarse_source: torch.Tensor
    coarse_target: torch.Tensor
    coarse_attribute: torch.Tensor


def complex_edge_attributes(values: torch.Tensor) -> torch.Tensor:
    """Encode complex coefficients as [Re, Im, magnitude, phase]."""
    return torch.stack(
        [values.real, values.imag, values.abs(), torch.angle(values)],
        dim=-1,
    )


class GeometryRegistry:
    """Own one immutable geometry per topology/kind and cache device views."""

    def __init__(self, geometries: list[HierarchyGeometry] | None = None):
        self._geometries: dict[tuple[str, str], HierarchyGeometry] = {}
        self._device_cache: dict[tuple[str, str, str], DeviceGeometry] = {}
        self._lock = threading.RLock()
        for geometry in geometries or []:
            self.register(geometry)

    @classmethod
    def from_bundle(cls, path: Path) -> tuple["GeometryRegistry", str]:
        geometries, digest = load_geometry_bundle(path)
        return cls(geometries), digest

    def register(self, geometry: HierarchyGeometry) -> None:
        key = (geometry.kind, geometry.topology_key)
        with self._lock:
            existing = self._geometries.get(key)
            if (
                existing is not None
                and existing.geometry_hash != geometry.geometry_hash
            ):
                raise ContractError(f"content collision for geometry {key}")
            self._geometries[key] = geometry

    def get(self, kind: str, topology_key: str) -> HierarchyGeometry:
        try:
            return self._geometries[(kind, topology_key)]
        except KeyError as error:
            raise ContractError(f"missing geometry {kind}:{topology_key}") from error

    def for_device(
        self,
        kind: str,
        topology_key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DeviceGeometry:
        geometry = self.get(kind, topology_key)
        cache_key = (geometry.geometry_hash, str(device), str(dtype))
        with self._lock:
            cached = self._device_cache.get(cache_key)
            if cached is not None:
                return cached

            def operator(value: SparseOperator) -> DeviceOperator:
                return DeviceOperator(
                    row=torch.tensor(value.row, device=device, dtype=torch.long),
                    col=torch.tensor(value.col, device=device, dtype=torch.long),
                    coefficient=torch.tensor(
                        value.coefficient,
                        device=device,
                        dtype=torch.complex64
                        if dtype == torch.float32
                        else torch.complex128,
                    ),
                    weight=torch.tensor(value.weight, device=device, dtype=dtype),
                )

            coarse_values = torch.tensor(
                geometry.coarse_graph.coefficient,
                device=device,
                dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
            )
            cached = DeviceGeometry(
                topology_key=topology_key,
                kind=kind,
                geometry_hash=geometry.geometry_hash,
                anchors=torch.tensor(
                    geometry.partition.anchors,
                    device=device,
                    dtype=torch.long,
                ),
                interior=torch.tensor(
                    geometry.interior,
                    device=device,
                    dtype=torch.long,
                ),
                restrict=operator(geometry.restrict),
                prolong=operator(geometry.prolong),
                coarse_source=torch.tensor(
                    geometry.coarse_graph.source,
                    device=device,
                    dtype=torch.long,
                ),
                coarse_target=torch.tensor(
                    geometry.coarse_graph.target,
                    device=device,
                    dtype=torch.long,
                ),
                coarse_attribute=complex_edge_attributes(coarse_values).to(dtype),
            )
            self._device_cache[cache_key] = cached
            return cached
