# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, stable-ID topology partitioning."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence

from gridfm_graphkit.fm_scaling.contracts import (
    ContractError,
    GridTopology,
    Partition,
)


PartitionBackend = Callable[[Sequence[Sequence[int]], int, int], Sequence[int]]


def pymetis_backend(
    adjacency: Sequence[Sequence[int]],
    num_parts: int,
    seed: int,
) -> Sequence[int]:
    """Run contiguous METIS with explicit options; fail if PyMetis is absent."""
    try:
        import pymetis
    except ImportError as error:
        raise ContractError(
            "pymetis is required for confirmatory partition construction",
        ) from error

    options = pymetis.Options(seed=seed, contig=1)
    _, membership = pymetis.part_graph(
        num_parts,
        adjacency=[list(neighbors) for neighbors in adjacency],
        options=options,
    )
    return membership


def _stable_adjacency(topology: GridTopology) -> tuple[list[list[int]], list[int]]:
    stable_order = sorted(
        range(len(topology.bus_ids)),
        key=topology.bus_ids.__getitem__,
    )
    stable_position = {original: stable for stable, original in enumerate(stable_order)}
    neighbors = [set() for _ in stable_order]
    for original_source, original_target in topology.fine_edges:
        source = stable_position[original_source]
        target = stable_position[original_target]
        neighbors[source].add(target)
        neighbors[target].add(source)
    adjacency = [sorted(items) for items in neighbors]
    if any(not items for items in adjacency):
        raise ContractError("METIS input contains an isolated bus")
    return adjacency, stable_order


def _is_connected(members: set[int], adjacency: Sequence[Sequence[int]]) -> bool:
    start = next(iter(members))
    reached = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in members and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached == members


class DeterministicPartitioner:
    """Stable-ID wrapper around a partition backend such as contiguous METIS."""

    def __init__(self, backend: PartitionBackend = pymetis_backend):
        self.backend = backend

    def partition(self, topology: GridTopology, rho: float, seed: int) -> Partition:
        adjacency, stable_order = _stable_adjacency(topology)
        num_parts = math.ceil(rho * len(stable_order))
        if not 1 < num_parts < len(stable_order):
            raise ContractError(
                f"rho={rho} yields {num_parts} cells for {len(stable_order)} buses",
            )

        raw = tuple(int(cell) for cell in self.backend(adjacency, num_parts, seed))
        if len(raw) != len(stable_order):
            raise ContractError(
                "partition backend returned the wrong membership length",
            )
        raw_cells = set(raw)
        if len(raw_cells) != num_parts or min(raw_cells) < 0:
            raise ContractError("partition backend returned empty or invalid cells")

        members_by_raw = {
            cell: {i for i, assigned in enumerate(raw) if assigned == cell}
            for cell in raw_cells
        }
        if any(
            not _is_connected(members, adjacency) for members in members_by_raw.values()
        ):
            raise ContractError("partition backend violated contiguous-cell policy")

        # Backend cell labels carry no meaning. Canonicalize them by the
        # minimum stable bus ID in each cell before restoring tensor order.
        ordered_raw = sorted(
            raw_cells,
            key=lambda cell: min(
                topology.bus_ids[stable_order[i]] for i in members_by_raw[cell]
            ),
        )
        canonical = {raw_cell: cell for cell, raw_cell in enumerate(ordered_raw)}
        stable_cells = [canonical[cell] for cell in raw]
        original_cells = [0] * len(stable_order)
        for stable_index, original_index in enumerate(stable_order):
            original_cells[original_index] = stable_cells[stable_index]

        degree = [0] * len(stable_order)
        for original_source, original_target in {
            tuple(sorted(edge)) for edge in topology.fine_edges
        }:
            degree[original_source] += 1
            degree[original_target] += 1

        anchors = []
        for cell in range(num_parts):
            members = [
                index
                for index, assigned in enumerate(original_cells)
                if assigned == cell
            ]
            anchor = min(
                members,
                key=lambda index: (-degree[index], topology.bus_ids[index]),
            )
            anchors.append(anchor)

        return Partition(
            cell_of_bus=tuple(original_cells),
            anchors=tuple(anchors),
            seed=seed,
        )
