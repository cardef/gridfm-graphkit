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


def _connected_components(
    members: set[int],
    adjacency: Sequence[Sequence[int]],
) -> list[set[int]]:
    """Return deterministically ordered connected components of ``members``."""
    remaining = set(members)
    components = []
    while remaining:
        start = min(remaining)
        reached = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in remaining and neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        remaining.difference_update(reached)
        components.append(reached)
    return components


def _split_connected_region(
    members: set[int],
    adjacency: Sequence[Sequence[int]],
) -> tuple[set[int], set[int]]:
    """Split a connected region while preserving connectivity on both sides."""
    if len(members) < 2:
        raise ContractError("cannot split a singleton partition cell")

    root = min(members)
    parent = {root: None}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in members and neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if set(parent) != members:
        raise ContractError("cannot split a disconnected partition cell")

    parents = {value for value in parent.values() if value is not None}
    leaf = max(node for node in members if node not in parents)
    remainder = members - {leaf}
    if not _is_connected(remainder, adjacency):
        raise ContractError("deterministic partition split broke connectivity")
    return remainder, {leaf}


def _repair_membership(
    raw: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    num_parts: int,
) -> tuple[int, ...]:
    """Repair empty or disconnected METIS cells deterministically.

    METIS may leave labels empty when ``nparts`` is large relative to a sparse
    graph, and some library versions have returned disconnected labels despite
    ``contig=1``. Start from the seeded METIS assignment, split labels into
    their connected components, then merge adjacent fragments or split
    connected cells until the requested cardinality is exact.
    """
    if any(cell < 0 for cell in raw):
        raise ContractError("partition backend returned a negative cell label")

    regions = []
    for cell in sorted(set(raw)):
        members = {index for index, assigned in enumerate(raw) if assigned == cell}
        regions.extend(_connected_components(members, adjacency))

    while len(regions) > num_parts:
        region_of = {
            node: region_index
            for region_index, members in enumerate(regions)
            for node in members
        }
        adjacent_pairs = set()
        for source, neighbors in enumerate(adjacency):
            for target in neighbors:
                left = region_of[source]
                right = region_of[target]
                if left != right:
                    adjacent_pairs.add(tuple(sorted((left, right))))
        if not adjacent_pairs:
            raise ContractError(
                "partition fragments cannot be merged without breaking connectivity",
            )
        left, right = min(
            adjacent_pairs,
            key=lambda pair: (
                len(regions[pair[0]]) + len(regions[pair[1]]),
                min(regions[pair[0]] | regions[pair[1]]),
                max(regions[pair[0]] | regions[pair[1]]),
            ),
        )
        merged = regions[left] | regions[right]
        regions = [
            members
            for index, members in enumerate(regions)
            if index not in {left, right}
        ]
        regions.append(merged)

    while len(regions) < num_parts:
        splittable = [
            (index, members)
            for index, members in enumerate(regions)
            if len(members) > 1
        ]
        if not splittable:
            raise ContractError("partition cells cannot be split to requested count")
        index, members = min(
            splittable,
            key=lambda item: (-len(item[1]), min(item[1])),
        )
        remainder, leaf = _split_connected_region(members, adjacency)
        regions[index] = remainder
        regions.append(leaf)

    if len(regions) != num_parts or any(
        not _is_connected(members, adjacency) for members in regions
    ):
        raise ContractError("deterministic partition repair failed")

    ordered = sorted(regions, key=min)
    membership = [0] * len(raw)
    for cell, members in enumerate(ordered):
        for node in members:
            membership[node] = cell
    return tuple(membership)


class DeterministicPartitioner:
    """Stable-ID wrapper around a partition backend such as contiguous METIS."""

    def __init__(self, backend: PartitionBackend = pymetis_backend):
        self.backend = backend

    def partition(self, topology: GridTopology, rho: float, seed: int) -> Partition:
        adjacency, stable_order = _stable_adjacency(topology)
        num_parts = max(2, math.ceil(rho * len(stable_order)))
        if not 1 < num_parts < len(stable_order):
            raise ContractError(
                f"rho={rho} yields {num_parts} cells for {len(stable_order)} buses",
            )

        raw = tuple(int(cell) for cell in self.backend(adjacency, num_parts, seed))
        if len(raw) != len(stable_order):
            raise ContractError(
                "partition backend returned the wrong membership length",
            )
        repaired = _repair_membership(raw, adjacency, num_parts)
        raw_cells = set(repaired)
        members_by_raw = {
            cell: {i for i, assigned in enumerate(repaired) if assigned == cell}
            for cell in raw_cells
        }
        # Backend cell labels carry no meaning. Canonicalize them by the
        # minimum stable bus ID in each cell before restoring tensor order.
        ordered_raw = sorted(
            raw_cells,
            key=lambda cell: min(
                topology.bus_ids[stable_order[i]] for i in members_by_raw[cell]
            ),
        )
        canonical = {raw_cell: cell for cell, raw_cell in enumerate(ordered_raw)}
        stable_cells = [canonical[cell] for cell in repaired]
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
            algorithm="metis-contiguous-repair-v1",
        )
