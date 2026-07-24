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
    """Split a connected region into two cells of at least two buses."""
    if len(members) < 4:
        raise ContractError("partition cell is too small for a covered split")

    root = min(members)
    parent = {}
    stack = [(root, None)]
    order = []
    while stack:
        node, predecessor = stack.pop()
        if node in parent:
            continue
        parent[node] = predecessor
        order.append(node)
        for neighbor in reversed(adjacency[node]):
            if neighbor in members and neighbor not in parent:
                stack.append((neighbor, node))
    if set(parent) != members:
        raise ContractError("cannot split a disconnected partition cell")

    subtrees = {node: {node} for node in members}
    for node in reversed(order[1:]):
        subtrees[parent[node]].update(subtrees[node])
    candidates = [
        node
        for node in order[1:]
        if 2 <= len(subtrees[node]) <= len(members) - 2
    ]
    if not candidates:
        raise ContractError("connected partition cell has no covered tree split")
    selected = min(
        candidates,
        key=lambda node: (
            len(subtrees[node]),
            min(subtrees[node]),
            node,
        ),
    )
    right = subtrees[selected]
    left = members - right
    if not _is_connected(left, adjacency) or not _is_connected(right, adjacency):
        raise ContractError("deterministic partition split broke connectivity")
    return left, right


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

    # A coarse cell needs at least one non-anchor bus so both Kron and
    # Quotient transports cover every coarse column. First borrow a removable
    # adjacent bus without changing cardinality; only merge a singleton when
    # no connectivity-preserving local move exists.
    while any(len(members) == 1 for members in regions):
        _, singleton_index = min(
            (min(members), index)
            for index, members in enumerate(regions)
            if len(members) == 1
        )
        singleton = regions[singleton_index]
        node = next(iter(singleton))
        region_of = {
            member: region_index
            for region_index, members in enumerate(regions)
            for member in members
        }
        borrowable = []
        for neighbor in adjacency[node]:
            donor_index = region_of[neighbor]
            if donor_index == singleton_index:
                continue
            donor = regions[donor_index]
            remainder = donor - {neighbor}
            if len(remainder) >= 2 and _is_connected(remainder, adjacency):
                borrowable.append((donor_index, neighbor))
        if borrowable:
            donor_index, neighbor = min(
                borrowable,
                key=lambda item: (
                    len(regions[item[0]]),
                    item[1],
                    min(regions[item[0]]),
                    max(regions[item[0]]),
                ),
            )
            regions[singleton_index] = singleton | {neighbor}
            regions[donor_index] = regions[donor_index] - {neighbor}
            continue

        adjacent = {
            region_of[neighbor]
            for neighbor in adjacency[node]
            if region_of[neighbor] != singleton_index
        }
        if not adjacent:
            raise ContractError(
                "singleton partition fragment has no adjacent repair cell",
            )
        neighbor_index = min(
            adjacent,
            key=lambda index: (
                len(regions[index]),
                min(regions[index]),
                max(regions[index]),
            ),
        )
        merged = singleton | regions[neighbor_index]
        regions = [
            members
            for index, members in enumerate(regions)
            if index not in {singleton_index, neighbor_index}
        ]
        regions.append(merged)

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
        splittable = []
        for index, members in enumerate(regions):
            if len(members) < 4:
                continue
            try:
                left, right = _split_connected_region(members, adjacency)
            except ContractError:
                continue
            splittable.append((index, members, left, right))
        if not splittable:
            raise ContractError(
                "partition cells cannot be split to requested covered count",
            )
        index, members, left, right = min(
            splittable,
            key=lambda item: (-len(item[1]), min(item[1])),
        )
        regions[index] = left
        regions.append(right)

    if len(regions) != num_parts or any(
        len(members) < 2 or not _is_connected(members, adjacency)
        for members in regions
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
        num_parts = math.ceil(rho * len(stable_order))
        if not 0 < num_parts < len(stable_order):
            raise ContractError(
                f"rho={rho} yields {num_parts} cells for {len(stable_order)} buses",
            )
        if num_parts > 1 and 2 * num_parts > len(stable_order):
            raise ContractError(
                f"{num_parts} covered cells require at least {2 * num_parts} buses",
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
            algorithm="metis-contiguous-covered-repair-v3",
        )
