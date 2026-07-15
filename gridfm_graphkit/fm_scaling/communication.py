# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Interchangeable communication cores with one shared tensor contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.registry import GeometryRegistry


BUS_EDGE = ("bus", "connects", "bus")
GEN_TO_BUS = ("gen", "connected_to", "bus")
BUS_TO_GEN = ("bus", "connected_to", "gen")


def _scatter_sum(
    values: torch.Tensor,
    index: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = values.new_zeros((size, values.size(-1)))
    output.index_add_(0, index, values)
    return output


def _scatter_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = _scatter_sum(values, index, size)
    count = values.new_zeros(size)
    count.index_add_(0, index, values.new_ones(index.numel()))
    return output / count.clamp_min(1).unsqueeze(-1)


def _scatter_max(
    values: torch.Tensor,
    index: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = values.new_full((size, values.size(-1)), -torch.inf)
    expanded = index[:, None].expand_as(values)
    output.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


@dataclass(frozen=True)
class HeteroLatents:
    bus: torch.Tensor
    gen: torch.Tensor


class CommunicationCore(Protocol):
    def forward(self, latents: HeteroLatents, batch) -> HeteroLatents: ...  # noqa: E704


class EdgeConditionedBlock(nn.Module):
    """Residual directed graph block used on both fine and coarse graphs."""

    def __init__(self, width: int, edge_dim: int):
        super().__init__()
        self.source = nn.Linear(width, width, bias=False)
        self.edge = nn.Sequential(
            nn.Linear(edge_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attribute: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        message = self.source(hidden[source]) + self.edge(edge_attribute)
        aggregate = _scatter_mean(message, target, hidden.size(0))
        return hidden + self.update(torch.cat([hidden, aggregate], dim=-1))


class HeteroLocalBlock(nn.Module):
    """One relation-aware fine-grid block shared by every headline arm."""

    def __init__(self, width: int, edge_dim: int):
        super().__init__()
        self.bus_source = nn.Linear(width, width, bias=False)
        self.bus_edge = nn.Sequential(
            nn.Linear(edge_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.gen_to_bus = nn.Linear(width, width, bias=False)
        self.bus_to_gen = nn.Linear(width, width, bias=False)
        self.bus_update = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.gen_update = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, latents: HeteroLatents, batch) -> HeteroLatents:
        bus_edge_index = batch.edge_index_dict[BUS_EDGE]
        bus_edge_attr = batch.edge_attr_dict[BUS_EDGE]
        bus_source, bus_target = bus_edge_index
        bus_messages = self.bus_source(latents.bus[bus_source]) + self.bus_edge(
            bus_edge_attr,
        )
        bus_from_bus = _scatter_mean(
            bus_messages,
            bus_target,
            latents.bus.size(0),
        )

        gen_to_bus = batch.edge_index_dict[GEN_TO_BUS]
        gen_messages = self.gen_to_bus(latents.gen[gen_to_bus[0]])
        bus_from_gen = _scatter_mean(
            gen_messages,
            gen_to_bus[1],
            latents.bus.size(0),
        )
        bus = latents.bus + self.bus_update(
            torch.cat([latents.bus, bus_from_bus, bus_from_gen], dim=-1),
        )

        bus_to_gen = batch.edge_index_dict[BUS_TO_GEN]
        generator_messages = self.bus_to_gen(latents.bus[bus_to_gen[0]])
        gen_from_bus = _scatter_mean(
            generator_messages,
            bus_to_gen[1],
            latents.gen.size(0),
        )
        gen = latents.gen + self.gen_update(
            torch.cat([latents.gen, gen_from_bus], dim=-1),
        )
        return HeteroLatents(bus=bus, gen=gen)


class LocalCore(nn.Module):
    def __init__(self, width: int, edge_dim: int, num_blocks: int):
        super().__init__()
        if num_blocks < 1:
            raise ContractError("Flat communication slot needs at least one block")
        self.blocks = nn.ModuleList(
            HeteroLocalBlock(width, edge_dim) for _ in range(num_blocks)
        )

    def forward(self, latents: HeteroLatents, batch) -> HeteroLatents:
        for block in self.blocks:
            latents = block(latents, batch)
        return latents


class GlobalSummaryCore(nn.Module):
    """One typewise mean-plus-max summary, broadcast, then one local block."""

    def __init__(self, width: int, edge_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, width),
            nn.SiLU(),
        )
        self.local = HeteroLocalBlock(width, edge_dim)

    def forward(self, latents: HeteroLatents, batch) -> HeteroLatents:
        bus_batch = batch.batch_dict["bus"]
        gen_batch = batch.batch_dict["gen"]
        num_graphs = int(batch.num_graphs)
        summary = torch.cat(
            [
                _scatter_mean(latents.bus, bus_batch, num_graphs),
                _scatter_max(latents.bus, bus_batch, num_graphs),
                _scatter_mean(latents.gen, gen_batch, num_graphs),
                _scatter_max(latents.gen, gen_batch, num_graphs),
            ],
            dim=-1,
        )
        broadcast = self.project(summary)
        latents = HeteroLatents(
            bus=latents.bus + broadcast[bus_batch],
            gen=latents.gen + broadcast[gen_batch],
        )
        return self.local(latents, batch)


def _topology_keys(batch) -> list[str]:
    value = getattr(batch, "topology_key", None)
    if value is None:
        raise ContractError("every confirmatory sample must carry topology_key")
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ContractError(
        f"unsupported batched topology_key value {type(value).__name__}",
    )


def _bus_ptr(batch) -> torch.Tensor:
    store = batch["bus"]
    if hasattr(store, "ptr"):
        return store.ptr
    return torch.tensor([0, store.x.size(0)], device=store.x.device)


class HierarchyCore(nn.Module):
    """One down/coarse/up latent adapter for Kron or Quotient geometry."""

    def __init__(self, width: int, registry: GeometryRegistry, kind: str):
        super().__init__()
        if kind not in {"kron", "quotient"}:
            raise ContractError(f"unsupported hierarchy kind {kind}")
        self.registry = registry
        self.kind = kind
        self.phi_down = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.phi_up = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.coarse = EdgeConditionedBlock(width, edge_dim=4)

    def forward(self, latents: HeteroLatents, batch) -> HeteroLatents:
        keys = _topology_keys(batch)
        ptr = _bus_ptr(batch)
        if len(keys) != ptr.numel() - 1:
            raise ContractError("topology_key count does not match batched bus graphs")
        output = latents.bus.clone()
        for graph_index, topology_key in enumerate(keys):
            start = int(ptr[graph_index])
            stop = int(ptr[graph_index + 1])
            geometry = self.registry.for_device(
                self.kind,
                topology_key,
                latents.bus.device,
                latents.bus.dtype,
            )
            if geometry.anchors.numel() + geometry.interior.numel() != stop - start:
                raise ContractError(
                    f"geometry bus count mismatch for {topology_key}: "
                    f"{geometry.anchors.numel() + geometry.interior.numel()} != "
                    f"{stop - start}",
                )
            hidden = latents.bus[start:stop]
            interior_hidden = hidden[geometry.interior]
            anchor_hidden = hidden[geometry.anchors]

            down_messages = interior_hidden[
                geometry.restrict.col
            ] * geometry.restrict.weight.unsqueeze(-1)
            restricted = _scatter_sum(
                down_messages,
                geometry.restrict.row,
                anchor_hidden.size(0),
            )
            coarse_hidden = anchor_hidden + self.phi_down(restricted)
            coarse_hidden = self.coarse(
                coarse_hidden,
                torch.stack(
                    [geometry.coarse_source, geometry.coarse_target],
                    dim=0,
                ),
                geometry.coarse_attribute,
            )

            up_messages = coarse_hidden[
                geometry.prolong.col
            ] * geometry.prolong.weight.unsqueeze(-1)
            prolonged = _scatter_sum(
                up_messages,
                geometry.prolong.row,
                interior_hidden.size(0),
            )
            updated_interior = interior_hidden + self.phi_up(
                torch.cat([interior_hidden, prolonged], dim=-1),
            )
            graph_output = hidden.clone()
            graph_output[geometry.interior] = updated_interior
            graph_output[geometry.anchors] = coarse_hidden
            output[start:stop] = graph_output
        return HeteroLatents(bus=output, gen=latents.gen)
