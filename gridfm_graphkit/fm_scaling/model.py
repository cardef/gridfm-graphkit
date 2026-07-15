# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Common PF backbone whose sole treatment is the communication core."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from gridfm_graphkit.datasets.globals import PG_H, VA_H, VM_H
from gridfm_graphkit.fm_scaling.communication import (
    GlobalSummaryCore,
    HeteroLatents,
    HeteroLocalBlock,
    HierarchyCore,
    LocalCore,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.registry import GeometryRegistry
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    PhysicsDecoderPF,
)


BUS_EDGE = ("bus", "connects", "bus")
GEN_TO_BUS = ("gen", "connected_to", "bus")


def _aggregate_generators(
    generator_output: torch.Tensor,
    gen_to_bus_index: torch.Tensor,
    num_bus: int,
) -> torch.Tensor:
    output = generator_output.new_zeros(num_bus)
    output.index_add_(0, gen_to_bus_index, generator_output.flatten())
    return output


@MODELS_REGISTRY.register("FMScalingPF")
class FMScalingPF(nn.Module):
    """One encoder/stem/slot/readout/decoder across all four experiment arms."""

    def __init__(self, args) -> None:
        super().__init__()
        if args.task.task_name not in {"PowerFlow", "FMScalingPowerFlow"}:
            raise ContractError("FMScalingPF is restricted to the PowerFlow task")
        if args.data.normalization != "CaseDeclaredMVANormalizer":
            raise ContractError("confirmatory runs require CaseDeclaredMVANormalizer")
        if not getattr(args.data, "confirmatory", False):
            raise ContractError("data.confirmatory must be true")
        if getattr(getattr(args.data, "hierarchy", None), "enable", False):
            raise ContractError("legacy data.hierarchy must remain disabled")

        config = args.model
        self.variant = str(config.communication_core).lower()
        if self.variant not in {"flat", "global", "kron", "quotient"}:
            raise ContractError(f"unknown communication core {self.variant}")
        self.width = int(config.hidden_size)
        self.edge_dim = int(config.edge_dim)
        self.geometry_bundle_hash: str | None = None

        self.bus_encoder = nn.Sequential(
            nn.Linear(int(config.input_bus_dim), self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
            nn.LayerNorm(self.width),
        )
        self.gen_encoder = nn.Sequential(
            nn.Linear(int(config.input_gen_dim), self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
            nn.LayerNorm(self.width),
        )
        self.pre = nn.ModuleList(
            HeteroLocalBlock(self.width, self.edge_dim)
            for _ in range(int(config.l_pre))
        )
        self.post = nn.ModuleList(
            HeteroLocalBlock(self.width, self.edge_dim)
            for _ in range(int(config.l_post))
        )

        if self.variant == "flat":
            self.communication_core = LocalCore(
                self.width,
                self.edge_dim,
                int(config.flat_blocks),
            )
        elif self.variant == "global":
            self.communication_core = GlobalSummaryCore(self.width, self.edge_dim)
        else:
            bundle_path = Path(str(config.geometry_bundle)).resolve()
            registry, self.geometry_bundle_hash = GeometryRegistry.from_bundle(
                bundle_path,
            )
            self.communication_core = HierarchyCore(
                self.width,
                registry,
                self.variant,
            )

        self.bus_decoder = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.LayerNorm(self.width),
            nn.SiLU(),
            nn.Linear(self.width, 2),
        )
        self.gen_decoder = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.LayerNorm(self.width),
            nn.SiLU(),
            nn.Linear(self.width, 1),
        )
        self.branch_flow = ComputeBranchFlow()
        self.node_injection = ComputeNodeInjection()
        self.physics_decoder = PhysicsDecoderPF()

    def forward(self, batch) -> dict[str, torch.Tensor]:
        latents = HeteroLatents(
            bus=self.bus_encoder(batch.x_dict["bus"]),
            gen=self.gen_encoder(batch.x_dict["gen"]),
        )
        for block in self.pre:
            latents = block(latents, batch)
        latents = self.communication_core(latents, batch)
        for block in self.post:
            latents = block(latents, batch)

        bus_prediction = self.bus_decoder(latents.bus)
        gen_prediction = self.gen_decoder(latents.gen)

        # Known quantities come from masked model inputs, never target tensors.
        bus_unknown = batch.mask_dict["bus"][:, VM_H : VA_H + 1]
        bus_fixed = batch.x_dict["bus"][:, VM_H : VA_H + 1]
        bus_prediction = torch.where(bus_unknown, bus_prediction, bus_fixed)
        gen_unknown = batch.mask_dict["gen"][:, PG_H : PG_H + 1]
        gen_fixed = batch.x_dict["gen"][:, PG_H : PG_H + 1]
        gen_prediction = torch.where(gen_unknown, gen_prediction, gen_fixed)

        bus_edge_index = batch.edge_index_dict[BUS_EDGE]
        bus_edge_attr = batch.edge_attr_dict[BUS_EDGE]
        p_flow, q_flow = self.branch_flow(
            bus_prediction,
            bus_edge_index,
            bus_edge_attr,
        )
        p_in, q_in = self.node_injection(
            p_flow,
            q_flow,
            bus_edge_index,
            bus_prediction.size(0),
        )
        gen_to_bus = batch.edge_index_dict[GEN_TO_BUS][1]
        aggregated_generation = _aggregate_generators(
            gen_prediction,
            gen_to_bus,
            bus_prediction.size(0),
        )
        bus_output = self.physics_decoder(
            p_in,
            q_in,
            bus_prediction,
            batch.x_dict["bus"],
            aggregated_generation,
            batch.mask_dict,
        )
        return {"bus": bus_output, "gen": gen_prediction}
