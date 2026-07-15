# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Per-graph, per-component objective shared by every confirmatory arm."""

from __future__ import annotations

import torch
from torch import nn

from gridfm_graphkit.datasets.globals import VA_H, VA_OUT, VM_H, VM_OUT
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.io.registries import LOSS_REGISTRY
from gridfm_graphkit.training.loss import PBELoss


def _graph_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    masked_values = torch.where(mask, values, torch.zeros_like(values))
    totals = values.new_zeros(num_graphs)
    counts = values.new_zeros(num_graphs)
    totals.index_add_(0, batch, masked_values)
    counts.index_add_(0, batch, mask.to(values.dtype))
    if bool((counts == 0).any()):
        raise ContractError("every graph must expose each predicted PF component")
    return totals / counts


def _batch_metadata(mask_dict) -> tuple[torch.Tensor, int]:
    try:
        return mask_dict["_bus_batch"], int(mask_dict["_num_graphs"])
    except KeyError as error:
        raise ContractError(
            "confirmatory loss requires graph batch metadata",
        ) from error


@LOSS_REGISTRY.register("GraphBalancedMaskedVMVA")
class GraphBalancedMaskedVMVA(nn.Module):
    def __init__(self, loss_args, args):
        super().__init__()
        del loss_args, args

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        model=None,
        x_dict=None,
    ):
        del edge_index_dict, edge_attr_dict, model, x_dict
        batch, num_graphs = _batch_metadata(mask_dict)
        pred = pred_dict["bus"]
        target = target_dict["bus"]
        vm = _graph_masked_mean(
            (pred[:, VM_OUT] - target[:, VM_H]).square(),
            mask_dict["bus"][:, VM_H],
            batch,
            num_graphs,
        )
        va = _graph_masked_mean(
            (pred[:, VA_OUT] - target[:, VA_H]).square(),
            mask_dict["bus"][:, VA_H],
            batch,
            num_graphs,
        )
        loss = torch.stack([vm, va], dim=-1).mean()
        return {
            "loss": loss,
            "Graph-balanced VM MSE": vm.mean().detach(),
            "Graph-balanced VA MSE": va.mean().detach(),
        }


@LOSS_REGISTRY.register("GraphBalancedPBE")
class GraphBalancedPBE(nn.Module):
    def __init__(self, loss_args, args):
        super().__init__()
        self.base = PBELoss(loss_args, args)
        self.base.visualization = True

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index_dict,
        edge_attr_dict,
        mask_dict,
        model=None,
        x_dict=None,
    ):
        batch, num_graphs = _batch_metadata(mask_dict)
        result = self.base(
            pred_dict,
            target_dict,
            edge_index_dict,
            edge_attr_dict,
            mask_dict,
            model,
            x_dict,
        )
        active = result.pop("Nodal Active Power Loss in p.u.")
        reactive = result.pop("Nodal Reactive Power Loss in p.u.")
        magnitude = torch.sqrt(active.square() + reactive.square())
        graph_loss = magnitude.new_zeros(num_graphs)
        counts = magnitude.new_zeros(num_graphs)
        graph_loss.index_add_(0, batch, magnitude)
        counts.index_add_(0, batch, magnitude.new_ones(magnitude.numel()))
        graph_loss = graph_loss / counts.clamp_min(1)
        loss = graph_loss.mean()
        result["loss"] = loss
        result["Graph-balanced PBE"] = loss.detach()
        return result
