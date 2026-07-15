# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth PF task with immutable per-scenario evaluation records."""

from __future__ import annotations

import json
import platform
import resource
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only

from gridfm_graphkit.datasets.globals import VA_H, VA_OUT, VM_H, VM_OUT
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.accounting import trainable_parameter_count
from gridfm_graphkit.fm_scaling.manifest import file_sha256
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)
from gridfm_graphkit.tasks.pf_task import (
    PowerFlowTask,
    _build_bus_target,
    _clamp_known_to_ground_truth,
)
from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask


def _graph_mean(
    values: torch.Tensor,
    batch_index: torch.Tensor,
    count: int,
) -> torch.Tensor:
    output = values.new_zeros(count)
    output.index_add_(0, batch_index, values)
    sizes = values.new_zeros(count)
    sizes.index_add_(0, batch_index, values.new_ones(values.numel()))
    return output / sizes.clamp_min(1)


def _graph_masked_rmse(
    squared_error: torch.Tensor,
    mask: torch.Tensor,
    batch_index: torch.Tensor,
    count: int,
) -> torch.Tensor:
    totals = squared_error.new_zeros(count)
    sizes = squared_error.new_zeros(count)
    totals.index_add_(0, batch_index, torch.where(mask, squared_error, 0.0))
    sizes.index_add_(0, batch_index, mask.to(squared_error.dtype))
    if bool((sizes == 0).any()):
        raise ContractError("every graph must contain masked VM and VA targets")
    return (totals / sizes).sqrt()


def graph_family_balanced_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    bus_mask: torch.Tensor,
    bus_batch: torch.Tensor,
    num_graphs: int,
    vm_scale: float,
    va_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the frozen masked VM/VA RMSE and Euclidean family scalar."""
    rmse_vm = _graph_masked_rmse(
        (prediction[:, VM_OUT] - target[:, VM_OUT]).square(),
        bus_mask[:, VM_H],
        bus_batch,
        num_graphs,
    )
    angle_error = (
        torch.remainder(
            prediction[:, VA_OUT] - target[:, VA_OUT] + torch.pi,
            2 * torch.pi,
        )
        - torch.pi
    )
    rmse_va = _graph_masked_rmse(
        angle_error.square(),
        bus_mask[:, VA_H],
        bus_batch,
        num_graphs,
    )
    error = torch.sqrt(
        ((rmse_vm / vm_scale).square() + (rmse_va / va_scale).square()) / 2,
    )
    return rmse_vm, rmse_va, error


def _keys(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@TASK_REGISTRY.register("FMScalingPowerFlow")
class FMScalingPowerFlowTask(ReconstructionTask):
    """Common training objective plus parseable, topology-level PF evidence."""

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)
        self.eval_records: list[dict] = []
        self.vm_scale = float(args.evaluation.vm_scale)
        self.va_scale = float(args.evaluation.va_scale)
        if min(self.vm_scale, self.va_scale) <= 0:
            raise ContractError("frozen VM/VA metric scales must be positive")

    @rank_zero_only
    def on_fit_start(self):
        super().on_fit_start()
        self._fit_started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if isinstance(self.logger, MLFlowLogger):
            artifact_dir = Path(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
                "stats",
            )
        else:
            artifact_dir = Path(self.logger.save_dir, "stats")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        topology_manifest = Path(str(self.args.data.topology_manifest)).resolve()
        payload = {
            "schema_version": "fm-scaling-runtime-contract-v1",
            "run_id": str(self.args.evaluation.run_id),
            "communication_core": str(self.args.model.communication_core),
            "seed": int(self.args.seed),
            "trainable_parameters": trainable_parameter_count(self.model),
            "geometry_bundle_sha256": self.model.geometry_bundle_hash,
            "topology_manifest": str(topology_manifest),
            "topology_manifest_sha256": file_sha256(topology_manifest),
            "train_networks": list(self.args.data.train_networks),
            "evaluation_networks": list(self.args.data.networks),
        }
        (artifact_dir / "fm_scaling_runtime_contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    @rank_zero_only
    def on_fit_end(self):
        elapsed = time.monotonic() - self._fit_started
        output = Path(str(self.args.training.runtime_output_path)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise ContractError(f"refusing to overwrite runtime summary {output}")
        gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        payload = {
            "schema_version": "fm-scaling-runtime-v1",
            "run_id": str(self.args.evaluation.run_id),
            "status": "TRAINED",
            "wall_seconds": elapsed,
            "gpu_count": gpu_count,
            "gpu_hours": elapsed * gpu_count / 3600,
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated()) if gpu_count else 0
            ),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "trainable_parameters": trainable_parameter_count(self.model),
            "geometry_bundle_sha256": self.model.geometry_bundle_hash,
        }
        with output.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def shared_step(self, batch):
        output = self.forward(batch)
        masks = dict(batch.mask_dict)
        masks["_bus_batch"] = batch.batch_dict["bus"]
        masks["_num_graphs"] = int(batch.num_graphs)
        loss_dict = self.loss_fn(
            output,
            batch.y_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            masks,
            model=self.model,
            x_dict=batch.x_dict,
        )
        return output, loss_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output, loss_dict = self.shared_step(batch)
        del loss_dict
        self.data_normalizers[dataloader_idx].inverse_transform(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output, batch)

        num_bus = batch.x_dict["bus"].size(0)
        target, gen_to_bus_index, _ = _build_bus_target(batch, num_bus)
        projected = _clamp_known_to_ground_truth(
            output["bus"],
            target,
            batch,
            gen_to_bus_index,
            num_bus,
        )
        edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
        edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]
        p_flow, q_flow = ComputeBranchFlow()(projected, edge_index, edge_attr)
        p_in, q_in = ComputeNodeInjection()(p_flow, q_flow, edge_index, num_bus)
        residual_p, residual_q = ComputeNodeResiduals()(
            p_in,
            q_in,
            projected,
            batch.x_dict["bus"],
        )

        bus_batch = batch.batch_dict["bus"]
        num_graphs = int(batch.num_graphs)
        rmse_vm, rmse_va, error = graph_family_balanced_error(
            output["bus"],
            target,
            batch.mask_dict["bus"],
            bus_batch,
            num_graphs,
            self.vm_scale,
            self.va_scale,
        )
        residual = _graph_mean(
            torch.sqrt(residual_p.square() + residual_q.square()),
            bus_batch,
            num_graphs,
        )
        base_mva = batch.baseMVA.reshape(-1).to(residual)
        if base_mva.numel() == 1:
            base_mva = base_mva.expand(num_graphs)
        residual = residual / base_mva
        topology_keys = _keys(batch.topology_key)
        scenario_ids = batch.scenario_id.reshape(-1).tolist()
        if len(topology_keys) != num_graphs or len(scenario_ids) != num_graphs:
            raise ContractError("batched evaluation metadata does not align")
        for graph in range(num_graphs):
            self.eval_records.append(
                {
                    "run_id": str(self.args.evaluation.run_id),
                    "system": str(self.args.model.communication_core),
                    "g_level": str(self.args.evaluation.g_level),
                    "seed": int(self.args.seed),
                    "checkpoint": str(self.args.evaluation.checkpoint),
                    "topology_key": topology_keys[graph],
                    "scenario_id": int(scenario_ids[graph]),
                    "rmse_vm_pu": float(rmse_vm[graph]),
                    "rmse_va_rad": float(rmse_va[graph]),
                    "family_balanced_error": float(error[graph]),
                    "dimensionless_residual": float(residual[graph]),
                },
            )

    def on_test_epoch_end(self):
        records = self.eval_records
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, records)
            records = [record for rank_records in gathered for record in rank_records]
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            output = Path(str(self.args.evaluation.output_path))
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ContractError(f"refusing to overwrite metrics {output}")
            with output.open("x") as handle:
                json.dump(records, handle, indent=2, sort_keys=True)
                handle.write("\n")
        self.eval_records = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return PowerFlowTask.predict_step(self, batch, batch_idx, dataloader_idx)
