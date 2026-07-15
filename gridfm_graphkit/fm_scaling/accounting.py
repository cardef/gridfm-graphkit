# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic parameter, FLOP, checkpoint, and compile-parity gates."""

from __future__ import annotations

import json
import itertools
import os
from pathlib import Path

import torch
import torch.distributed as dist
from lightning.pytorch import Callback
from torch.utils.flop_counter import FlopCounterMode

from gridfm_graphkit.fm_scaling.contracts import ContractError


def trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def parameter_match_report(models: dict[str, torch.nn.Module]) -> dict:
    counts = {name: trainable_parameter_count(model) for name, model in models.items()}
    minimum = min(counts.values())
    maximum = max(counts.values())
    relative_gap = (maximum - minimum) / max(minimum, 1)
    return {
        "counts": counts,
        "relative_gap": relative_gap,
        "passed": relative_gap <= 0.02,
    }


def deterministic_capacity_match(
    candidate_counts: dict[str, list[tuple[dict, int]]],
    tolerance: float = 0.02,
) -> dict:
    """Choose the lexicographically deterministic feasible capacity tuple."""
    if not candidate_counts or not 0 <= tolerance < 1:
        raise ContractError("invalid capacity-search inputs")
    arms = sorted(candidate_counts)
    if any(not candidate_counts[arm] for arm in arms):
        raise ContractError("every arm needs at least one capacity candidate")
    feasible = []
    for combination in itertools.product(*(candidate_counts[arm] for arm in arms)):
        counts = [item[1] for item in combination]
        gap = (max(counts) - min(counts)) / max(min(counts), 1)
        if gap <= tolerance:
            specs = [item[0] for item in combination]
            tie_key = json.dumps(specs, sort_keys=True, separators=(",", ":"))
            feasible.append((gap, max(counts), tie_key, combination))
    if not feasible:
        raise ContractError("no parameter-matched capacity tuple within tolerance")
    gap, _, _, selected = min(feasible)
    return {
        "relative_gap": gap,
        "passed": True,
        "selection": {
            arm: {"spec": item[0], "parameters": item[1]}
            for arm, item in zip(arms, selected)
        },
    }


def counted_forward_flops(module: torch.nn.Module, *args, **kwargs) -> int:
    """Count executed forward operations with PyTorch's operator counter."""
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        module(*args, **kwargs)
    return int(counter.get_total_flops())


def output_and_gradient_parity(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    batch,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict:
    """Compare output and parameter gradients for compile/upstream parity gates."""
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    reference_output = reference(batch)
    candidate_output = candidate(batch)
    if reference_output.keys() != candidate_output.keys():
        return {"passed": False, "reason": "output keys differ"}
    output_passed = all(
        torch.allclose(
            reference_output[key],
            candidate_output[key],
            atol=atol,
            rtol=rtol,
        )
        for key in reference_output
    )
    sum(value.float().sum() for value in reference_output.values()).backward()
    sum(value.float().sum() for value in candidate_output.values()).backward()
    reference_gradients = [
        parameter.grad
        for parameter in reference.parameters()
        if parameter.requires_grad
    ]
    candidate_gradients = [
        parameter.grad
        for parameter in candidate.parameters()
        if parameter.requires_grad
    ]
    gradient_passed = len(reference_gradients) == len(candidate_gradients) and all(
        first is not None
        and second is not None
        and torch.allclose(first, second, atol=atol, rtol=rtol)
        for first, second in zip(reference_gradients, candidate_gradients)
    )
    return {
        "passed": output_passed and gradient_passed,
        "output_passed": output_passed,
        "gradient_passed": gradient_passed,
    }


class CumulativeFlopCheckpoint(Callback):
    """Save the first checkpoint crossing each frozen cumulative-FLOP target."""

    def __init__(self, thresholds: list[int], output_dir: str):
        super().__init__()
        if not thresholds or any(value <= 0 for value in thresholds):
            raise ContractError("FLOP checkpoint thresholds must be positive")
        if sorted(set(thresholds)) != thresholds:
            raise ContractError("FLOP checkpoint thresholds must be unique and sorted")
        self.thresholds = thresholds
        self.output_dir = Path(output_dir)
        self.cumulative_flops = 0
        self.crossed: dict[int, dict] = {}
        self._counter: FlopCounterMode | None = None

    def on_fit_start(self, trainer, pl_module):
        del trainer, pl_module
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ContractError(
                f"refusing to overwrite FLOP artifacts in {self.output_dir}",
            )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del trainer, pl_module, batch, batch_idx
        self._counter = FlopCounterMode(display=False)
        self._counter.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch
        if self._counter is None:
            raise ContractError("FLOP counter was not started")
        self._counter.__exit__(None, None, None)
        batch_flops = int(self._counter.get_total_flops())
        self._counter = None
        if batch_flops <= 0:
            raise ContractError("training batch produced no counted FLOPs")
        if dist.is_available() and dist.is_initialized():
            value = torch.tensor(batch_flops, device=trainer.strategy.root_device)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            batch_flops = int(value.item())
        previous = self.cumulative_flops
        self.cumulative_flops += batch_flops
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for threshold in self.thresholds:
            if threshold in self.crossed or self.cumulative_flops < threshold:
                continue
            checkpoint = self.output_dir / f"flops_{threshold}.pt"
            if trainer.is_global_zero:
                try:
                    with checkpoint.open("xb") as handle:
                        torch.save(pl_module.state_dict(), handle)
                except FileExistsError as error:
                    raise ContractError(
                        f"refusing to overwrite {checkpoint}",
                    ) from error
            self.crossed[threshold] = {
                "threshold": threshold,
                "previous_flops": previous,
                "crossing_flops": self.cumulative_flops,
                "batch_index": int(batch_idx),
                "global_step": int(trainer.global_step),
                "checkpoint": str(checkpoint),
            }
            self._write_ledger()
        if self.cumulative_flops >= self.thresholds[-1]:
            trainer.should_stop = True

    def _write_ledger(self) -> None:
        ledger = {
            "thresholds": self.thresholds,
            "cumulative_flops": self.cumulative_flops,
            "crossed": [self.crossed[key] for key in sorted(self.crossed)],
        }
        path = self.output_dir / "flop_checkpoints.json"
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            temporary = path.with_suffix(".json.tmp")
            with temporary.open("x") as handle:
                json.dump(ledger, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)

    def state_dict(self):
        return {
            "cumulative_flops": self.cumulative_flops,
            "crossed": self.crossed,
        }

    def load_state_dict(self, state_dict):
        self.cumulative_flops = int(state_dict["cumulative_flops"])
        self.crossed = {int(key): value for key, value in state_dict["crossed"].items()}
