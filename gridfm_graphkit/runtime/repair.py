# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Repair operators for projecting learned outputs toward feasibility."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

from .types import PhysicsReport, PredictionBundle, TensorMap

EnergyFunction = Callable[[TensorMap, Any], torch.Tensor]


class GradientPhysicsRepair:
    """Perform a small trust-region gradient repair in output space."""

    def __init__(
        self,
        energy_fn: EnergyFunction,
        *,
        keys: Iterable[str] | None = None,
        steps: int = 8,
        learning_rate: float = 0.1,
        max_update: float | None = None,
        stop_energy: float = 0.0,
    ) -> None:
        if steps <= 0 or learning_rate <= 0:
            raise ValueError("steps and learning_rate must be positive.")
        if max_update is not None and max_update <= 0:
            raise ValueError("max_update must be positive when provided.")
        self.energy_fn = energy_fn
        self.keys = None if keys is None else frozenset(keys)
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.max_update = max_update
        self.stop_energy = float(stop_energy)

    def __call__(self, prediction: TensorMap, batch: Any, report: PhysicsReport | None) -> PredictionBundle:
        del report
        selected = [
            key for key, value in prediction.items()
            if value.is_floating_point() and (self.keys is None or key in self.keys)
        ]
        if not selected:
            raise ValueError("No floating outputs to optimize.")
        anchors = {key: prediction[key].detach() for key in selected}
        current = dict(prediction)
        current.update({key: anchors[key].clone().requires_grad_(True) for key in selected})
        trace: list[float] = []
        with torch.enable_grad():
            for _ in range(self.steps):
                energy = self.energy_fn(current, batch)
                if not isinstance(energy, torch.Tensor):
                    energy = torch.as_tensor(energy)
                energy = energy.mean()
                if not torch.isfinite(energy):
                    break
                value = float(energy.detach().item())
                trace.append(value)
                if value <= self.stop_energy:
                    break
                variables = [current[key] for key in selected]
                gradients = torch.autograd.grad(energy, variables, allow_unused=True)
                updated = {}
                for key, variable, gradient in zip(selected, variables, gradients):
                    candidate = variable if gradient is None else variable - self.learning_rate * gradient
                    if self.max_update is not None:
                        delta = (candidate - anchors[key]).clamp(-self.max_update, self.max_update)
                        candidate = anchors[key] + delta
                    updated[key] = candidate.detach().requires_grad_(True)
                current.update(updated)
        return PredictionBundle(
            outputs={key: value.detach() for key, value in current.items()},
            metadata={"repair": "gradient_physics", "energy_trace": tuple(trace)},
        )
