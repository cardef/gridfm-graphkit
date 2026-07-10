# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Composable physical contracts for model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from .types import PhysicsReport, TensorMap

ResidualFunction = Callable[[TensorMap, Any], torch.Tensor]


@dataclass(frozen=True)
class ResidualConstraint:
    """A physical contract expressed as a residual tensor."""

    name: str
    residual_fn: ResidualFunction
    tolerance: float
    reduction: str = "max"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Constraint names must be non-empty.")
        if self.tolerance < 0:
            raise ValueError("Constraint tolerance must be non-negative.")
        if self.reduction not in {"max", "mean", "rms"}:
            raise ValueError("reduction must be one of: max, mean, rms.")
        if self.weight <= 0:
            raise ValueError("Constraint weight must be positive.")

    def evaluate(self, prediction: TensorMap, batch: Any) -> torch.Tensor:
        residual = self.residual_fn(prediction, batch)
        if not isinstance(residual, torch.Tensor):
            residual = torch.as_tensor(residual)
        if residual.numel() == 0:
            raise ValueError(f"Constraint {self.name!r} returned an empty residual.")
        if not torch.isfinite(residual).all():
            return torch.full((), float("inf"), device=residual.device)
        magnitude = residual.abs()
        if self.reduction == "max":
            return magnitude.max()
        if self.reduction == "mean":
            return magnitude.mean()
        return magnitude.square().mean().sqrt()


class PhysicsCertificate:
    """Aggregate physical residuals into a fail-closed certificate."""

    def __init__(self, constraints: Iterable[ResidualConstraint]) -> None:
        self.constraints = tuple(constraints)
        if not self.constraints:
            raise ValueError("PhysicsCertificate requires at least one constraint.")
        names = [constraint.name for constraint in self.constraints]
        if len(set(names)) != len(names):
            raise ValueError("Constraint names must be unique.")

    def __call__(self, prediction: TensorMap, batch: Any) -> PhysicsReport:
        residuals: dict[str, torch.Tensor] = {}
        component_scores: dict[str, torch.Tensor] = {}
        passed: list[torch.Tensor] = []
        for constraint in self.constraints:
            residual = constraint.evaluate(prediction, batch)
            residuals[constraint.name] = residual
            tolerance = torch.as_tensor(
                constraint.tolerance,
                dtype=residual.dtype,
                device=residual.device,
            )
            if constraint.tolerance == 0:
                normalized = torch.where(
                    residual == 0,
                    torch.zeros_like(residual),
                    torch.full_like(residual, float("inf")),
                )
            else:
                normalized = residual / tolerance
            component_scores[constraint.name] = constraint.weight * normalized
            passed.append(residual <= tolerance)
        score = torch.stack(tuple(component_scores.values())).max()
        return PhysicsReport(
            residuals=residuals,
            component_scores=component_scores,
            score=score,
            passed=torch.stack(passed).all(),
            metadata={"num_constraints": len(self.constraints)},
        )


def equality_constraint(
    name: str,
    residual_fn: ResidualFunction,
    tolerance: float,
    *,
    reduction: str = "max",
    weight: float = 1.0,
) -> ResidualConstraint:
    return ResidualConstraint(name, residual_fn, tolerance, reduction, weight)


def upper_bound_constraint(
    name: str,
    value_fn: ResidualFunction,
    upper: float | torch.Tensor,
    tolerance: float = 0.0,
    *,
    reduction: str = "max",
    weight: float = 1.0,
) -> ResidualConstraint:
    def violation(prediction: TensorMap, batch: Any) -> torch.Tensor:
        value = value_fn(prediction, batch)
        limit = torch.as_tensor(upper, dtype=value.dtype, device=value.device)
        return torch.relu(value - limit)

    return ResidualConstraint(name, violation, tolerance, reduction, weight)


def lower_bound_constraint(
    name: str,
    value_fn: ResidualFunction,
    lower: float | torch.Tensor,
    tolerance: float = 0.0,
    *,
    reduction: str = "max",
    weight: float = 1.0,
) -> ResidualConstraint:
    def violation(prediction: TensorMap, batch: Any) -> torch.Tensor:
        value = value_fn(prediction, batch)
        limit = torch.as_tensor(lower, dtype=value.dtype, device=value.device)
        return torch.relu(limit - value)

    return ResidualConstraint(name, violation, tolerance, reduction, weight)
