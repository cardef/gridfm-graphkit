# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts for trustworthy GridFM inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

TensorMap = Mapping[str, torch.Tensor]


class RuntimeDecision(str, Enum):
    """Terminal decision made by GridFMRuntime."""

    ACCEPTED = "accepted"
    REPAIRED = "repaired"
    FALLBACK = "fallback"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PredictionBundle:
    """A model prediction plus optional latent state and metadata."""

    outputs: TensorMap
    latent: TensorMap | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: "PredictionBundle | TensorMap") -> "PredictionBundle":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "Predictors must return PredictionBundle or Mapping[str, Tensor], "
                f"got {type(value)!r}.",
            )
        bad = [key for key, tensor in value.items() if not isinstance(tensor, torch.Tensor)]
        if bad:
            raise TypeError(f"Prediction entries must be tensors; invalid keys: {bad}.")
        return cls(outputs=dict(value))


@dataclass(frozen=True)
class PhysicsReport:
    """Result of checking a prediction against physical contracts."""

    residuals: TensorMap
    component_scores: TensorMap
    score: torch.Tensor
    passed: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return bool(self.passed.detach().all().item())


@dataclass(frozen=True)
class UncertaintyReport:
    """Calibrated intervals and an aggregate runtime risk score."""

    intervals: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    score: torch.Tensor
    calibrated: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeResult:
    """Full inference trace returned by the trustworthy runtime."""

    prediction: PredictionBundle
    raw_prediction: PredictionBundle
    decision: RuntimeDecision
    physics: PhysicsReport | None = None
    uncertainty: UncertaintyReport | None = None
    attempts: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def trusted(self) -> bool:
        return self.decision is not RuntimeDecision.REJECTED


@runtime_checkable
class Predictor(Protocol):
    def __call__(self, batch: Any) -> PredictionBundle | TensorMap: ...


@runtime_checkable
class PhysicsChecker(Protocol):
    def __call__(self, prediction: TensorMap, batch: Any) -> PhysicsReport: ...


@runtime_checkable
class UncertaintyProvider(Protocol):
    def __call__(self, prediction: TensorMap, batch: Any) -> UncertaintyReport: ...


@runtime_checkable
class RepairOperator(Protocol):
    def __call__(
        self,
        prediction: TensorMap,
        batch: Any,
        report: PhysicsReport | None,
    ) -> PredictionBundle | TensorMap: ...


@runtime_checkable
class FallbackSolver(Protocol):
    def __call__(
        self,
        batch: Any,
        warm_start: TensorMap,
        diagnostics: Mapping[str, Any],
    ) -> PredictionBundle | TensorMap: ...
