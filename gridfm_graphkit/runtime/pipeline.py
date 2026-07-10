# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Predict -> quantify -> certify -> repair -> fallback orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from .types import PredictionBundle, RuntimeDecision, RuntimeResult


@dataclass(frozen=True)
class RuntimePolicy:
    max_uncertainty_score: float | None = None
    attempt_repair: bool = True
    attempt_fallback: bool = True
    trust_fallback: bool = True
    fail_closed: bool = True


class GridFMRuntime:
    """A trustworthy inference shell around any GridFM-compatible predictor."""

    def __init__(
        self,
        predictor,
        *,
        physics_checker=None,
        uncertainty_provider=None,
        repair_operator=None,
        fallback_solver=None,
        policy: RuntimePolicy | None = None,
    ) -> None:
        self.predictor = predictor
        self.physics_checker = physics_checker
        self.uncertainty_provider = uncertainty_provider
        self.repair_operator = repair_operator
        self.fallback_solver = fallback_solver
        self.policy = policy or RuntimePolicy()

    def _assess(self, prediction: PredictionBundle, batch: Any):
        reasons = []
        physics = None if self.physics_checker is None else self.physics_checker(prediction.outputs, batch)
        uncertainty = None if self.uncertainty_provider is None else self.uncertainty_provider(prediction.outputs, batch)
        if physics is not None and not physics.all_passed:
            reasons.append("physics")
        threshold = self.policy.max_uncertainty_score
        if uncertainty is not None and threshold is not None:
            score = float(uncertainty.score.detach().max().item())
            if not torch.isfinite(uncertainty.score).all() or score > threshold:
                reasons.append("uncertainty")
        return physics, uncertainty, not reasons, tuple(reasons)

    def __call__(self, batch: Any) -> RuntimeResult:
        timings = {}
        attempts = []
        t0 = perf_counter()
        raw = PredictionBundle.coerce(self.predictor(batch))
        timings["predict_s"] = perf_counter() - t0
        physics, uncertainty, safe, reasons = self._assess(raw, batch)
        if safe:
            return RuntimeResult(raw, raw, RuntimeDecision.ACCEPTED, physics, uncertainty, diagnostics={"timings": timings, "reasons": ()})

        candidate = raw
        if self.policy.attempt_repair and self.repair_operator is not None:
            attempts.append("repair")
            candidate = PredictionBundle.coerce(self.repair_operator(candidate.outputs, batch, physics))
            physics, uncertainty, safe, reasons = self._assess(candidate, batch)
            if safe:
                return RuntimeResult(candidate, raw, RuntimeDecision.REPAIRED, physics, uncertainty, tuple(attempts), {"timings": timings, "reasons": ()})

        if self.policy.attempt_fallback and self.fallback_solver is not None:
            attempts.append("fallback")
            fallback = PredictionBundle.coerce(self.fallback_solver(batch, candidate.outputs, {"reasons": reasons, "physics": physics, "uncertainty": uncertainty}))
            if self.policy.trust_fallback:
                return RuntimeResult(fallback, raw, RuntimeDecision.FALLBACK, physics, uncertainty, tuple(attempts), {"timings": timings, "reasons": reasons})
            physics, uncertainty, safe, reasons = self._assess(fallback, batch)
            if safe or not self.policy.fail_closed:
                return RuntimeResult(fallback, raw, RuntimeDecision.FALLBACK, physics, uncertainty, tuple(attempts), {"timings": timings, "reasons": reasons})
            candidate = fallback

        decision = RuntimeDecision.REJECTED if self.policy.fail_closed else RuntimeDecision.ACCEPTED
        return RuntimeResult(candidate, raw, decision, physics, uncertainty, tuple(attempts), {"timings": timings, "reasons": reasons})
