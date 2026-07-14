# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free split-conformal calibration primitives."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch

from .types import TensorMap, UncertaintyReport

ScaleProvider = Callable[[str, TensorMap, Any], torch.Tensor | None]


def _finite_scores(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor | None,
    mask: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}.",
        )
    score = (target - prediction).abs()
    if scale is not None:
        score = score / scale.abs().clamp_min(eps)
    if mask is not None:
        if mask.shape != score.shape:
            mask = torch.broadcast_to(mask, score.shape)
        score = score[mask.bool()]
    else:
        score = score.reshape(-1)
    return score[torch.isfinite(score)]


class SplitConformalRegressor:
    """Symmetric split-conformal intervals with finite-sample correction."""

    def __init__(self, alpha: float = 0.1, eps: float = 1e-8) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.qhat: torch.Tensor | None = None
        self.num_calibration = 0

    @property
    def fitted(self) -> bool:
        return self.qhat is not None

    def fit(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> "SplitConformalRegressor":
        scores = _finite_scores(prediction, target, scale, mask, self.eps)
        if scores.numel() == 0:
            raise ValueError("No finite calibration scores were provided.")
        scores = scores.detach().sort().values
        n = scores.numel()
        rank = min(math.ceil((n + 1) * (1 - self.alpha)), n)
        self.qhat = scores[rank - 1]
        self.num_calibration = n
        return self

    def interval(
        self,
        prediction: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qhat is None:
            raise RuntimeError("Calibrator is not fitted.")
        radius = self.qhat.to(device=prediction.device, dtype=prediction.dtype)
        if scale is not None:
            radius = radius * scale.abs().clamp_min(self.eps)
        return prediction - radius, prediction + radius


class MultiOutputConformal:
    """Calibrate and serve intervals for a dictionary of model outputs."""

    def __init__(
        self,
        calibrators: Mapping[str, SplitConformalRegressor],
        scale_provider: ScaleProvider | None = None,
    ) -> None:
        if not calibrators:
            raise ValueError("At least one output calibrator is required.")
        self.calibrators = dict(calibrators)
        self.scale_provider = scale_provider

    def fit(
        self,
        predictions: TensorMap,
        targets: TensorMap,
        *,
        scales: Mapping[str, torch.Tensor] | None = None,
        masks: Mapping[str, torch.Tensor] | None = None,
    ) -> "MultiOutputConformal":
        for key, calibrator in self.calibrators.items():
            if key not in predictions or key not in targets:
                raise KeyError(f"Missing conformal output {key!r}.")
            calibrator.fit(
                predictions[key],
                targets[key],
                scale=None if scales is None else scales.get(key),
                mask=None if masks is None else masks.get(key),
            )
        return self

    def __call__(self, prediction: TensorMap, batch: Any) -> UncertaintyReport:
        intervals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        scores: list[torch.Tensor] = []
        calibration_sizes: dict[str, int] = {}
        for key, calibrator in self.calibrators.items():
            if key not in prediction:
                raise KeyError(f"Prediction is missing calibrated output {key!r}.")
            scale = (
                None
                if self.scale_provider is None
                else self.scale_provider(key, prediction, batch)
            )
            intervals[key] = calibrator.interval(prediction[key], scale=scale)
            assert calibrator.qhat is not None
            scores.append(calibrator.qhat.to(prediction[key].device))
            calibration_sizes[key] = calibrator.num_calibration
        return UncertaintyReport(
            intervals=intervals,
            score=torch.stack(scores).max(),
            calibrated=True,
            metadata={
                "alpha": {key: cal.alpha for key, cal in self.calibrators.items()},
                "num_calibration": calibration_sizes,
            },
        )
