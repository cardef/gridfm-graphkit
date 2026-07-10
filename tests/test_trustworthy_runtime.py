# Copyright contributors to the gridfm-graphkit project
# SPDX-License-Identifier: Apache-2.0

import torch

from gridfm_graphkit.runtime import (
    GridFMRuntime,
    GradientPhysicsRepair,
    PhysicsCertificate,
    RuntimeDecision,
    SplitConformalRegressor,
    upper_bound_constraint,
)


def certificate():
    return PhysicsCertificate([
        upper_bound_constraint("x_le_one", lambda p, b: p["x"], 1.0, 1e-3),
    ])


def test_split_conformal_finite_sample_quantile():
    cal = SplitConformalRegressor(alpha=0.2).fit(torch.zeros(10), torch.arange(10.0))
    assert cal.qhat.item() == 8.0


def test_runtime_accepts_safe_prediction():
    result = GridFMRuntime(lambda b: {"x": torch.tensor([0.5])}, physics_checker=certificate())(None)
    assert result.decision is RuntimeDecision.ACCEPTED


def test_runtime_repairs_unsafe_prediction():
    repair = GradientPhysicsRepair(
        lambda p, b: torch.relu(p["x"] - 1.0).square().mean(),
        keys=["x"], steps=20, learning_rate=0.25, stop_energy=1e-10,
    )
    result = GridFMRuntime(
        lambda b: {"x": torch.tensor([2.0])},
        physics_checker=certificate(),
        repair_operator=repair,
    )(None)
    assert result.decision is RuntimeDecision.REPAIRED
    assert result.prediction.outputs["x"].item() <= 1.001


def test_runtime_fallback_and_fail_closed():
    fallback = GridFMRuntime(
        lambda b: {"x": torch.tensor([3.0])},
        physics_checker=certificate(),
        fallback_solver=lambda b, w, d: {"x": torch.tensor([0.75])},
    )(None)
    rejected = GridFMRuntime(
        lambda b: {"x": torch.tensor([3.0])}, physics_checker=certificate(),
    )(None)
    assert fallback.decision is RuntimeDecision.FALLBACK
    assert rejected.decision is RuntimeDecision.REJECTED
