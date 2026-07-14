# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Trustworthy, solver-backed runtime for GridFM models."""

from .conformal import MultiOutputConformal, SplitConformalRegressor
from .physics import (
    PhysicsCertificate,
    ResidualConstraint,
    equality_constraint,
    lower_bound_constraint,
    upper_bound_constraint,
)
from .pipeline import GridFMRuntime, RuntimePolicy
from .repair import GradientPhysicsRepair
from .types import (
    PhysicsReport,
    PredictionBundle,
    RuntimeDecision,
    RuntimeResult,
    UncertaintyReport,
)

__all__ = [
    "GridFMRuntime",
    "GradientPhysicsRepair",
    "MultiOutputConformal",
    "PhysicsCertificate",
    "PhysicsReport",
    "PredictionBundle",
    "ResidualConstraint",
    "RuntimeDecision",
    "RuntimePolicy",
    "RuntimeResult",
    "SplitConformalRegressor",
    "UncertaintyReport",
    "equality_constraint",
    "lower_bound_constraint",
    "upper_bound_constraint",
]
