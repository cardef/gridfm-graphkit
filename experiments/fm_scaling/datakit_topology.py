# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic source-only topology normalization for data generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from gridfm_graphkit.fm_scaling.contracts import ContractError


# MATPOWER v2 column indices. Keeping the normalization independent of
# Datakit's imports lets its behavior be unit-tested without initializing Julia.
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
NONE = 4
GEN_BUS = 0
GEN_STATUS = 7
F_BUS = 0
T_BUS = 1
BR_STATUS = 10

TOPOLOGY_PREPROCESSING_POLICY = "drop_declared_inert_type4_buses_v1"


def prune_declared_inert_buses(mpc: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    """Drop only MATPOWER type-4 buses that are electrically inert.

    PowerModels excludes declared isolated buses from its solution dictionary,
    while Datakit expects every encoded bus to have a solution. Removing only
    type-4 buses with zero injection/shunt and no in-service incident element
    reconciles those semantics without reading scenarios or solver outcomes.
    Any non-inert type-4 bus fails closed.
    """
    buses = np.asarray(mpc["bus"])
    gens = np.asarray(mpc["gen"])
    branches = np.asarray(mpc["branch"])
    isolated = buses[:, BUS_TYPE].astype(np.int64) == NONE
    if not isolated.any():
        return mpc, []

    dropped_ids = buses[isolated, BUS_I].astype(np.int64)
    dropped = set(int(value) for value in dropped_ids)
    inert_columns = buses[isolated][:, [PD, QD, GS, BS]]
    if not np.allclose(inert_columns, 0.0, atol=0.0, rtol=0.0):
        raise ContractError(
            "declared type-4 buses have nonzero load or shunt; refusing to drop them",
        )

    gen_on_dropped = np.isin(gens[:, GEN_BUS].astype(np.int64), dropped_ids)
    if np.any(gen_on_dropped & (gens[:, GEN_STATUS] != 0)):
        raise ContractError(
            "declared type-4 buses have in-service generators; refusing to drop them",
        )

    branch_on_dropped = np.isin(
        branches[:, F_BUS].astype(np.int64),
        dropped_ids,
    ) | np.isin(branches[:, T_BUS].astype(np.int64), dropped_ids)
    if np.any(branch_on_dropped & (branches[:, BR_STATUS] != 0)):
        raise ContractError(
            "declared type-4 buses have in-service branches; refusing to drop them",
        )

    keep_bus = ~isolated
    keep_gen = ~gen_on_dropped
    keep_branch = ~branch_on_dropped
    normalized = dict(mpc)
    normalized["bus"] = buses[keep_bus].copy()
    normalized["gen"] = gens[keep_gen].copy()
    normalized["branch"] = branches[keep_branch].copy()

    gencost = np.asarray(mpc["gencost"])
    if gencost.shape[0] == gens.shape[0]:
        normalized["gencost"] = gencost[keep_gen].copy()
    elif gencost.shape[0] == 2 * gens.shape[0]:
        normalized["gencost"] = np.concatenate(
            (gencost[: gens.shape[0]][keep_gen], gencost[gens.shape[0] :][keep_gen]),
        )
    elif gen_on_dropped.any():
        raise ContractError(
            "cannot align generator costs while dropping type-4-bus generators",
        )

    return normalized, sorted(dropped)


def normalize_datakit_network(network):
    """Apply the frozen topology policy to a Datakit ``Network`` instance."""
    normalized_mpc, dropped = prune_declared_inert_buses(network.mpc)
    if not dropped:
        return network, dropped
    from gridfm_datakit.network import Network

    normalized = Network(normalized_mpc)
    if not normalized.check_single_connected_component():
        raise ContractError("energized topology remains disconnected after normalization")
    return normalized, dropped
