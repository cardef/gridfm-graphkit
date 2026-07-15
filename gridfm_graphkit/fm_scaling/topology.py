# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Read static topology/admittance columns without exposing PF outcomes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from gridfm_graphkit.fm_scaling.contracts import (
    ComplexCOO,
    ContractError,
    GridTopology,
)


RAW_TOPOLOGY_FILES = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
    "y_bus_data.parquet",
    "args.log",
)


def raw_data_sha256(raw: Path) -> str:
    digest = hashlib.sha256()
    for name in RAW_TOPOLOGY_FILES:
        path = raw / name
        if not path.is_file():
            raise ContractError(f"missing raw data artifact {path}")
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(name.encode())
        digest.update(file_digest.hexdigest().encode())
    return digest.hexdigest()


def _assert_static(
    frame: pd.DataFrame,
    coordinate_columns: list[str],
    value_columns: list[str],
) -> None:
    grouped = frame.groupby(coordinate_columns, sort=True)[value_columns].nunique()
    if (grouped > 1).any().any():
        raise ContractError(
            f"topology columns vary by scenario for coordinates {coordinate_columns}",
        )
    expected = frame.groupby(coordinate_columns, sort=True).ngroups
    counts = frame.groupby("scenario", sort=True).size()
    if not (counts == expected).all():
        raise ContractError("topology coordinate coverage varies by scenario")


def load_grid_topology(
    data_root: Path,
    network: str,
    record: dict,
) -> GridTopology:
    raw = data_root / network / "raw"
    y_path = raw / "y_bus_data.parquet"
    branch_path = raw / "branch_data.parquet"
    if not y_path.exists() or not branch_path.exists():
        raise ContractError(f"missing static topology parquet for {network}")

    y_bus = pd.read_parquet(
        y_path,
        columns=["scenario", "index1", "index2", "G", "B"],
    )
    branches = pd.read_parquet(
        branch_path,
        columns=["scenario", "from_bus", "to_bus", "br_status"],
    )
    _assert_static(y_bus, ["index1", "index2"], ["G", "B"])
    _assert_static(branches, ["from_bus", "to_bus"], ["br_status"])

    first_y = (
        y_bus[y_bus["scenario"] == y_bus["scenario"].min()]
        .sort_values(["index1", "index2"])
        .drop_duplicates(["index1", "index2"])
    )
    first_branches = (
        branches[branches["scenario"] == branches["scenario"].min()]
        .sort_values(["from_bus", "to_bus"])
        .drop_duplicates(["from_bus", "to_bus"])
    )
    first_branches = first_branches[first_branches["br_status"] == 1]
    num_bus = int(record["bus_count"])
    coordinates = set(first_y["index1"]).union(first_y["index2"])
    if coordinates != set(range(num_bus)):
        raise ContractError(f"Y-bus indices for {network} do not match bus_count")

    return GridTopology(
        topology_key=str(record["topology_key"]),
        bus_ids=tuple(range(num_bus)),
        fine_edges=tuple(
            (int(row.from_bus), int(row.to_bus))
            for row in first_branches.itertuples(index=False)
        ),
        y_bus=ComplexCOO(
            shape=(num_bus, num_bus),
            row=tuple(int(value) for value in first_y["index1"]),
            col=tuple(int(value) for value in first_y["index2"]),
            value=tuple(
                complex(real, imag) for real, imag in zip(first_y["G"], first_y["B"])
            ),
        ),
        base_mva=float(record["baseMVA"]),
        provenance_group=str(record["provenance_group"]),
    )
