# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Frozen per-network scenario split contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from gridfm_graphkit.fm_scaling.contracts import ContractError

SPLIT_SCHEMA = "fm-scaling-splits-v1"


def split_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_spec(payload: dict, topology_payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("schema_version") != SPLIT_SCHEMA:
        raise ContractError("split manifest has the wrong schema")
    splits = payload.get("splits")
    topologies = topology_payload["topologies"]
    if not isinstance(splits, dict) or set(splits) != set(topologies):
        raise ContractError("split manifest networks differ from topology manifest")
    for network, record in splits.items():
        if set(record) != {"train", "val", "test"}:
            raise ContractError(f"{network} split requires train/val/test")
        lists = {key: [int(value) for value in record[key]] for key in record}
        if any(len(values) != len(set(values)) for values in lists.values()):
            raise ContractError(f"{network} contains duplicate scenario IDs")
        sets = {key: set(values) for key, values in lists.items()}
        if any(
            sets[first] & sets[second]
            for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ContractError(f"{network} split sets overlap")
        expected = set(range(int(topologies[network]["scenario_count"])))
        if set().union(*sets.values()) != expected:
            raise ContractError(f"{network} split does not cover every scenario")
        split = topologies[network]["split"]
        if split == "source" and (not sets["train"] or not sets["val"]):
            raise ContractError(f"source {network} requires train and val scenarios")
        if split == "target" and (
            sets["train"] or sets["val"] or sets["test"] != expected
        ):
            raise ContractError(f"target {network} must be test-only")
        splits[network] = lists
    return payload


def validate_materialized_splits(
    manifest_path: Path,
    split_root: Path,
    topology_payload: dict,
) -> dict:
    payload = validate_split_spec(
        yaml.safe_load(manifest_path.read_text()),
        topology_payload,
    )
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(payload["splits"]):
        raise ContractError("split file records are missing")
    for network, records in files.items():
        if set(records) != {"train", "val", "test"}:
            raise ContractError(f"{network} split file records are incomplete")
        for split, record in records.items():
            if set(record) != {"path", "sha256"}:
                raise ContractError(f"{network}/{split} has an invalid file record")
            path = (split_root / record["path"]).resolve()
            if split_root.resolve() not in path.parents:
                raise ContractError("split file escapes the frozen split root")
            if not path.is_file() or split_file_sha256(path) != record["sha256"]:
                raise ContractError(f"{network}/{split} split hash mismatch")
    return payload
