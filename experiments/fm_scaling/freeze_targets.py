# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Freeze source/source-development/target membership without reading outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.data import load_topology_manifest


TARGET_FREEZE_SCHEMA = "fm-scaling-target-freeze-v1"


def freeze_targets(
    manifest_path: Path,
    selection_path: Path,
    output_path: Path,
) -> dict:
    manifest = load_topology_manifest(manifest_path)
    selection = yaml.safe_load(selection_path.read_text())
    if (
        not isinstance(selection, dict)
        or selection.get("schema_version") != TARGET_FREEZE_SCHEMA
    ):
        raise ContractError("target freeze has the wrong schema")
    selection_freeze = manifest.get("selection_freeze", {})
    if selection.get("inventory_sha256") != selection_freeze.get("inventory_sha256"):
        raise ContractError("target membership must match the pre-generation inventory")
    source = [
        network
        for network, record in manifest["topologies"].items()
        if record["split"] == "source"
    ]
    source_dev = [
        network
        for network, record in manifest["topologies"].items()
        if record["split"] == "source_dev"
    ]
    targets = [
        network
        for network, record in manifest["topologies"].items()
        if record["split"] == "target"
    ]
    if not targets:
        raise ContractError("pre-generation inventory has no target networks")
    if len(source) < 26:
        raise ContractError("target freeze requires at least 26 source networks")
    target_groups = {
        manifest["topologies"][network]["provenance_group"] for network in targets
    }
    non_target_groups = {
        manifest["topologies"][network]["provenance_group"]
        for network in source + source_dev
    }
    if target_groups & non_target_groups:
        raise ContractError("target provenance groups overlap source groups")
    if len(target_groups) < 6:
        raise ContractError("target freeze requires at least six provenance groups")
    ordered_targets = sorted(
        targets,
        key=lambda network: (
            int(manifest["topologies"][network]["bus_count"]),
            network,
        ),
    )
    quotient, remainder = divmod(len(ordered_targets), 3)
    sizes = [quotient + (index < remainder) for index in range(3)]
    if min(sizes) < 4:
        raise ContractError("derived target size terciles require at least four cases")
    target_metadata = {}
    cursor = 0
    source_max_buses = max(
        int(manifest["topologies"][network]["bus_count"]) for network in source
    )
    for tercile, size in zip(("smallest", "middle", "largest"), sizes):
        for network in ordered_targets[cursor : cursor + size]:
            target_metadata[network] = {
                "size_tercile": tercile,
                "extrapolation": (
                    int(manifest["topologies"][network]["bus_count"]) > source_max_buses
                ),
            }
        cursor += size
    extrapolative = [
        network
        for network, record in target_metadata.items()
        if record["extrapolation"]
    ]
    extrapolation_groups = {
        manifest["topologies"][network]["provenance_group"] for network in extrapolative
    }
    if len(extrapolative) < 4 or len(extrapolation_groups) < 2:
        raise ContractError(
            "derived extrapolation subset requires >=4 cases from >=2 groups",
        )
    for network, record in manifest["topologies"].items():
        record.pop("size_tercile", None)
        record.pop("extrapolation", None)
        if network in targets:
            record["split"] = "target"
            record.update(target_metadata[network])
        elif network in source:
            record["split"] = "source"
        else:
            record["split"] = "source_dev"
    manifest["target_freeze"] = {
        "rule": "stable_bus_count_terciles_source_max_extrapolation",
        "source_max_bus_count": source_max_buses,
        "target_count": len(targets),
        "extrapolative_count": len(extrapolative),
        "extrapolative_group_count": len(extrapolation_groups),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    freeze_targets(
        args.manifest.resolve(),
        args.selection.resolve(),
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
