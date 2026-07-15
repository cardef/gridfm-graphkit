# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Materialize explicit per-network scenario IDs as immutable tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.splits import split_file_sha256, validate_split_spec


def materialize(
    spec_path: Path,
    topology_path: Path,
    output_root: Path,
    manifest_path: Path,
) -> dict:
    topology = load_topology_manifest(topology_path)
    payload = validate_split_spec(yaml.safe_load(spec_path.read_text()), topology)
    files = {}
    for network, splits in payload["splits"].items():
        network_dir = output_root / network
        network_dir.mkdir(parents=True, exist_ok=True)
        files[network] = {}
        for split, values in splits.items():
            path = network_dir / f"{split}.pt"
            torch.save(torch.tensor(values, dtype=torch.long), path)
            files[network][split] = {
                "path": str(path.relative_to(output_root)),
                "sha256": split_file_sha256(path),
            }
    payload["files"] = files
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return payload


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    materialize(
        args.spec.resolve(),
        args.topology_manifest.resolve(),
        args.output_root.resolve(),
        args.manifest.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
