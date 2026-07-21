# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Freeze outcome-blind, explicit per-network scenario split IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.splits import SPLIT_SCHEMA, validate_split_spec


SPLIT_POLICY = "pcg64_80_10_10_source_heldout_test_only_v1"
SPLIT_SEED_BASE = 20270714


def freeze_splits(topology_path: Path, output_path: Path) -> dict:
    """Create explicit IDs using only topology names, splits, and counts."""
    topology = load_topology_manifest(topology_path)
    splits = {}
    seeds = {}
    for ordinal, network in enumerate(sorted(topology["topologies"])):
        record = topology["topologies"][network]
        count = int(record["scenario_count"])
        if count < 1:
            raise ContractError(f"{network} has no scenarios to split")
        split = str(record["split"])
        if split == "source":
            if count < 10:
                raise ContractError(
                    f"source {network} needs at least ten scenarios for 80/10/10",
                )
            seed = SPLIT_SEED_BASE + ordinal
            ids = np.random.Generator(np.random.PCG64(seed)).permutation(count).tolist()
            train_stop = (8 * count) // 10
            val_stop = train_stop + count // 10
            splits[network] = {
                "train": ids[:train_stop],
                "val": ids[train_stop:val_stop],
                "test": ids[val_stop:],
            }
            seeds[network] = seed
        elif split in {"source_dev", "target"}:
            splits[network] = {
                "train": [],
                "val": [],
                "test": list(range(count)),
            }
        else:  # load_topology_manifest should already reject this.
            raise ContractError(f"unsupported topology split {split}")
    payload = {
        "schema_version": SPLIT_SCHEMA,
        "freeze": {
            "policy": SPLIT_POLICY,
            "seed_base": SPLIT_SEED_BASE,
            "source_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
            "source_dev_policy": "test_only",
            "target_policy": "test_only",
            "source_seeds": seeds,
            "outcomes_read": False,
        },
        "splits": splits,
    }
    validate_split_spec(payload, topology)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_splits(args.topology_manifest.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
