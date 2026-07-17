# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Run datakit generation and emit provenance from that exact process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Datakit's feasibility-bound worker uses multiprocessing.Pool. Force spawn
# before Julia starts: forking the initialized multithreaded Julia runtime
# deadlocks both parent and child on Abacus.
multiprocessing.set_start_method("spawn", force=True)

# JuliaCall must initialize before GraphKit imports Torch; the reverse order is a
# documented segfault risk in mixed Julia/PyTorch processes.
import juliacall  # noqa: F401

from experiments.fm_scaling.prepare_data import assert_datakit_checkout
from gridfm_graphkit.fm_scaling.manifest import file_sha256


def _environment_hash() -> str:
    rows = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datakit_root = assert_datakit_checkout(
        args.repo_root.resolve(),
        args.expected_commit,
    )
    import gridfm_datakit
    from gridfm_datakit.generate import generate_power_flow_data_distributed

    generate_power_flow_data_distributed(str(args.config.resolve()))
    payload = {
        "schema_version": "fm-scaling-datakit-provenance-v1",
        "status": "GENERATED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config.resolve()),
        "datakit_commit": args.expected_commit,
        "datakit_root": str(datakit_root),
        "module_path": str(Path(gridfm_datakit.__file__).resolve()),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment_sha256": _environment_hash(),
        "inventory_sha256": args.inventory_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
