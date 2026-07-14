#!/usr/bin/env python3
# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Promote ARIS's ignored research contract into the Git snapshot."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


CONTRACT_FILES = (
    "FINAL_PROPOSAL.md",
    "EXPERIMENT_PLAN.md",
    "EXPERIMENT_TRACKER.md",
    "REVIEW_SUMMARY.md",
    "NOVELTY_REPORT.md",
    "REFINEMENT_REPORT.md",
    "REFINE_STATE.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    source_root = repo / "refine-logs"
    destination_root = repo / "research" / "kron-schur"
    destination_root.mkdir(parents=True, exist_ok=True)

    missing = [name for name in CONTRACT_FILES if not (source_root / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"missing ARIS contract files in {source_root}: {names}")

    for name in CONTRACT_FILES:
        source = source_root / name
        destination = destination_root / name
        shutil.copy2(source, destination)
        print(f"promoted {name} sha256={sha256(destination)}")

    print(
        "Review the Git diff, then commit the promoted snapshot on the research branch.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
