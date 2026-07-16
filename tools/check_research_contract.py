#!/usr/bin/env python3
# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Enforce one ARIS-native source of truth for the Kron-Schur contract."""

from __future__ import annotations

import argparse
from pathlib import Path


CONTRACT_FILES = (
    "FINAL_PROPOSAL.md",
    "EXPERIMENT_PLAN.md",
    "EXPERIMENT_TRACKER.md",
    "REVIEW_SUMMARY.md",
    "NOVELTY_REPORT.md",
    "REFINEMENT_REPORT.md",
)


def validate(repo: Path) -> list[str]:
    canonical_root = repo / "refine-logs"
    duplicate_root = repo / "research" / "kron-schur"
    errors: list[str] = []

    for name in CONTRACT_FILES:
        canonical = canonical_root / name
        if not canonical.is_file():
            errors.append(f"missing canonical contract file: {canonical}")

        duplicate = duplicate_root / name
        if duplicate.exists() or duplicate.is_symlink():
            errors.append(
                f"duplicate contract file is forbidden: {duplicate}; use {canonical}",
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to this script's parent repository)",
    )
    args = parser.parse_args()

    errors = validate(args.repo.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Kron-Schur contract has one ARIS-native source of truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
