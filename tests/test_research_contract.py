# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tools.check_research_contract import CONTRACT_FILES, validate


def make_canonical_contract(repo: Path) -> None:
    root = repo / "refine-logs"
    root.mkdir(parents=True)
    for name in CONTRACT_FILES:
        (root / name).write_text("canonical\n", encoding="utf-8")


def test_validate_accepts_only_canonical_contract(tmp_path: Path) -> None:
    make_canonical_contract(tmp_path)

    assert validate(tmp_path) == []


def test_validate_rejects_research_directory_duplicate(tmp_path: Path) -> None:
    make_canonical_contract(tmp_path)
    duplicate = tmp_path / "research" / "kron-schur" / "FINAL_PROPOSAL.md"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text("duplicate\n", encoding="utf-8")

    errors = validate(tmp_path)

    assert len(errors) == 1
    assert str(duplicate) in errors[0]
