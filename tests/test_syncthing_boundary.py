# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from tools import check_syncthing_boundary as boundary


FOLDERS = {"gridfm-papers": PurePosixPath("papers")}


def git_result(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout, "")


def test_git_boundary_accepts_disjoint_git_and_syncthing_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
        if args[0] == "ls-tree":
            return git_result(stdout="refine-logs/FINAL_PROPOSAL.md\n")
        return git_result()

    monkeypatch.setattr(boundary, "_run_git", fake_run_git)

    boundary.validate_git_boundary(
        tmp_path,
        FOLDERS,
        "HEAD",
        ci_tree=False,
    )


def test_git_boundary_rejects_tracked_file_in_syncthing_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
        if args[0] == "ls-tree":
            return git_result(stdout="papers/tracked.pdf\n")
        return git_result()

    monkeypatch.setattr(boundary, "_run_git", fake_run_git)

    with pytest.raises(boundary.BoundaryError, match="tracked.pdf"):
        boundary.validate_git_boundary(
            tmp_path,
            FOLDERS,
            "HEAD",
            ci_tree=False,
        )


def test_live_config_must_contain_every_declared_folder(tmp_path: Path) -> None:
    config = tmp_path / "config.xml"
    config.write_text("<configuration />\n", encoding="utf-8")

    with pytest.raises(boundary.BoundaryError, match="gridfm-papers"):
        boundary.validate_syncthing_config(config, tmp_path, FOLDERS)


def test_layout_rejects_git_and_syncthing_overlap(tmp_path: Path) -> None:
    layout = tmp_path / "sync-layout.json"
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_local_folders": {"gridfm-papers": "papers"},
                "ownership": {
                    "git_roots": ["papers/manifests"],
                    "syncthing_roots": ["papers"],
                },
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(boundary.BoundaryError, match="ownership overlap"):
        boundary.load_layout(layout)
