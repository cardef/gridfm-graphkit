# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import pytest

from tools.check_syncthing_boundary import BoundaryError, validate_syncthing_config


FOLDERS = {
    "gridfm-mlruns": PurePosixPath("mlruns"),
    "gridfm-papers": PurePosixPath("papers"),
}


def _write_config(path: Path, folders: list[tuple[str, Path]]) -> Path:
    root = ET.Element("configuration")
    for folder_id, folder_path in folders:
        ET.SubElement(root, "folder", id=folder_id, path=str(folder_path))
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def test_accepts_declared_repo_local_roots(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = _write_config(
        tmp_path / "config.xml",
        [
            ("gridfm-mlruns", repo / "mlruns"),
            ("gridfm-papers", repo / "papers"),
        ],
    )

    assert validate_syncthing_config(config, repo, FOLDERS) == 2


def test_rejects_syncthing_root_containing_repository(tmp_path):
    repo = tmp_path / "sync-root" / "repo"
    repo.mkdir(parents=True)
    config = _write_config(
        tmp_path / "config.xml",
        [("unsafe", repo.parent)],
    )

    with pytest.raises(BoundaryError, match="contains the repository"):
        validate_syncthing_config(config, repo, FOLDERS)


def test_rejects_undeclared_root_inside_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = _write_config(
        tmp_path / "config.xml",
        [("unexpected", repo / "source")],
    )

    with pytest.raises(BoundaryError, match="undeclared Syncthing folder"):
        validate_syncthing_config(config, repo, FOLDERS)


def test_rejects_xml_entity_declarations(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "config.xml"
    config.write_text(
        '<!DOCTYPE configuration [<!ENTITY unsafe "value">]>'
        "<configuration></configuration>",
        encoding="utf-8",
    )

    with pytest.raises(BoundaryError, match="forbidden XML declaration"):
        validate_syncthing_config(config, repo, FOLDERS)
