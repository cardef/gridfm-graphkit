#!/usr/bin/env python3
# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Fail if Syncthing can overlap deployment files tracked by ``main``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath


class BoundaryError(RuntimeError):
    """Raised when the Git/Syncthing ownership boundary is unsafe."""


MAX_CONFIG_BYTES = 10 * 1024 * 1024


def load_layout(path: Path) -> dict[str, PurePosixPath]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise BoundaryError(f"unsupported layout version in {path}")

    raw_folders = data.get("repo_local_folders")
    if not isinstance(raw_folders, dict) or not raw_folders:
        raise BoundaryError(f"{path} must declare repo_local_folders")

    folders: dict[str, PurePosixPath] = {}
    for folder_id, raw_path in raw_folders.items():
        if not isinstance(folder_id, str) or not isinstance(raw_path, str):
            raise BoundaryError("folder ids and paths must be strings")
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or relative == PurePosixPath("."):
            raise BoundaryError(f"unsafe repo-local path for {folder_id}: {raw_path}")
        if ".." in relative.parts:
            raise BoundaryError(f"path escapes repository for {folder_id}: {raw_path}")
        folders[folder_id] = relative

    roots = list(folders.items())
    for index, (folder_id, root) in enumerate(roots):
        for other_id, other in roots[index + 1 :]:
            if root == other or root in other.parents or other in root.parents:
                raise BoundaryError(
                    f"overlapping roots: {folder_id}={root} and {other_id}={other}",
                )

    raw_ownership = data.get("ownership")
    if not isinstance(raw_ownership, dict):
        raise BoundaryError(f"{path} must declare ownership")

    ownership: dict[str, list[PurePosixPath]] = {}
    for owner in ("git_roots", "syncthing_roots"):
        raw_roots = raw_ownership.get(owner)
        if not isinstance(raw_roots, list) or not all(
            isinstance(root, str) for root in raw_roots
        ):
            raise BoundaryError(f"{path} ownership.{owner} must be a path list")
        parsed = [PurePosixPath(root) for root in raw_roots]
        if any(root.is_absolute() or ".." in root.parts for root in parsed):
            raise BoundaryError(f"{path} ownership.{owner} contains an unsafe path")
        ownership[owner] = parsed

    declared_sync_roots = set(folders.values())
    if set(ownership["syncthing_roots"]) != declared_sync_roots:
        raise BoundaryError(
            "ownership.syncthing_roots must exactly match repo_local_folders",
        )

    for git_root in ownership["git_roots"]:
        for sync_root in ownership["syncthing_roots"]:
            if (
                git_root == sync_root
                or git_root in sync_root.parents
                or sync_root in git_root.parents
            ):
                raise BoundaryError(
                    f"Git and Syncthing ownership overlap: {git_root} and {sync_root}",
                )
    return folders


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BoundaryError("git is required for the local boundary check") from exc


def validate_git_boundary(
    repo: Path,
    folders: dict[str, PurePosixPath],
    ref: str,
    *,
    ci_tree: bool,
) -> None:
    tree = _run_git(repo, "ls-tree", "-r", "--name-only", ref)
    if tree.returncode != 0:
        raise BoundaryError(f"cannot read Git tree {ref}: {tree.stderr.strip()}")
    tracked = [PurePosixPath(line) for line in tree.stdout.splitlines() if line]

    for folder_id, relative in folders.items():
        offenders = [
            path for path in tracked if path == relative or relative in path.parents
        ]
        if offenders:
            preview = ", ".join(str(path) for path in offenders[:5])
            raise BoundaryError(
                f"{ref} tracks files inside Syncthing root {folder_id}={relative}: {preview}",
            )

        probe = repo / Path(*relative.parts) / ".syncthing-boundary-probe"
        ignored = _run_git(
            repo,
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            str(probe),
        )
        if ignored.returncode != 0:
            raise BoundaryError(f"Syncthing root is not ignored by Git: {relative}")

        if ci_tree:
            root = repo / Path(*relative.parts)
            if root.exists() and any(
                path.is_file() or path.is_symlink() for path in root.rglob("*")
            ):
                raise BoundaryError(
                    f"CI checkout contains files inside Syncthing-only root: {relative}",
                )


def _resolve_syncthing_path(raw_path: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path.home() / path
    return path.resolve(strict=False)


def validate_syncthing_config(
    config_path: Path,
    repo: Path,
    folders: dict[str, PurePosixPath],
) -> int:
    try:
        raw_config = config_path.read_bytes()
    except OSError as exc:
        raise BoundaryError(
            f"cannot parse Syncthing config {config_path}: {exc}",
        ) from exc
    if len(raw_config) > MAX_CONFIG_BYTES:
        raise BoundaryError(f"Syncthing config exceeds {MAX_CONFIG_BYTES} bytes")
    upper_config = raw_config.upper()
    if b"<!DOCTYPE" in upper_config or b"<!ENTITY" in upper_config:
        raise BoundaryError("Syncthing config contains a forbidden XML declaration")
    try:
        # DTD/entity declarations are rejected above before using stdlib XML.
        root = ET.fromstring(raw_config)  # nosec B314
    except ET.ParseError as exc:
        raise BoundaryError(
            f"cannot parse Syncthing config {config_path}: {exc}",
        ) from exc

    repo = repo.resolve()
    expected = {
        folder_id: (repo / Path(*relative.parts)).resolve(strict=False)
        for folder_id, relative in folders.items()
    }
    seen_ids: set[str] = set()
    checked = 0

    for element in root.findall("folder"):
        folder_id = element.get("id")
        raw_path = element.get("path")
        if not folder_id or not raw_path:
            raise BoundaryError("Syncthing folder is missing id or path")
        if folder_id in seen_ids:
            raise BoundaryError(f"duplicate Syncthing folder id: {folder_id}")
        seen_ids.add(folder_id)

        sync_root = _resolve_syncthing_path(raw_path)
        if sync_root == repo or sync_root in repo.parents:
            raise BoundaryError(
                f"Syncthing root {folder_id}={sync_root} contains the repository",
            )

        if repo in sync_root.parents:
            try:
                relative = sync_root.relative_to(repo)
            except ValueError as exc:  # pragma: no cover - guarded above
                raise BoundaryError(
                    f"cannot resolve Syncthing root {sync_root}",
                ) from exc
            declared = expected.get(folder_id)
            if declared is None:
                raise BoundaryError(
                    f"undeclared Syncthing folder inside repository: {folder_id}={relative}",
                )
            if sync_root != declared:
                raise BoundaryError(
                    f"Syncthing folder {folder_id} maps to {relative}, expected "
                    f"{declared.relative_to(repo)}",
                )
            checked += 1
        elif folder_id in expected and sync_root != expected[folder_id]:
            raise BoundaryError(
                f"declared folder {folder_id} moved to {sync_root}; update sync-layout.json",
            )

    missing = sorted(set(expected) - seen_ids)
    if missing:
        raise BoundaryError(
            f"declared Syncthing folders missing from live config: {', '.join(missing)}",
        )

    return checked


def default_syncthing_config() -> Path | None:
    candidates = [
        Path.home() / "Library/Application Support/Syncthing/config.xml",
        Path.home() / ".config/syncthing/config.xml",
    ]
    return next((path for path in candidates if path.is_file()), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, help="machine-readable layout file")
    parser.add_argument("--syncthing-config", type=Path, help="Syncthing config.xml")
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git tree to protect (defaults to the checked-out tree)",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="skip the machine-local Syncthing configuration check",
    )
    parser.add_argument(
        "--require-local-config",
        action="store_true",
        help="fail when no Syncthing config is available",
    )
    parser.add_argument(
        "--ci-tree",
        action="store_true",
        help="also require declared sync roots to be absent from the checkout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    layout_path = args.layout or repo / "sync-layout.json"
    try:
        folders = load_layout(layout_path)
        validate_git_boundary(repo, folders, args.ref, ci_tree=args.ci_tree)

        checked = 0
        if not args.static_only:
            config_path = args.syncthing_config or default_syncthing_config()
            if config_path is None:
                if args.require_local_config:
                    raise BoundaryError("no Syncthing config found")
            else:
                checked = validate_syncthing_config(config_path, repo, folders)
    except BoundaryError as exc:
        print(f"Syncthing boundary FAILED: {exc}", file=sys.stderr)
        return 1

    print(
        f"Syncthing boundary OK: {args.ref} has no tracked files under "
        f"{len(folders)} declared roots; checked {checked} live mappings",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
