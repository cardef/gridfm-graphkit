# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed M0 provenance and artifact-store preflight.

This module is intentionally independent of the legacy hierarchy and training
entry points. It records the repository/environment boundary and validates the
child MLflow store that I010 must later bind to the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import unquote, urlparse


CommandRunner = Callable[[Sequence[str], Path], str]
EXPECTED_GRAPHKIT_ORIGIN = "git@github.com:cardef/gridfm-graphkit.git"
EXPECTED_GRAPHKIT_UPSTREAM = "git@github.com:gridfm/gridfm-graphkit.git"
EXPECTED_DATAKIT_ORIGIN = "git@github.com:cardef/gridfm-datakit.git"
PINNED_UPSTREAM_COMMIT = (
    "b3d663b62179222c1ebec00ee29f67ea50e68c0b"  # pragma: allowlist secret
)


def _run(command: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_repo(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _editable_root(
    distribution_name: str,
) -> tuple[Path | None, bool, dict | None]:
    distribution = importlib.metadata.distribution(distribution_name)
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return None, False, None
    direct_url = json.loads(raw)
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file":
        return None, False, direct_url
    editable = direct_url.get("dir_info", {}).get("editable") is True
    return Path(unquote(parsed.path)).resolve(), editable, direct_url


def validate_repository_identity(
    origin_url: str,
    upstream_url: str,
    upstream_commit: str,
    merge_base: str,
    expected_upstream_commit: str,
    origin_ref: str,
    origin_refs_containing_head: list[str],
) -> list[dict]:
    return [
        {
            "name": "origin_is_research_fork",
            "passed": origin_url == EXPECTED_GRAPHKIT_ORIGIN,
            "detail": origin_url,
        },
        {
            "name": "upstream_is_canonical_repository",
            "passed": upstream_url == EXPECTED_GRAPHKIT_UPSTREAM,
            "detail": upstream_url,
        },
        {
            "name": "upstream_commit_matches_plan_pin",
            "passed": upstream_commit == expected_upstream_commit,
            "detail": upstream_commit,
        },
        {
            "name": "merge_base_matches_plan_pin",
            "passed": merge_base == expected_upstream_commit,
            "detail": merge_base,
        },
        {
            "name": "fork_commit_is_reachable_from_origin_ref",
            "passed": origin_ref in origin_refs_containing_head,
            "detail": (
                f"required={origin_ref}; containing_refs={origin_refs_containing_head}"
            ),
        },
    ]


def validate_datakit_identity(
    git_root: Path,
    expected_root: Path,
    editable_root: Path | None,
    editable: bool,
    origin_url: str,
    commit: str,
    expected_commit: str,
    worktree_state: str,
    origin_ref: str,
    origin_refs_containing_commit: list[str],
) -> list[dict]:
    return [
        {
            "name": "datakit_git_root_is_exact_checkout",
            "passed": git_root == expected_root,
            "detail": str(git_root),
        },
        {
            "name": "datakit_editable_root_is_exact_checkout",
            "passed": editable_root == expected_root,
            "detail": str(editable_root),
        },
        {
            "name": "datakit_install_is_editable",
            "passed": editable,
            "detail": f"editable={editable}",
        },
        {
            "name": "datakit_origin_is_research_fork",
            "passed": origin_url == EXPECTED_DATAKIT_ORIGIN,
            "detail": origin_url,
        },
        {
            "name": "datakit_commit_matches_pin",
            "passed": commit == expected_commit,
            "detail": commit,
        },
        {
            "name": "datakit_worktree_is_clean",
            "passed": not worktree_state,
            "detail": worktree_state or "clean",
        },
        {
            "name": "datakit_commit_is_reachable_from_origin_ref",
            "passed": origin_ref in origin_refs_containing_commit,
            "detail": (
                f"required={origin_ref}; containing_refs="
                f"{origin_refs_containing_commit}"
            ),
        },
    ]


def validate_store_layout(artifact_root: Path, mlflow_store: Path) -> list[dict]:
    """Return deterministic checks for Syncthing/MLflow path isolation."""
    root = artifact_root.resolve()
    store = mlflow_store.resolve()
    strict_child = store != root and root in store.parents
    return [
        {
            "name": "mlflow_store_is_strict_child",
            "passed": strict_child,
            "detail": f"artifact_root={root}; mlflow_store={store}",
        },
        {
            "name": "mlflow_store_has_no_syncthing_marker",
            "passed": not (store / ".stfolder").exists(),
            "detail": str(store / ".stfolder"),
        },
        {
            "name": "syncthing_marker_is_outside_store",
            "passed": not (root / ".stfolder").exists() or strict_child,
            "detail": str(root / ".stfolder"),
        },
    ]


def smoke_mlflow_store(mlflow_store: Path) -> dict:
    """Create, find, and delete a disposable experiment in the exact store."""
    store = mlflow_store.resolve()
    marker = store / ".stfolder"
    if marker.exists():
        return {
            "name": "mlflow_store_create_search_smoke",
            "passed": False,
            "detail": f"refusing store containing {marker}",
        }

    experiment_id = None
    client = None
    passed = False
    detail = f"tracking_uri={store.as_uri()}"
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        from mlflow.tracking import MlflowClient

        store.mkdir(parents=True, exist_ok=True)
        client = MlflowClient(tracking_uri=store.as_uri())
        name = f"__gridfm_preflight_{uuid.uuid4().hex}"
        experiment_id = client.create_experiment(name)
        found = client.get_experiment_by_name(name)
        passed = found is not None and found.experiment_id == experiment_id
        detail += f"; experiment_id={experiment_id}"
    except Exception as error:  # fail closed and preserve the primary error
        passed = False
        detail += f"; {type(error).__name__}: {error}"
    finally:
        if experiment_id is not None and client is not None:
            try:
                client.delete_experiment(experiment_id)
                deleted = client.get_experiment(experiment_id)
                cleanup_passed = deleted.lifecycle_stage == "deleted"
                passed = passed and cleanup_passed
                detail += f"; cleanup_deleted={cleanup_passed}"
            except Exception as error:
                passed = False
                detail += f"; cleanup_{type(error).__name__}: {error}"
    return {
        "name": "mlflow_store_create_search_smoke",
        "passed": passed,
        "detail": detail,
    }


def _distribution_lock() -> list[dict[str, str]]:
    packages = {
        (dist.metadata.get("Name") or "unknown").lower(): dist.version
        for dist in importlib.metadata.distributions()
    }
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def build_record(
    repo_root: Path,
    artifact_root: Path,
    mlflow_store: Path,
    expected_datakit_root: Path,
    expected_datakit_commit: str,
    upstream_ref: str = "upstream/main",
    origin_ref: str = "origin/research/kron-schur",
    datakit_origin_ref: str = "origin/main",
    expected_upstream_commit: str = PINNED_UPSTREAM_COMMIT,
    runner: CommandRunner = _run,
) -> dict:
    """Build I001 evidence while mutating only the ignored MLflow store."""
    repo_root = repo_root.resolve()
    artifact_root = _resolve_from_repo(artifact_root, repo_root)
    mlflow_store = _resolve_from_repo(mlflow_store, repo_root)
    expected_datakit_root = _resolve_from_repo(
        expected_datakit_root,
        repo_root,
    )
    checks = validate_store_layout(artifact_root, mlflow_store)
    if all(check["passed"] for check in checks[:2]):
        checks.append(smoke_mlflow_store(mlflow_store))
    else:
        checks.append(
            {
                "name": "mlflow_store_create_search_smoke",
                "passed": False,
                "detail": "skipped because store layout is unsafe",
            },
        )

    head = runner(["git", "rev-parse", "HEAD"], repo_root)
    upstream_commit = runner(["git", "rev-parse", upstream_ref], repo_root)
    merge_base = runner(["git", "merge-base", "HEAD", upstream_ref], repo_root)
    worktree_state = runner(["git", "status", "--porcelain"], repo_root)
    origin_url = runner(["git", "remote", "get-url", "origin"], repo_root)
    upstream_url = runner(["git", "remote", "get-url", "upstream"], repo_root)
    origin_refs_containing_head = runner(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short)",
            "--contains",
            head,
            "refs/remotes/origin/",
        ],
        repo_root,
    ).splitlines()

    import gridfm_datakit
    import torch

    datakit_module = Path(gridfm_datakit.__file__).resolve()
    datakit_git_root = Path(
        runner(["git", "rev-parse", "--show-toplevel"], datakit_module.parent),
    ).resolve()
    datakit_commit = runner(["git", "rev-parse", "HEAD"], datakit_git_root)
    datakit_status = runner(["git", "status", "--porcelain"], datakit_git_root)
    datakit_origin = runner(
        ["git", "remote", "get-url", "origin"],
        datakit_git_root,
    )
    datakit_origin_refs = runner(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short)",
            "--contains",
            datakit_commit,
            "refs/remotes/origin/",
        ],
        datakit_git_root,
    ).splitlines()
    datakit_editable_root, datakit_editable, datakit_direct_url = _editable_root(
        "gridfm-datakit",
    )
    checks.append(
        {
            "name": "worktree_is_clean",
            "passed": not worktree_state,
            "detail": worktree_state or "clean",
        },
    )
    checks.extend(
        validate_repository_identity(
            origin_url=origin_url,
            upstream_url=upstream_url,
            upstream_commit=upstream_commit,
            merge_base=merge_base,
            expected_upstream_commit=expected_upstream_commit,
            origin_ref=origin_ref,
            origin_refs_containing_head=origin_refs_containing_head,
        ),
    )
    checks.extend(
        validate_datakit_identity(
            git_root=datakit_git_root,
            expected_root=expected_datakit_root,
            editable_root=datakit_editable_root,
            editable=datakit_editable,
            origin_url=datakit_origin,
            commit=datakit_commit,
            expected_commit=expected_datakit_commit,
            worktree_state=datakit_status,
            origin_ref=datakit_origin_ref,
            origin_refs_containing_commit=datakit_origin_refs,
        ),
    )

    pyproject = repo_root / "pyproject.toml"
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "gate_id": "I001",
        "generated_at": now,
        "status": "PASS" if all(c["passed"] for c in checks) else "BLOCKED",
        "checks": checks,
        "repository": {
            "root": str(repo_root),
            "origin_url": origin_url,
            "fork_commit": head,
            "worktree_state": worktree_state or "clean",
            "upstream_url": upstream_url,
            "upstream_ref": upstream_ref,
            "upstream_commit": upstream_commit,
            "merge_base": merge_base,
            "origin_ref": origin_ref,
            "origin_refs_containing_head": origin_refs_containing_head,
        },
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            ),
            "datakit_version": importlib.metadata.version("gridfm-datakit"),
            "datakit_module": str(datakit_module),
            "datakit_git_root": str(datakit_git_root),
            "datakit_origin_url": datakit_origin,
            "datakit_commit": datakit_commit,
            "datakit_worktree_state": datakit_status or "clean",
            "datakit_editable_root": str(datakit_editable_root),
            "datakit_editable": datakit_editable,
            "datakit_direct_url": datakit_direct_url,
            "expected_datakit_root": str(expected_datakit_root),
            "expected_datakit_commit": expected_datakit_commit,
            "datakit_origin_ref": datakit_origin_ref,
            "datakit_origin_refs_containing_commit": datakit_origin_refs,
            "pyproject_sha256": _sha256(pyproject),
            "distributions": _distribution_lock(),
        },
        "artifact_layout": {
            "syncthing_root": str(artifact_root.resolve()),
            "mlflow_store": str(mlflow_store.resolve()),
        },
        "clean_clone": {
            "commands": [
                f"git clone {origin_url} gridfm-graphkit",
                "cd gridfm-graphkit",
                f"git checkout {head}",
                f"git clone {datakit_origin} ../gridfm-datakit",
                f"git -C ../gridfm-datakit checkout {datakit_commit}",
                "uv pip install --python ../.venv/bin/python -e '.[dev,test]'",
                ("uv pip install --python ../.venv/bin/python -e ../gridfm-datakit"),
            ],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, default=Path("mlruns"))
    parser.add_argument(
        "--mlflow-store",
        type=Path,
        default=Path("mlruns/fm-scaling/mlflow-store"),
    )
    parser.add_argument(
        "--expected-datakit-root",
        type=Path,
        default=Path("../gridfm-datakit"),
    )
    parser.add_argument("--expected-datakit-commit", required=True)
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument("--origin-ref", default="origin/research/kron-schur")
    parser.add_argument("--datakit-origin-ref", default="origin/main")
    parser.add_argument(
        "--expected-upstream-commit",
        default=PINNED_UPSTREAM_COMMIT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _blocked_record(error: Exception) -> dict:
    return {
        "schema_version": 1,
        "gate_id": "I001",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED",
        "fatal_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    output = _resolve_from_repo(args.output, repo_root)
    try:
        record = build_record(
            repo_root=repo_root,
            artifact_root=args.artifact_root,
            mlflow_store=args.mlflow_store,
            expected_datakit_root=args.expected_datakit_root,
            expected_datakit_commit=args.expected_datakit_commit,
            upstream_ref=args.upstream_ref,
            origin_ref=args.origin_ref,
            datakit_origin_ref=args.datakit_origin_ref,
            expected_upstream_commit=args.expected_upstream_commit,
        )
    except Exception as error:  # fail closed and preserve primary evidence
        record = _blocked_record(error)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "output": str(output)}))
    return 0 if record["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
