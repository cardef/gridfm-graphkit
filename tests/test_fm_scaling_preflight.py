# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import experiments.fm_scaling.preflight as preflight
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import validate_gate_evidence
from experiments.fm_scaling.preflight import (
    _resolve_from_repo,
    smoke_mlflow_store,
    validate_datakit_identity,
    validate_repository_identity,
    validate_store_layout,
)


def _checks_by_name(checks):
    return {check["name"]: check["passed"] for check in checks}


def test_real_cli_import_denies_all_legacy_confirmatory_modules():
    script = textwrap.dedent(
        """
        import importlib.abc
        import runpy
        import sys
        forbidden = {
            'gridfm_graphkit.models.gnn_hetero_hier',
            'gridfm_graphkit.datasets.hierarchy',
            'gridfm_graphkit.datasets.normalizers',
        }
        class Deny(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in forbidden or any(
                    fullname.startswith(name + '.') for name in forbidden
                ):
                    raise RuntimeError('forbidden import: ' + fullname)
        sys.meta_path.insert(0, Deny())
        sys.argv = ['gridfm_graphkit', 'train', '--help']
        try:
            runpy.run_module('gridfm_graphkit', run_name='__main__')
        except SystemExit as error:
            assert error.code == 0
        assert not (forbidden & set(sys.modules))
        """,
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_syncthing_marker_at_artifact_root_is_isolated(tmp_path):
    artifact_root = tmp_path / "mlruns"
    store = artifact_root / "fm-scaling" / "mlflow-store"
    (artifact_root / ".stfolder").mkdir(parents=True)
    store.mkdir(parents=True)

    checks = _checks_by_name(validate_store_layout(artifact_root, store))

    assert checks == {
        "mlflow_store_is_strict_child": True,
        "mlflow_store_has_no_syncthing_marker": True,
        "syncthing_marker_is_outside_store": True,
    }


def test_syncthing_root_cannot_be_mlflow_store(tmp_path):
    artifact_root = tmp_path / "mlruns"
    (artifact_root / ".stfolder").mkdir(parents=True)

    checks = _checks_by_name(validate_store_layout(artifact_root, artifact_root))

    assert checks["mlflow_store_is_strict_child"] is False
    assert checks["mlflow_store_has_no_syncthing_marker"] is False


def test_marker_inside_child_store_blocks_preflight(tmp_path):
    artifact_root = tmp_path / "mlruns"
    store = artifact_root / "fm-scaling" / "mlflow-store"
    (store / ".stfolder").mkdir(parents=True)

    checks = _checks_by_name(validate_store_layout(artifact_root, store))

    assert checks["mlflow_store_is_strict_child"] is True
    assert checks["mlflow_store_has_no_syncthing_marker"] is False


def test_mlflow_store_create_search_smoke(tmp_path):
    result = smoke_mlflow_store(tmp_path / "mlflow-store")

    assert result["passed"] is True, result["detail"]
    assert "cleanup_deleted=True" in result["detail"]


def test_marker_refuses_mlflow_smoke(tmp_path):
    store = tmp_path / "mlflow-store"
    (store / ".stfolder").mkdir(parents=True)

    result = smoke_mlflow_store(store)

    assert result["passed"] is False
    assert "refusing store containing" in result["detail"]


def test_relative_paths_are_anchored_to_repo_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    repo_root.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = _resolve_from_repo(Path("mlruns/fm-scaling"), repo_root)

    assert resolved == repo_root / "mlruns" / "fm-scaling"


def test_repository_identity_requires_exact_remotes_pins_and_reachability():
    checks = _checks_by_name(
        validate_repository_identity(
            origin_url="git@github.com:wrong/gridfm-graphkit.git",
            upstream_url="git@github.com:gridfm/gridfm-graphkit.git",
            upstream_commit="upstream-pin",
            merge_base="upstream-pin",
            expected_upstream_commit="upstream-pin",
            origin_ref="origin/research/kron-schur",
            origin_refs_containing_head=[],
        ),
    )

    assert checks["origin_is_research_fork"] is False
    assert checks["upstream_is_canonical_repository"] is True
    assert checks["upstream_commit_matches_plan_pin"] is True
    assert checks["merge_base_matches_plan_pin"] is True
    assert checks["fork_commit_is_reachable_from_origin_ref"] is False


def test_datakit_identity_requires_exact_editable_clean_reachable_checkout(
    tmp_path,
):
    expected = tmp_path / "gridfm-datakit"
    nested = expected / ".claude" / "worktrees" / "other"
    checks = _checks_by_name(
        validate_datakit_identity(
            git_root=nested,
            expected_root=expected,
            editable_root=nested,
            editable=False,
            origin_url="git@github.com:cardef/gridfm-datakit.git",
            commit="nested-commit",
            expected_commit="expected-commit",
            worktree_state="?? dirty",
            origin_ref="origin/main",
            origin_refs_containing_commit=[],
        ),
    )

    assert checks["datakit_git_root_is_exact_checkout"] is False
    assert checks["datakit_editable_root_is_exact_checkout"] is False
    assert checks["datakit_install_is_editable"] is False
    assert checks["datakit_origin_is_research_fork"] is True
    assert checks["datakit_commit_matches_pin"] is False
    assert checks["datakit_worktree_is_clean"] is False
    assert checks["datakit_commit_is_reachable_from_origin_ref"] is False


def test_main_converts_provenance_exception_to_blocked_json(
    tmp_path,
    monkeypatch,
):
    output = Path("evidence/I001.json")
    args = argparse.Namespace(
        repo_root=tmp_path,
        artifact_root=Path("mlruns"),
        mlflow_store=Path("mlruns/fm-scaling/mlflow-store"),
        expected_datakit_root=Path("../gridfm-datakit"),
        expected_datakit_commit="pin",
        upstream_ref="upstream/main",
        origin_ref="origin/research/kron-schur",
        datakit_origin_ref="origin/main",
        expected_upstream_commit="upstream-pin",
        output=output,
    )
    monkeypatch.setattr(preflight, "_parse_args", lambda: args)

    def fail(**_kwargs):
        raise RuntimeError("missing upstream ref")

    monkeypatch.setattr(preflight, "build_record", fail)

    return_code = preflight.main()
    record = json.loads((tmp_path / output).read_text())

    assert return_code == 2
    assert record["status"] == "BLOCKED"
    assert record["fatal_error"] == {
        "type": "RuntimeError",
        "message": "missing upstream ref",
    }


def test_gate_evidence_cannot_wrap_an_arbitrary_pass_file(tmp_path):
    path = tmp_path / "arbitrary.json"
    path.write_text(json.dumps({"status": "PASS"}))
    with pytest.raises(ContractError, match="typed schema"):
        validate_gate_evidence(path, "R003")
