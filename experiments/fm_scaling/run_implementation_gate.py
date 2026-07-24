# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Execute frozen CPU test packs and emit typed I002--I009 evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import GATE_EVIDENCE_KIND, file_sha256


@dataclass(frozen=True)
class GateSpec:
    tests: tuple[str, ...]
    inputs: tuple[str, ...]


GATE_SPECS = {
    "I002": GateSpec(
        tests=(
            "tests/test_fm_scaling_geometry.py::test_kron_and_quotient_share_partition_and_conservative_transport",
            "tests/test_fm_scaling_model.py::test_topology_manifest_rejects_outcome_fields",
            "tests/test_fm_scaling_data_pipeline.py::test_inventory_and_rendered_config_disable_all_structural_perturbations",
        ),
        inputs=(
            "gridfm_graphkit/fm_scaling/contracts.py",
            "gridfm_graphkit/fm_scaling/data.py",
            "tests/test_fm_scaling_geometry.py",
            "tests/test_fm_scaling_model.py",
            "tests/test_fm_scaling_data_pipeline.py",
        ),
    ),
    "I003": GateSpec(
        tests=(
            "tests/test_fm_scaling_geometry.py::test_partition_is_stable_id_permutation_invariant",
            "tests/test_fm_scaling_geometry.py::test_real_pymetis_backend_is_contiguous_and_deterministic",
            "tests/test_fm_scaling_geometry.py::test_partition_repairs_empty_backend_cells_deterministically",
            "tests/test_fm_scaling_geometry.py::test_partition_repairs_disconnected_backend_cells",
            "tests/test_fm_scaling_geometry.py::test_partition_uses_two_cells_for_small_positive_rho",
        ),
        inputs=(
            "gridfm_graphkit/fm_scaling/contracts.py",
            "gridfm_graphkit/fm_scaling/partition.py",
            "tests/test_fm_scaling_geometry.py",
        ),
    ),
    "I004": GateSpec(
        tests=(
            "tests/test_fm_scaling_geometry.py::test_kron_and_quotient_share_partition_and_conservative_transport",
            "tests/test_fm_scaling_geometry.py::test_common_sparsity_cap_fails_closed",
        ),
        inputs=(
            "gridfm_graphkit/fm_scaling/contracts.py",
            "gridfm_graphkit/fm_scaling/geometry.py",
            "tests/test_fm_scaling_geometry.py",
        ),
    ),
    "I005": GateSpec(
        tests=(
            "tests/test_fm_scaling_geometry.py::test_kron_and_quotient_share_partition_and_conservative_transport",
            "tests/test_fm_scaling_geometry.py::test_common_sparsity_cap_fails_closed",
        ),
        inputs=(
            "gridfm_graphkit/fm_scaling/contracts.py",
            "gridfm_graphkit/fm_scaling/geometry.py",
            "tests/test_fm_scaling_geometry.py",
        ),
    ),
    "I006": GateSpec(
        tests=(
            "tests/test_fm_scaling_geometry.py::test_geometry_bundle_round_trip_and_device_cache",
            "tests/test_fm_scaling_geometry.py::test_geometry_identity_excludes_measured_build_diagnostics",
        ),
        inputs=(
            "gridfm_graphkit/fm_scaling/contracts.py",
            "gridfm_graphkit/fm_scaling/registry.py",
            "tests/test_fm_scaling_geometry.py",
        ),
    ),
    "I007": GateSpec(
        tests=(
            "tests/test_fm_scaling_model.py::test_all_cores_share_output_schema_and_ignore_targets",
            "tests/test_fm_scaling_model.py::test_hierarchy_core_batches_multiple_topology_keys",
            "tests/test_fm_scaling_model.py::test_confirmatory_import_subprocess_denies_legacy_modules",
            "tests/test_fm_scaling_model.py::test_full_task_entrypoint_denies_legacy_modules",
            "tests/test_fm_scaling_analysis.py::test_output_gradient_parity_and_first_crossing_checkpoint",
        ),
        inputs=(
            "gridfm_graphkit/fm_scaling/communication.py",
            "gridfm_graphkit/fm_scaling/model.py",
            "gridfm_graphkit/fm_scaling/task.py",
            "gridfm_graphkit/fm_scaling/accounting.py",
            "tests/test_fm_scaling_model.py",
            "tests/test_fm_scaling_analysis.py",
        ),
    ),
    "I008": GateSpec(
        tests=(
            "tests/test_fm_scaling_model.py::test_case_declared_normalizer_fit_is_target_independent",
            "tests/test_fm_scaling_model.py::test_topology_manifest_rejects_outcome_fields",
            "tests/test_fm_scaling_data_pipeline.py",
            "tests/test_fm_scaling_preflight.py::test_datakit_identity_requires_exact_editable_clean_reachable_checkout",
        ),
        inputs=(
            "gridfm_graphkit/fm_scaling/data.py",
            "experiments/fm_scaling/prepare_data.py",
            "experiments/fm_scaling/finalize_data.py",
            "experiments/fm_scaling/freeze_targets.py",
            "experiments/fm_scaling/make_splits.py",
            "tests/test_fm_scaling_model.py",
            "tests/test_fm_scaling_data_pipeline.py",
            "tests/test_fm_scaling_preflight.py",
        ),
    ),
    "I009": GateSpec(
        tests=(
            "tests/test_fm_scaling_model.py::test_family_balanced_metric_is_masked_wrapped_and_euclidean",
            "tests/test_fm_scaling_model.py::test_confirmatory_task_uses_finite_graph_balanced_objective",
            "tests/test_fm_scaling_analysis.py",
            "tests/test_samplers.py",
        ),
        inputs=(
            "gridfm_graphkit/fm_scaling/loss.py",
            "gridfm_graphkit/fm_scaling/task.py",
            "gridfm_graphkit/fm_scaling/accounting.py",
            "gridfm_graphkit/fm_scaling/analysis.py",
            "gridfm_graphkit/datasets/samplers.py",
            "tests/test_fm_scaling_model.py",
            "tests/test_fm_scaling_analysis.py",
            "tests/test_samplers.py",
        ),
    ),
}


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_gate(
    gate_id: str,
    repo_root: Path,
    output: Path,
    log_path: Path,
) -> dict:
    if gate_id not in GATE_SPECS:
        raise ContractError(f"unsupported implementation gate {gate_id}")
    if _git(repo_root, "status", "--short"):
        raise ContractError("implementation gate requires a clean worktree")
    fork_commit = _git(repo_root, "rev-parse", "HEAD")
    containing = _git(repo_root, "branch", "-r", "--contains", fork_commit).splitlines()
    if "origin/research/kron-schur" not in {line.strip() for line in containing}:
        raise ContractError(
            "fork commit is not reachable from origin/research/kron-schur",
        )

    spec = GATE_SPECS[gate_id]
    command = [sys.executable, "-m", "pytest", "-q", *spec.tests]
    environment = os.environ.copy()
    environment["MLFLOW_ALLOW_FILE_STORE"] = "true"
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    transcript = completed.stdout + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("$ " + " ".join(command) + "\n" + transcript)
    print(transcript, end="")

    passed_match = re.search(r"(\d+) passed", transcript)
    passed_count = int(passed_match.group(1)) if passed_match else 0
    input_paths = tuple(
        dict.fromkeys(
            (
                "experiments/fm_scaling/run_implementation_gate.py",
                *spec.inputs,
            ),
        ),
    )
    inputs = []
    for relative in input_paths:
        path = (repo_root / relative).resolve()
        if not path.is_file():
            raise ContractError(f"missing gate input {path}")
        inputs.append({"path": str(path), "sha256": file_sha256(path)})

    passed = completed.returncode == 0
    payload = {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": gate_id,
        "status": "PASS" if passed else "FAIL",
        "checks": [
            {"name": f"{gate_id.lower()}_criteria", "passed": passed},
            {"name": "immutable_inputs", "passed": True},
        ],
        "inputs": inputs,
        "results": {
            "evidence_kind": GATE_EVIDENCE_KIND[gate_id],
            "fork_commit": fork_commit,
            "origin_ref": "origin/research/kron-schur",
            "command": command,
            "selected_tests": list(spec.tests),
            "passed_tests": passed_count,
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "log_path": str(log_path.resolve()),
            "log_sha256": file_sha256(log_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-id", choices=sorted(GATE_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    payload = run_gate(
        args.gate_id,
        args.repo_root.resolve(),
        args.output.resolve(),
        args.log.resolve(),
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
