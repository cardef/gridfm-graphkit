# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Run the CPU-compatible subset of I010 and preserve remaining blockers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import file_sha256


UPSTREAM_COMMIT = "b3d663b62179222c1ebec00ee29f67ea50e68c0b"  # pragma: allowlist secret


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def mlflow_smoke_passed(preflight: dict) -> bool:
    return any(
        check.get("name") == "mlflow_store_create_search_smoke"
        and check.get("passed") is True
        for check in preflight.get("checks", [])
    )


def check_cpu_compatibility(
    repo_root: Path,
    i001_evidence_path: Path,
    output: Path,
    log_path: Path,
) -> dict:
    if _git(repo_root, "status", "--short"):
        raise ContractError("I010 CPU checks require a clean worktree")
    fork_commit = _git(repo_root, "rev-parse", "HEAD")
    merge_base = _git(repo_root, "merge-base", "HEAD", "upstream/main")
    upstream_identity = (
        _git(repo_root, "rev-parse", "upstream/main") == UPSTREAM_COMMIT
        and merge_base == UPSTREAM_COMMIT
    )
    i001 = json.loads(i001_evidence_path.read_text())
    mlflow_passed = i001.get("status") == "PASS" and mlflow_smoke_passed(i001)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="fm-scaling-clean-clone-") as temporary:
        clone = Path(temporary) / "gridfm-graphkit"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(repo_root), str(clone)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--quiet", fork_commit],
            cwd=clone,
            check=True,
        )
        environment = os.environ.copy()
        environment["MLFLOW_ALLOW_FILE_STORE"] = "true"
        environment["PYTHONPATH"] = str(clone)
        import_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import gridfm_graphkit,pathlib; "
                "print(pathlib.Path(gridfm_graphkit.__file__).resolve())",
            ],
            cwd=clone,
            env=environment,
            capture_output=True,
            text=True,
        )
        clone_imported = import_probe.returncode == 0 and Path(
            import_probe.stdout.strip(),
        ).resolve().is_relative_to(clone.resolve())
        tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_fm_scaling_model.py::test_all_cores_share_output_schema_and_ignore_targets[flat]",
                "tests/test_fm_scaling_model.py::test_gpu_probe_reproduces_generator_q_limit_merge",
                "tests/test_fm_scaling_analysis.py::test_output_gradient_parity_and_first_crossing_checkpoint",
            ],
            cwd=clone,
            env=environment,
            capture_output=True,
            text=True,
        )
    elapsed = time.monotonic() - started
    transcript = (
        "$ clean-clone import probe\n"
        + import_probe.stdout
        + import_probe.stderr
        + "$ clean-clone CPU compatibility tests\n"
        + tests.stdout
        + tests.stderr
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(transcript)
    print(transcript, end="")

    checks = [
        {"name": "upstream_identity", "passed": upstream_identity},
        {"name": "mlflow_child_store_smoke", "passed": mlflow_passed},
        {"name": "clean_clone_import", "passed": clone_imported},
        {"name": "flat_schema_and_flop_cpu", "passed": tests.returncode == 0},
        {
            "name": "upstream_flat_checkpoint_load",
            "passed": False,
            "reason": "no reviewed upstream-flat checkpoint fixture exists",
        },
        {
            "name": "cuda_compile_and_flop_parity",
            "passed": False,
            "reason": "CUDA unavailable on the local CPU host",
        },
        {
            "name": "largest_grid_host_and_accelerator_peaks",
            "passed": False,
            "reason": "admissible generated data and largest-grid measurements are absent",
        },
    ]
    passed = all(check["passed"] for check in checks)
    payload = {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": "I010",
        "status": "PASS" if passed else "BLOCKED",
        "checks": checks,
        "inputs": [
            {
                "path": str(i001_evidence_path.resolve()),
                "sha256": file_sha256(i001_evidence_path.resolve()),
            },
            {
                "path": str(
                    (
                        repo_root / "experiments/fm_scaling/check_cpu_compatibility.py"
                    ).resolve(),
                ),
                "sha256": file_sha256(
                    repo_root / "experiments/fm_scaling/check_cpu_compatibility.py",
                ),
            },
        ],
        "results": {
            "fork_commit": fork_commit,
            "upstream_commit": UPSTREAM_COMMIT,
            "merge_base": merge_base,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
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
    parser.add_argument("--i001-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    payload = check_cpu_compatibility(
        args.repo_root.resolve(),
        args.i001_evidence.resolve(),
        args.output.resolve(),
        args.log.resolve(),
    )
    print(json.dumps({"I010": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
