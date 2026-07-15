# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Wrap reviewed PASS evidence in the strict campaign gate schema."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import (
    GATE_SCHEMA,
    GATE_EVIDENCE_KIND,
    REQUIRED_GATES,
    file_sha256,
    validate_gate_evidence,
)


def record_gate(
    gate_id: str,
    fork_commit: str,
    evidence: list[tuple[str, Path]],
    output: Path,
    repo_root: Path,
) -> dict:
    if gate_id not in REQUIRED_GATES:
        raise ContractError(f"unknown gate {gate_id}")
    if len(fork_commit) != 40:
        raise ContractError("gate requires a full fork commit")
    if not evidence:
        raise ContractError("gate requires at least one evidence artifact")
    records = []
    expected_kind = GATE_EVIDENCE_KIND[gate_id]
    for kind, path in evidence:
        if kind != expected_kind:
            raise ContractError(
                f"gate {gate_id} requires evidence kind {expected_kind}",
            )
        resolved = path.resolve()
        if not resolved.is_file():
            raise ContractError(f"missing gate evidence {resolved}")
        validate_gate_evidence(resolved, gate_id)
        try:
            stored_path = str(resolved.relative_to(repo_root))
        except ValueError:
            stored_path = str(resolved)
        records.append(
            {
                "kind": kind,
                "path": stored_path,
                "sha256": file_sha256(resolved),
            },
        )
    payload = {
        "schema_version": GATE_SCHEMA,
        "gate_id": gate_id,
        "status": "PASS",
        "fork_commit": fork_commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evidence": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--fork-commit", required=True)
    parser.add_argument(
        "--evidence",
        nargs="+",
        required=True,
        help="Gate-specific KIND=PATH evidence records.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    record_gate(
        args.gate_id,
        args.fork_commit,
        [
            (value.split("=", 1)[0], Path(value.split("=", 1)[1]))
            for value in args.evidence
            if "=" in value
        ],
        args.output,
        args.repo_root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
