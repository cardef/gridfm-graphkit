# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Inventory a pinned PGLib checkout and audit the R002 split constraints."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.manifest import file_sha256


SOURCE_SCHEMA = "fm-scaling-pglib-source-v1"
INVENTORY_SCHEMA = "fm-scaling-pglib-inventory-v1"
SPLIT_AUDIT_SCHEMA = "fm-scaling-r002-audit-v1"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def provenance_group(case_name: str) -> str:
    suffix = re.sub(r"^case[0-9]+(?:sp|sop|wop|wp)?_?", "", case_name)
    if suffix.startswith("ieee"):
        return "ieee"
    return suffix.split("_")[-1]


def parse_case(path: Path) -> dict:
    text = path.read_text(errors="strict")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", text)
    bus_match = re.search(r"mpc\.bus\s*=\s*\[(.*?)\];", text, re.DOTALL)
    if base_match is None or bus_match is None:
        raise ContractError(f"cannot parse PGLib metadata from {path}")
    case_name = path.stem.removeprefix("pglib_opf_")
    buses = [
        line
        for line in bus_match.group(1).splitlines()
        if line.strip() and not line.lstrip().startswith("%")
    ]
    encoded_match = re.match(r"case([0-9]+)", case_name)
    if encoded_match is None:
        raise ContractError(f"case name has no encoded bus count: {case_name}")
    bus_count = len(buses)
    encoded_bus_count = int(encoded_match.group(1))
    base_mva = float(base_match.group(1))
    if base_mva <= 0:
        raise ContractError(f"case has nonpositive baseMVA: {case_name}")
    return {
        "network": case_name,
        "topology_key": f"pglib-v23:{case_name}",
        "provenance_group": provenance_group(case_name),
        "bus_count": bus_count,
        "encoded_bus_count": encoded_bus_count,
        "baseMVA": base_mva,
        "integrity_status": (
            "PASS"
            if encoded_bus_count == bus_count
            else "WARN_ENCODED_BUS_COUNT_MISMATCH"
        ),
        "raw_sha256": file_sha256(path),
    }


def audit_split(cases: list[dict], source: dict) -> dict:
    requirements = source["requirements"]
    target_range = source["target_bus_range"]
    source_count = int(requirements["source_count"])
    target_count = int(requirements["target_count"])
    target_group_count = int(requirements["target_group_count"])
    extrapolative_count = int(requirements["extrapolative_target_count"])
    extrapolative_group_count = int(
        requirements["extrapolative_target_group_count"],
    )
    eligible_cases = [
        case for case in cases if case.get("integrity_status", "PASS") == "PASS"
    ]
    groups = sorted({case["provenance_group"] for case in eligible_cases})
    feasible = []
    for count in range(1, len(groups) + 1):
        for target_groups_tuple in itertools.combinations(groups, count):
            target_groups = set(target_groups_tuple)
            source_cases = sorted(
                [
                    case
                    for case in eligible_cases
                    if case["provenance_group"] not in target_groups
                ],
                key=lambda case: (case["bus_count"], case["network"]),
            )
            target_cases = sorted(
                [
                    case
                    for case in eligible_cases
                    if case["provenance_group"] in target_groups
                    and int(target_range["minimum"])
                    <= case["bus_count"]
                    <= int(target_range["maximum"])
                ],
                key=lambda case: (case["bus_count"], case["network"]),
            )
            observed_target_groups = {case["provenance_group"] for case in target_cases}
            if (
                len(source_cases) < source_count
                or len(target_cases) < target_count
                or len(observed_target_groups) < target_group_count
            ):
                continue
            selected_sources = source_cases[:source_count]
            source_max = max(case["bus_count"] for case in selected_sources)
            extrapolative = [
                case for case in target_cases if case["bus_count"] > source_max
            ]
            if (
                len(extrapolative) < extrapolative_count
                or len({case["provenance_group"] for case in extrapolative})
                < extrapolative_group_count
            ):
                continue
            feasible.append(
                {
                    "target_groups": list(target_groups_tuple),
                    "source_networks": [case["network"] for case in selected_sources],
                    "target_networks": [case["network"] for case in target_cases],
                    "source_max_bus_count": source_max,
                    "extrapolative_targets": [
                        case["network"] for case in extrapolative
                    ],
                },
            )
        if feasible:
            break
    selected = min(
        feasible,
        key=lambda item: (
            item["source_max_bus_count"],
            -len(item["target_networks"]),
            item["target_groups"],
        ),
        default=None,
    )
    return {
        "schema_version": SPLIT_AUDIT_SCHEMA,
        "gate_id": "R002",
        "status": "PASS" if selected is not None else "BLOCKED",
        "selection_rule": (
            "fewest_held_out_groups_then_min_source_max_then_max_targets"
        ),
        "requirements": requirements,
        "target_bus_range": target_range,
        "candidate_group_count": len(groups),
        "feasible_assignment_count_at_minimum_group_count": len(feasible),
        "selected": selected,
        "block_reason": (
            None
            if selected is not None
            else "no whole-provenance-group assignment satisfies the frozen "
            "source, target, independent-group, and size-extrapolation constraints"
        ),
    }


def typed_split_evidence(
    audit: dict,
    source_path: Path,
    inventory_path: Path,
) -> dict:
    """Bind a deterministic split audit to immutable formal-gate inputs."""
    return {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": "R002",
        "status": audit["status"],
        "checks": [
            {
                "name": "r002_criteria",
                "passed": audit["status"] == "PASS",
            },
            {"name": "immutable_inputs", "passed": True},
        ],
        "inputs": [
            {
                "path": str(source_path.resolve()),
                "sha256": file_sha256(source_path.resolve()),
            },
            {
                "path": str(inventory_path.resolve()),
                "sha256": file_sha256(inventory_path.resolve()),
            },
        ],
        "results": {
            "selection_rule": audit["selection_rule"],
            "requirements": audit["requirements"],
            "target_bus_range": audit["target_bus_range"],
            "candidate_group_count": audit["candidate_group_count"],
            "feasible_assignment_count_at_minimum_group_count": audit[
                "feasible_assignment_count_at_minimum_group_count"
            ],
            "outcomes_read": False,
        },
        "selected": audit["selected"],
        "block_reason": audit["block_reason"],
    }


def inventory(
    pglib_root: Path,
    source_path: Path,
    inventory_path: Path,
    r001_evidence_path: Path,
    r002_audit_path: Path,
) -> tuple[dict, dict, dict]:
    source = yaml.safe_load(source_path.read_text())
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ContractError("PGLib source lock has the wrong schema")
    if _git(pglib_root, "status", "--short"):
        raise ContractError("PGLib source checkout must be clean")
    observed_commit = _git(pglib_root, "rev-parse", "HEAD")
    if observed_commit != source["commit"]:
        raise ContractError(
            f"PGLib commit mismatch: {observed_commit} != {source['commit']}",
        )
    observed_remote = _git(pglib_root, "remote", "get-url", "origin")
    if observed_remote.rstrip("/") not in {
        str(source["remote"]).rstrip("/"),
        "git@github.com:power-grid-lib/pglib-opf.git",
    }:
        raise ContractError(f"unexpected PGLib origin {observed_remote}")
    paths = sorted(pglib_root.glob(str(source["case_glob"])))
    if not paths:
        raise ContractError("PGLib source checkout contains no cases")
    cases = sorted(
        (parse_case(path) for path in paths),
        key=lambda case: (case["bus_count"], case["network"]),
    )
    if len({case["network"] for case in cases}) != len(cases):
        raise ContractError("PGLib inventory contains duplicate case names")
    group_counts = dict(
        sorted(Counter(case["provenance_group"] for case in cases).items()),
    )
    inventory_payload = {
        "schema_version": INVENTORY_SCHEMA,
        "source": {
            "remote": source["remote"],
            "commit": observed_commit,
            "grouping_rule": source["grouping_rule"],
        },
        "case_count": len(cases),
        "eligible_case_count": sum(
            case["integrity_status"] == "PASS" for case in cases
        ),
        "provenance_group_counts": group_counts,
        "bus_count_range": [cases[0]["bus_count"], cases[-1]["bus_count"]],
        "cases": cases,
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(inventory_payload, indent=2, sort_keys=True) + "\n",
    )
    r001_payload = {
        "schema_version": "fm-scaling-evidence-v1",
        "gate_id": "R001",
        "status": "PASS",
        "checks": [
            {"name": "r001_criteria", "passed": True},
            {"name": "immutable_inputs", "passed": True},
        ],
        "inputs": [
            {
                "path": str(source_path.resolve()),
                "sha256": file_sha256(source_path.resolve()),
            },
            {
                "path": str(inventory_path.resolve()),
                "sha256": file_sha256(inventory_path.resolve()),
            },
        ],
        "results": {
            "source_commit": observed_commit,
            "case_count": len(cases),
            "eligible_case_count": inventory_payload["eligible_case_count"],
            "integrity_warning_count": (
                len(cases) - inventory_payload["eligible_case_count"]
            ),
            "provenance_group_count": len(group_counts),
            "bus_count_range": inventory_payload["bus_count_range"],
            "outcomes_read": False,
        },
    }
    r001_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    r001_evidence_path.write_text(
        json.dumps(r001_payload, indent=2, sort_keys=True) + "\n",
    )
    r002_payload = typed_split_evidence(
        audit_split(cases, source),
        source_path,
        inventory_path,
    )
    r002_audit_path.parent.mkdir(parents=True, exist_ok=True)
    r002_audit_path.write_text(
        json.dumps(r002_payload, indent=2, sort_keys=True) + "\n",
    )
    return inventory_payload, r001_payload, r002_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pglib-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--r001-evidence", type=Path, required=True)
    parser.add_argument("--r002-audit", type=Path, required=True)
    args = parser.parse_args()
    _, _, r002 = inventory(
        args.pglib_root.resolve(),
        args.source.resolve(),
        args.inventory.resolve(),
        args.r001_evidence.resolve(),
        args.r002_audit.resolve(),
    )
    print(json.dumps({"R001": "PASS", "R002": r002["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
