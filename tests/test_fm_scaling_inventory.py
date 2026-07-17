# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.fm_scaling.inventory_pglib import (
    audit_split,
    parse_case,
    provenance_group,
    typed_split_evidence,
)
from gridfm_graphkit.fm_scaling.manifest import validate_gate_evidence


def _case(name, buses, group):
    return {
        "network": name,
        "topology_key": name,
        "provenance_group": group,
        "bus_count": buses,
        "encoded_bus_count": buses,
        "baseMVA": 100.0,
        "integrity_status": "PASS",
        "raw_sha256": "a" * 64,
    }


def test_parse_case_and_conservative_provenance_group(tmp_path):
    path = tmp_path / "pglib_opf_case3_ieee_rts.m"
    path.write_text(
        "mpc.baseMVA = 100.0;\nmpc.bus = [\n1 1;\n2 1;\n3 1;\n];\n",
    )
    parsed = parse_case(path)
    assert parsed["bus_count"] == 3
    assert parsed["baseMVA"] == 100.0
    assert parsed["integrity_status"] == "PASS"
    assert parsed["provenance_group"] == "ieee"
    assert provenance_group("case2746wp_k") == "k"


def test_split_audit_blocks_when_whole_groups_are_infeasible():
    cases = [_case(f"source-{index}", index + 1, "source") for index in range(28)]
    cases += [_case(f"target-{index}", 500 + index, "target") for index in range(12)]
    source = {
        "target_bus_range": {"minimum": 500, "maximum": 13659},
        "requirements": {
            "source_count": 26,
            "source_levels": [8, 16, 26],
            "source_dev_group_count": 2,
            "samples_total": 11655,
            "training_batch_size": 1,
            "minimum_endpoint_batches_per_case": 128,
            "target_count": 12,
            "target_group_count": 6,
            "extrapolative_target_count": 4,
            "extrapolative_target_group_count": 2,
        },
    }
    result = audit_split(cases, source)
    assert result["status"] == "BLOCKED"
    assert result["selected"] is None


def test_split_audit_finds_a_valid_whole_group_assignment():
    cases = [_case(f"source-{index}", index + 1, f"s{index % 7}") for index in range(26)]
    cases += [
        _case("source-dev-0", 27, "source-dev-0"),
        _case("source-dev-1", 28, "source-dev-1"),
    ]
    for group in range(6):
        cases.extend(
            _case(f"target-{group}-{index}", 500 + group * 100 + index, f"g{group}")
            for index in range(2)
        )
    source = {
        "target_bus_range": {"minimum": 500, "maximum": 13659},
        "requirements": {
            "source_count": 26,
            "source_levels": [8, 16, 26],
            "source_dev_group_count": 2,
            "samples_total": 11655,
            "training_batch_size": 1,
            "minimum_endpoint_batches_per_case": 128,
            "target_count": 12,
            "target_group_count": 6,
            "extrapolative_target_count": 4,
            "extrapolative_target_group_count": 2,
        },
    }
    result = audit_split(cases, source)
    assert result["status"] == "PASS"
    assert result["selected"] is not None
    assert result["selected"]["source_dev_groups"] == ["source-dev-0", "source-dev-1"]
    assert result["selected"]["training_balance"]["G26"] == {
        "provenance_group_count": 7,
        "provenance_group_case_counts": {
            "s0": 4,
            "s1": 4,
            "s2": 4,
            "s3": 4,
            "s4": 4,
            "s5": 3,
            "s6": 3,
        },
        "batches_per_group": 1665,
        "minimum_batches_per_case": 416,
    }
    assert len(result["selected"]["target_groups"]) == 6


def test_split_audit_can_be_recorded_as_typed_gate_evidence(tmp_path):
    cases = [_case(f"source-{index}", index + 1, f"s{index % 7}") for index in range(26)]
    cases += [
        _case("source-dev-0", 27, "source-dev-0"),
        _case("source-dev-1", 28, "source-dev-1"),
    ]
    for group in range(6):
        cases.extend(
            _case(f"target-{group}-{index}", 500 + group * 100 + index, f"g{group}")
            for index in range(2)
        )
    source = {
        "target_bus_range": {"minimum": 500, "maximum": 13659},
        "requirements": {
            "source_count": 26,
            "source_levels": [8, 16, 26],
            "source_dev_group_count": 2,
            "samples_total": 11655,
            "training_batch_size": 1,
            "minimum_endpoint_batches_per_case": 128,
            "target_count": 12,
            "target_group_count": 6,
            "extrapolative_target_count": 4,
            "extrapolative_target_group_count": 2,
        },
    }
    source_path = tmp_path / "source.yaml"
    inventory_path = tmp_path / "inventory.json"
    source_path.write_text("source: frozen\n")
    inventory_path.write_text("{}\n")
    evidence_path = tmp_path / "R002.json"
    evidence_path.write_text(
        json.dumps(
            typed_split_evidence(
                audit_split(cases, source),
                source_path,
                inventory_path,
            ),
        ),
    )

    evidence = validate_gate_evidence(evidence_path, "R002")
    assert evidence["selected"]["source_max_bus_count"] == 26
