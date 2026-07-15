# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

from experiments.fm_scaling.inventory_pglib import (
    audit_split,
    parse_case,
    provenance_group,
)


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
    cases = [_case(f"source-{index}", index + 1, "source") for index in range(32)]
    cases += [_case(f"target-{index}", 500 + index, "target") for index in range(12)]
    source = {
        "target_bus_range": {"minimum": 500, "maximum": 13659},
        "requirements": {
            "source_count": 32,
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
    cases = [_case(f"source-{index}", index + 1, "source") for index in range(32)]
    for group in range(6):
        cases.extend(
            _case(f"target-{group}-{index}", 500 + group * 100 + index, f"g{group}")
            for index in range(2)
        )
    source = {
        "target_bus_range": {"minimum": 500, "maximum": 13659},
        "requirements": {
            "source_count": 32,
            "target_count": 12,
            "target_group_count": 6,
            "extrapolative_target_count": 4,
            "extrapolative_target_group_count": 2,
        },
    }
    result = audit_split(cases, source)
    assert result["status"] == "PASS"
    assert result["selected"] is not None
    assert len(result["selected"]["target_groups"]) == 6
