# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

from experiments.fm_scaling.run_implementation_gate import GATE_SPECS
from gridfm_graphkit.fm_scaling.manifest import GATE_EVIDENCE_KIND


def test_implementation_gate_specs_are_complete_and_typed():
    expected = {f"I{index:03d}" for index in range(2, 10)}
    assert set(GATE_SPECS) == expected
    for gate_id, spec in GATE_SPECS.items():
        assert GATE_EVIDENCE_KIND[gate_id].endswith("-tests")
        assert spec.tests
        assert spec.inputs
        assert all(test.startswith("tests/") for test in spec.tests)
        assert all(path.endswith(".py") for path in spec.inputs)
