# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

from experiments.fm_scaling.check_cpu_compatibility import mlflow_smoke_passed
from experiments.fm_scaling.check_gpu_compatibility import resolve_compile_policy


def test_mlflow_smoke_requires_the_named_pass_check():
    assert mlflow_smoke_passed(
        {
            "checks": [
                {"name": "mlflow_store_create_search_smoke", "passed": True},
            ],
        },
    )
    assert not mlflow_smoke_passed(
        {
            "checks": [
                {"name": "mlflow_store_create_search_smoke", "passed": False},
            ],
        },
    )


def test_compile_policy_enables_only_after_output_gradient_and_flop_parity():
    enabled = resolve_compile_policy({"passed": True}, 100, 101)
    assert enabled["selected_mode"] == "default"
    assert enabled["passed"] is True

    bad_gradient = resolve_compile_policy({"passed": False}, 100, 100)
    assert bad_gradient["selected_mode"] == "disabled"
    assert bad_gradient["passed"] is True

    bad_flops = resolve_compile_policy({"passed": True}, 100, 103)
    assert bad_flops["selected_mode"] == "disabled"
    assert bad_flops["passed"] is True

    unavailable = resolve_compile_policy({"passed": True}, 0, 0)
    assert unavailable["selected_mode"] == "disabled"
    assert unavailable["relative_flop_gap"] is None
    assert unavailable["passed"] is True
