# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

from experiments.fm_scaling.check_cpu_compatibility import mlflow_smoke_passed


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
