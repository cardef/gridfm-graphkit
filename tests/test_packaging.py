# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages


def test_runtime_utils_are_in_distribution_package_discovery():
    packages = find_packages(include=["gridfm_graphkit*"])
    assert "gridfm_graphkit.utils" in packages
