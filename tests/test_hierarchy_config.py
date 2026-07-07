# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Per-network hierarchy config resolution (boundary-fraction knob)."""

from gridfm_graphkit.datasets.hierarchy import AddHierarchy, hierarchy_cache_name
from gridfm_graphkit.io.param_handler import NestedNamespace


def _args(hierarchy):
    return NestedNamespace(**{"data": {"hierarchy": hierarchy}})


def test_defaults_without_per_network():
    t = AddHierarchy(_args({"enable": True}))
    t.set_root("/data/case14_ieee")
    assert t.target_frac == 0.27
    assert t.tol == 1e-3


def test_per_network_override_and_fallback():
    t = AddHierarchy(
        _args(
            {
                "enable": True,
                "target_frac": 0.30,
                "per_network": {
                    "case118_ieee": {"target_frac": 0.46},
                    "case2000_goc": {"tol": 1e-4},
                },
            },
        ),
    )
    t.set_root("/data/case118_ieee/")  # trailing slash must not matter
    assert t.target_frac == 0.46
    assert t.tol == 1e-3  # override sets only target_frac; tol falls back

    t.set_root("/data/case2000_goc")
    assert t.target_frac == 0.30  # global default from config
    assert t.tol == 1e-4

    t.set_root("/data/case14_ieee")  # no override entry -> defaults
    assert t.target_frac == 0.30
    assert t.tol == 1e-3


def test_cache_name_tracks_effective_values():
    assert hierarchy_cache_name(0.46, 1e-3) == "hierarchy_b0.46_tol0.001.pt"
    assert hierarchy_cache_name(0.27, 1e-3) != hierarchy_cache_name(0.46, 1e-3)
