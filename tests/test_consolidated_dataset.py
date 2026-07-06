"""Equivalence test: HeteroGridDatasetMMap must return exactly the same
samples as the per-file HeteroGridDatasetDisk for the same raw data."""

import torch
import yaml

from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.powergrid_hetero_dataset import (
    HeteroGridDatasetDisk,
    HeteroGridDatasetMMap,
)
from gridfm_graphkit.io.param_handler import NestedNamespace

_ROOT = "tests/data/case14_ieee"
_STATS_PATH = f"{_ROOT}/processed/data_stats_HeteroDataMVANormalizer.pt"
_CONFIG_PATH = "tests/config/datamodule_test_base_config.yaml"


def _fitted_normalizer():
    with open(_CONFIG_PATH) as f:
        args = NestedNamespace(**yaml.safe_load(f))
    normalizer = HeteroDataMVANormalizer(args)
    normalizer.fit_from_dict(torch.load(_STATS_PATH, weights_only=True))
    return normalizer


def test_consolidated_matches_per_file_dataset():
    disk = HeteroGridDatasetDisk(root=_ROOT, data_normalizer=_fitted_normalizer())
    mmap = HeteroGridDatasetMMap(root=_ROOT, data_normalizer=_fitted_normalizer())

    assert len(mmap) == len(disk) > 0

    for idx in range(len(disk)):
        dict_disk = disk[idx].to_dict()
        dict_mmap = mmap[idx].to_dict()
        assert set(dict_disk) == set(dict_mmap)
        for group, attrs in dict_disk.items():
            assert set(attrs) == set(dict_mmap[group]), group
            for attr, expected in attrs.items():
                actual = dict_mmap[group][attr]
                assert expected.dtype == actual.dtype, (group, attr)
                assert expected.shape == actual.shape, (group, attr)
                assert torch.equal(expected, actual), (group, attr)


def test_consolidated_out_of_range():
    mmap = HeteroGridDatasetMMap(root=_ROOT, data_normalizer=_fitted_normalizer())
    try:
        mmap[len(mmap)]
    except IndexError:
        return
    raise AssertionError("expected IndexError past the last scenario")
