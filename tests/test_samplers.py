# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the E005 size-balanced same-grid batch sampler."""

import copy
from collections import Counter

import pytest
import yaml

from gridfm_graphkit.datasets.samplers import (
    ProvenanceBalancedSameGridBatchSampler,
    SizeBalancedSameGridBatchSampler,
)

# mimics the M0 local mixture shape: unequal per-grid train splits
SIZES = [100, 40, 7]
BS = 8


def _batches(sampler):
    return list(iter(sampler))


def _grid_of(idx, offsets):
    for gi in range(len(offsets) - 1):
        if offsets[gi] <= idx < offsets[gi + 1]:
            return gi
    raise AssertionError(f"index {idx} outside dataset")


def test_same_grid_property():
    s = SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=0)
    for batch in _batches(s):
        assert len(batch) == BS  # static shape: no partial batches
        grids = {_grid_of(i, s.offsets) for i in batch}
        assert len(grids) == 1, f"mixed-grid batch: {batch}"


def test_size_balance_and_len():
    s = SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=0)
    batches = _batches(s)
    assert s.samples_per_grid == max(SIZES)
    per_grid = max(SIZES) // BS
    assert len(batches) == len(s) == len(SIZES) * per_grid
    counts = Counter(_grid_of(b[0], s.offsets) for b in batches)
    assert all(counts[gi] == per_grid for gi in range(len(SIZES)))


def test_oversampling_cycles_full_permutations():
    # a grid smaller than samples_per_grid must be fully covered before any
    # index repeats: occurrence counts differ by at most 1
    s = SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=0)
    small = 2  # size 7, oversampled to 96 drawn samples (12 batches x 8)
    idx = [i for b in _batches(s) for i in b if _grid_of(i, s.offsets) == small]
    counts = Counter(idx)
    assert set(counts) == set(range(s.offsets[small], s.offsets[small + 1]))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_no_duplicates_within_large_grid_epoch():
    # a grid with >= samples_per_grid samples is drawn without replacement
    s = SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=0)
    idx = [i for b in _batches(s) for i in b if _grid_of(i, s.offsets) == 0]
    assert len(idx) == len(set(idx))


def test_determinism_and_epoch_reshuffle():
    a = SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=1)
    a.set_epoch(3)
    e3 = _batches(a)  # bumps internal epoch to 4
    e4 = _batches(a)
    assert e3 != e4  # epochs reshuffle even without set_epoch calls

    b = SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=1)
    b.set_epoch(3)
    assert _batches(b) == e3  # same (seed, epoch) -> same batches

    c = SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=2)
    c.set_epoch(3)
    assert _batches(c) != e3  # different seed -> different batches


def test_samples_per_grid_override():
    s = SizeBalancedSameGridBatchSampler(
        SIZES,
        batch_size=BS,
        samples_per_grid=16,
        seed=0,
    )
    batches = _batches(s)
    assert len(batches) == len(SIZES) * 2  # 16 // 8 per grid


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        SizeBalancedSameGridBatchSampler([10, 0], batch_size=4)
    with pytest.raises(ValueError):
        SizeBalancedSameGridBatchSampler([10], batch_size=8, samples_per_grid=4)


def test_ddp_sharding_matches_strided_split(monkeypatch):
    # unsharded reference at the same (seed, epoch)
    full = _batches(SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=0))
    world = 3
    per_rank = len(full) // world
    for rank in range(world):
        monkeypatch.setattr(
            SizeBalancedSameGridBatchSampler,
            "_world",
            staticmethod(lambda r=rank: (r, world)),
        )
        s = SizeBalancedSameGridBatchSampler(SIZES, batch_size=BS, seed=0)
        assert len(s) == per_rank
        # disjoint + equal-length by construction of the strided split
        assert _batches(s) == full[rank::world][:per_rank]


def test_provenance_sampler_equalizes_groups_then_cases():
    sampler = ProvenanceBalancedSameGridBatchSampler(
        dataset_sizes=[20, 20, 20],
        provenance_groups=["group-a", "group-a", "group-b"],
        batch_size=2,
        samples_total=24,
        seed=3,
    )
    batches = _batches(sampler)
    case_counts = Counter(_grid_of(batch[0], sampler.offsets) for batch in batches)
    group_counts = Counter(
        sampler.provenance_groups[_grid_of(batch[0], sampler.offsets)]
        for batch in batches
    )
    assert len(batches) == len(sampler) == 12
    assert group_counts == {"group-a": 6, "group-b": 6}
    assert abs(case_counts[0] - case_counts[1]) <= 1
    assert all(
        len({_grid_of(index, sampler.offsets) for index in batch}) == 1
        for batch in batches
    )


def test_provenance_sampler_requires_exact_total():
    with pytest.raises(ValueError, match="divisible"):
        ProvenanceBalancedSameGridBatchSampler(
            dataset_sizes=[10, 10],
            provenance_groups=["a", "b"],
            batch_size=4,
            samples_total=10,
        )


def test_datamodule_integration(generate_processed_test_data):
    # end-to-end: config flag routes train_dataloader through the sampler
    from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (
        LitGridHeteroDataModule,
    )
    from gridfm_graphkit.io.param_handler import NestedNamespace

    with open("tests/config/datamodule_test_base_config.yaml") as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    cfg["data"]["same_grid_batches"] = True
    cfg["data"]["workers"] = 0

    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")

    class DummyTrainer:
        is_global_zero = True
        logger = None

    dm.trainer = DummyTrainer()
    dm.setup("fit")
    loader = dm.train_dataloader()
    assert isinstance(loader.batch_sampler, SizeBalancedSameGridBatchSampler)
    n_batches = 0
    for batch in loader:
        assert batch.num_graphs == cfg["training"]["batch_size"]
        n_batches += 1
    assert n_batches == len(loader.batch_sampler)
