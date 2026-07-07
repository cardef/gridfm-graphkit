# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Size-balanced same-grid batch sampler (engineering plan E4 / tracker E005).

Multi-grid training concatenates per-grid datasets (``ConcatDataset``) and
the default shuffled loader mixes grids inside a batch. That has two costs:

- dynamic shapes: every batch has a different node/edge count, so
  ``torch.compile`` recompiles endlessly (the single biggest
  compile-friendliness lever is one static shape per grid);
- imbalance: grids with more scenarios dominate the epoch, starving the
  others (the experiment plan requires a size-balanced mixture for
  R010/R032).

:class:`SizeBalancedSameGridBatchSampler` fixes both: every batch contains
``batch_size`` samples from exactly one grid (last partial batch per grid is
dropped, so each grid contributes one static shape), and every grid
contributes exactly ``samples_per_grid`` samples per epoch (default: the
largest per-grid split; smaller grids are oversampled by cycling reshuffled
permutations, so no epoch-level duplicate appears before every sample of
that grid has been seen).

Train-time only: validation/test loaders keep the default per-dataset
sequential batching (no oversampling — balanced sampling would distort
metrics).

DDP: when ``torch.distributed`` is initialized the shuffled batch list is
sharded round-robin across ranks and trimmed to equal length. Lightning must
then NOT inject its own sampler — the CLI sets
``Trainer(use_distributed_sampler=False)`` when ``data.same_grid_batches``
is enabled. Lightning calls :meth:`set_epoch` each epoch; plain PyTorch
loops get a fresh shuffle per ``__iter__`` anyway via an internal counter.
"""

from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class SizeBalancedSameGridBatchSampler(Sampler[List[int]]):
    """Batch sampler over a ``ConcatDataset`` of per-grid datasets.

    Args:
        dataset_sizes: length of each per-grid dataset, in ``ConcatDataset``
            order (e.g. ``[len(d) for d in datamodule.train_datasets]``).
        batch_size: samples per batch; every yielded batch has exactly this
            many indices, all from one grid.
        samples_per_grid: per-epoch sample count drawn from every grid.
            Defaults to ``max(dataset_sizes)``. Must be >= batch_size.
        seed: base seed; the effective shuffle seed is ``(seed, epoch)``.
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        batch_size: int,
        samples_per_grid: Optional[int] = None,
        seed: int = 0,
    ):
        if not dataset_sizes or min(dataset_sizes) < 1:
            raise ValueError(f"Empty grid dataset in {dataset_sizes}")
        self.dataset_sizes = list(dataset_sizes)
        self.batch_size = int(batch_size)
        self.samples_per_grid = int(samples_per_grid or max(dataset_sizes))
        if self.samples_per_grid < self.batch_size:
            raise ValueError(
                f"samples_per_grid={self.samples_per_grid} < "
                f"batch_size={self.batch_size}: every grid would yield 0 batches",
            )
        self.seed = int(seed)
        self._epoch = 0
        # offsets into the ConcatDataset
        self.offsets = [0]
        for n in self.dataset_sizes:
            self.offsets.append(self.offsets[-1] + n)
        self.batches_per_grid = self.samples_per_grid // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Set the shuffle epoch (called by Lightning / DDP training loops)."""
        self._epoch = int(epoch)

    @staticmethod
    def _world():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _grid_indices(self, gi: int, g: torch.Generator) -> List[int]:
        """samples_per_grid global indices for grid gi, oversampled by
        cycling reshuffled permutations (no duplicate before full coverage)."""
        n, off = self.dataset_sizes[gi], self.offsets[gi]
        idx: List[int] = []
        while len(idx) < self.samples_per_grid:
            idx.extend((torch.randperm(n, generator=g) + off).tolist())
        return idx[: self.samples_per_grid]

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self._epoch)
        batches = []
        for gi in range(len(self.dataset_sizes)):
            idx = self._grid_indices(gi, g)
            for b in range(self.batches_per_grid):  # drop_last per grid
                batches.append(idx[b * self.batch_size : (b + 1) * self.batch_size])
        order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in order]

        rank, world = self._world()
        if world > 1:
            per_rank = len(batches) // world
            batches = batches[rank::world][:per_rank]

        self._epoch += 1  # plain-PyTorch loops reshuffle without set_epoch
        return iter(batches)

    def __len__(self) -> int:
        _, world = self._world()
        total = len(self.dataset_sizes) * self.batches_per_grid
        return total // world if world > 1 else total
