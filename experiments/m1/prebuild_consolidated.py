# Copyright contributors to the GridFM project
# SPDX-License-Identifier: Apache-2.0

"""Pre-build data/<grid>/processed/consolidated.pt once, single process,
before the R010 training array launches. HeteroGridDatasetMMap.process()
accumulates every scenario's tensors in RAM before the single write (see
its own docstring/comment) -- with 6 concurrent case2000_goc training
tasks each independently racing to build this missing cache, all 6 OOM'd
even at --mem=64G. Building it once here (generous memory, no GPU
contention) means every training task just mmaps the finished file.

data_normalizer is unused by process()/consolidation (only get() applies
it), so a no-op stands in -- same pattern as tests/test_degenerate_scenario_filter.py.
"""

import sys

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetMMap


class _NoOpNormalizer:
    def transform(self, data):
        return data


for grid in sys.argv[1:]:
    root = f"data/{grid}"
    print(f"=== building consolidated cache for {grid} ===", flush=True)
    ds = HeteroGridDatasetMMap(root=root, data_normalizer=_NoOpNormalizer())
    print(f"{grid}: {len(ds)} scenarios consolidated -> {root}/processed/consolidated.pt", flush=True)
