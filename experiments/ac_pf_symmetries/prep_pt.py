"""Piano X, X3 data: renumber the downloaded partitions of the public pretraining subgrids
(data/genco_pretrain/<grid>/raw) into data/<grid>/raw with contiguous scenario ids (graphkit's preprocessing requires
them), write the training list cluster/pt_train_networks.txt, and draw a fixed 2,048-scenario subset (seed 0) of the
case118 evaluation set of GENCO's transfer experiment (data/genco_transfer/data/case118_ieee/raw) into data/case118_tr.

python -m experiments.ac_pf_symmetries.prep_pt
"""

from pathlib import Path

import numpy as np

from .run import ROOT
from .subset_data import source_ids, write_subset


def main():
    data = ROOT / "data"
    grids = sorted(p.name for p in (data / "genco_pretrain").glob("subnet_*") if (p / "raw").is_dir())
    for g in grids:
        if (data / g / "raw").exists():
            print("exists", g, flush=True)
            continue
        src = data / "genco_pretrain" / g / "raw"
        ids = source_ids(src)
        write_subset(src, ids, data / g)
        print(g, len(ids), "scenarios", flush=True)
    lst = Path(__file__).resolve().parent / "cluster" / "pt_train_networks.txt"
    lst.write_text("\n".join(grids) + "\n")
    tr = data / "case118_tr"
    if not (tr / "raw").exists():
        src = data / "genco_transfer" / "data" / "case118_ieee" / "raw"
        pool = source_ids(src)
        write_subset(src, np.random.default_rng(0).choice(pool, size=2048, replace=False), tr)
        print("case118_tr: 2048 of", len(pool), flush=True)
    print("PREP_DONE", len(grids), "grids ->", lst, flush=True)


if __name__ == "__main__":
    main()
