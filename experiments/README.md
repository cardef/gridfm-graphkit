# Experiment ownership

Git owns experiment definitions in this directory: runner code, canonical
configuration, SLURM launchers, and reproducibility helpers. Run outputs do
not belong here.

New training jobs must write their MLflow data, logs, checkpoints, and other
mutable artifacts below the repository-level `mlruns/` Syncthing root. The
SLURM launchers use `MLRUNS_ROOT` from `slurm/cluster.env`; their scheduler
output goes to `mlruns/slurm-logs/`.

Legacy scripts that still emit compact JSON summaries honor
`GRIDFM_RESULTS_ROOT`. The cluster environment sets it to
`mlruns/result-summaries/`; their historical local fallback remains the
existing M0/M1 `results/` directory.

Before launching a durable run:

1. commit its code and canonical configuration on a research or experiment
   branch;
2. record the Git SHA and configuration hash in the run manifest;
3. tag the executed revision as `run/<id>`;
4. keep checkpoints, raw logs, metrics, and expanded per-run configuration in
   `mlruns/<experiment-id>/<run-id>/`.

The existing M0/M1 `results/` directories are retained only as historical
prototype evidence. Do not add new result snapshots there. The migrated M1
R010 payload is described by
`research/kron-schur/runs/legacy-m1-r010-20260715.json`.
