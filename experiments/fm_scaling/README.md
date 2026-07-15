# Confirmatory FM-scaling experiments

This namespace implements the communication-only Kron--Schur research
contract. It must not import the legacy M0/M1 hierarchy, configurations, or
results.

Before reserving a GPU, run the M0 preflight from the repository root:

```bash
../.venv/bin/python -m experiments.fm_scaling.preflight \
  --expected-datakit-commit "$(git -C ../gridfm-datakit rev-parse HEAD)" \
  --output mlruns/fm-scaling/result-summaries/I001.json
```

The captured datakit SHA becomes the I001 pin; the check still fails unless
that commit is clean and reachable from `origin/main`.

The fixed mutable layout is:

- `mlruns/fm-scaling/mlflow-store/`: MLflow file store;
- `mlruns/fm-scaling/slurm-logs/`: scheduler logs;
- `mlruns/fm-scaling/result-summaries/`: compact gate and run records.

Repository-level `mlruns/` is a Syncthing root and contains `.stfolder`.
Passing it directly as `--log_dir` is forbidden: the legacy M1 launch did so
and forty jobs failed while MLflow tried to parse `.stfolder` as an experiment
ID. This M0 tool validates a candidate store. I010 must bind that same resolved
store from the frozen run manifest into the Lightning logger before any GPU
launch is authorized.

I001 evidence becomes valid only when the preflight is run from a clean graphkit
commit reachable from `origin/research/kron-schur`, with the exact clean
datakit worktree and commit named on the command line. Later campaign launchers
must also consume an explicit run manifest; wildcard YAML discovery is
forbidden.

## Implementation layout

- `gridfm_graphkit/fm_scaling/contracts.py`: immutable topology, partition,
  sparse-operator, geometry, budget, and provenance schemas;
- `partition.py`, `geometry.py`, `registry.py`: stable-ID contiguous METIS,
  Kron/Quotient construction, conservative transport, resource gates, and the
  content-addressed device registry;
- `communication.py`, `model.py`: the common fine backbone and the Flat,
  Global, Kron, and Quotient communication slots;
- `data.py`, `loss.py`, `task.py`: case-declared normalization, topology keys,
  per-graph/component objective, known-value projection, and ground-truth
  evaluation records;
- `accounting.py`, `analysis.py`: parameter matching, executed FLOP counting,
  first-crossing checkpoints, scenario/topology/group aggregation, and exact
  sign-flip inference;
- `select_geometry.py`, `freeze_calibration.py`: deterministic R003--R012
  reducers over source-only calibration and profile records;
- `build_geometry.py`, `make_campaign.py`, `launch.py`,
  `evaluate_campaign.py`: topology-only bundle construction, explicit hashed
  20-run manifests, fail-closed train-only launch, and matrix-sealed evaluation.
- `prepare_data.py`, `finalize_data.py`, `freeze_targets.py`, `make_splits.py`:
  exact sibling-fork data generation, static-admittance/completeness audit,
  whole-provenance-group target freeze, and per-network hashed splits;
- `analyze_campaign.py`: complete-scenario validation, locked aggregation,
  exact sign-flip inversion, and the conjunctive C1/C2 decision.

Start from `freeze.template.yaml`; it intentionally contains invalid zero/empty
placeholders. Only source-only calibration and PASS gate artifacts may replace
them. `make_campaign.py` writes exactly E001--E020 and never discovers configs
with a wildcard. Batches balance provenance groups, then cases, and contain one
topology each.

The launcher verifies the clean fork commit, upstream pin and merge base, every
I/R/C/P/S gate record and hash, every config hash, the topology and geometry
hashes, the child MLflow store smoke, and CUDA availability. It records a
machine-readable `BLOCKED` result on any failure.

`launch.py --execute` performs training only and seals a `TRAINED` record with
hashes for all first-crossing checkpoints, the FLOP ledger, and runtime
evidence. `evaluate_campaign.py` refuses to evaluate any target until all 20
training records are sealed and unchanged. It then evaluates one run at the
three locked checkpoints and seals distinct metric hashes. This phase split is
mandatory; editing a checkpoint, ledger, runtime record, metric file, or
campaign manifest invalidates the downstream phase.

## Data and freeze sequence

The existing legacy datasets are not admissible: their Y-bus/admittance can
vary by scenario. Prepare replacements only through the shared environment and
the exact sibling research fork:

```bash
../.venv/bin/python -m experiments.fm_scaling.prepare_data \
  --inventory experiments/fm_scaling/frozen/data_inventory.yaml \
  --config-dir experiments/fm_scaling/frozen/datakit-configs \
  --data-root data \
  --manifest experiments/fm_scaling/frozen/topology_manifest.draft.yaml
```

The default writes configs only; add `--execute` to run and validate datakit.
The command refuses a non-editable package, any checkout other than
`../gridfm-datakit`, a dirty worktree, a non-fork origin, a commit mismatch, or
an environment other than `../.venv`. All generated configs set topology,
admittance, and generation perturbations to `none`.

After generation, finalize hashes and static-topology evidence, freeze target
membership without reading outcomes, and materialize explicit scenario IDs:

```bash
../.venv/bin/python -m experiments.fm_scaling.finalize_data \
  --draft experiments/fm_scaling/frozen/topology_manifest.draft.yaml \
  --config-dir experiments/fm_scaling/frozen/datakit-configs \
  --data-root data \
  --output experiments/fm_scaling/frozen/topology_manifest.audited.yaml

../.venv/bin/python -m experiments.fm_scaling.freeze_targets \
  --manifest experiments/fm_scaling/frozen/topology_manifest.audited.yaml \
  --selection experiments/fm_scaling/frozen/target_freeze.yaml \
  --output experiments/fm_scaling/frozen/topology_manifest.yaml

../.venv/bin/python -m experiments.fm_scaling.make_splits \
  --spec experiments/fm_scaling/frozen/splits.yaml \
  --topology-manifest experiments/fm_scaling/frozen/topology_manifest.yaml \
  --output-root experiments/fm_scaling/frozen/splits \
  --manifest experiments/fm_scaling/frozen/split_manifest.yaml
```

These commands prepare evidence; they do not make R001-R014 pass. The frozen
templates intentionally contain invalid placeholders until source-only
decisions and machine evidence exist.
