# Repository Instructions for Codex

These instructions apply to the whole repository. Treat the current tree and
the canonical files named below as authoritative; dated experiment reports and
agent memories are evidence snapshots, not current state by default.

## Environment

The shared virtual environment is in the parent directory:
`/Users/carmine/Code/FM/.venv`. It is Python 3.12, managed with `uv`, and has no
`pip` executable. Activate it with:

```bash
source ../.venv/bin/activate
```

Alternatively, invoke `../.venv/bin/python` directly. Install packages with
`uv pip`, for example:

```bash
uv pip install --python ../.venv/bin/python -e ".[dev,test]"
```

The environment is shared with `../gridfm-datakit`. This repository pins a
PyPI version of `gridfm-datakit`, while development normally uses an editable
checkout. Installing this repository with its dependencies can replace that
editable checkout with the pinned package. Before work that crosses the two
repositories, verify the active source instead of assuming it:

```bash
../.venv/bin/python -c \
  'import gridfm_datakit, pathlib; print(pathlib.Path(gridfm_datakit.__file__).resolve())'
```

Restore the intended sibling checkout when appropriate:

```bash
uv pip install --python ../.venv/bin/python -e ../gridfm-datakit
```

`torch-scatter` and `torch-sparse` are not graphkit dependencies. Native
PyTorch implementations live in `gridfm_graphkit/utils/scatter.py`, with
parity coverage in `tests/test_native_scatter.py`. External plugins may still
carry those packages for their own code.

## Verification Commands

Run commands from the repository root, normally in the shared environment:

```bash
MLFLOW_ALLOW_FILE_STORE=true pytest tests/
MLFLOW_ALLOW_FILE_STORE=true pytest tests/test_losses.py::test_pbe_loss_zero_with_real_data
MLFLOW_ALLOW_FILE_STORE=true pytest integrationtests/test_base_set.py
pre-commit run --all-files
bandit --severity-level high .
```

Integration tests are slow and may require cluster GPUs. CI runs unit tests
with coverage and `MLFLOW_ALLOW_FILE_STORE=true`, plus security checks. Ruff
and flake8 ignore `E501`.

The session fixture in `tests/conftest.py` creates processed case14 data under
`tests/data/case14_ieee/processed/` when needed. Remove that generated
directory only when a clean regeneration is required.

## CLI and Architecture

The console script enters through `gridfm_graphkit/__main__.py`; execution and
configuration handling continue in `gridfm_graphkit/cli.py`:

```bash
gridfm_graphkit {train|finetune|evaluate|predict|benchmark} \
  --config path/to/config.yaml [--data_path data]
```

Example configurations are under `examples/config/`; test configurations are
under `tests/config/`.

The codebase is YAML-configured and registry-driven:

1. The CLI parses a YAML configuration into `NestedNamespace` and applies CLI
   overrides.
2. Factories in `gridfm_graphkit/io/param_handler.py` resolve configured names
   through the registries in `gridfm_graphkit/io/registries.py`.
3. Lightning runs the selected task with MLflow logging. Model and normalizer
   artifacts are written below `mlruns/<exp_id>/<run_id>/artifacts/`.

The principal registries are `MODELS_REGISTRY`, `TASK_REGISTRY`,
`LOSS_REGISTRY`, `NORMALIZERS_REGISTRY`, `TRANSFORM_REGISTRY`,
`PHYSICS_DECODER_REGISTRY`, and `DATASET_WRAPPER_REGISTRY`. Registration is
decorator-based. When adding a task, keep its task, data-transform, and physics
decoder registrations under the same configured name.

Main areas:

- `tasks/`: Lightning task definitions and AC/DC diagnostic baselines.
- `models/`: flat GNS, GRIT, and the current hierarchical GNS prototype.
- `datasets/`: parquet ingestion, PyG processing, task transforms,
  normalization, cached positional encodings, and the optional consolidated
  mmap dataset selected by `data.consolidated: true`.
- `training/`: registered losses and callbacks.

`HeteroDataMVANormalizer` fits the training split and can be restored with
`--normalizer_stats`; `HeteroDataPerSampleMVANormalizer` always normalizes each
sample independently.

External plugins register additional components when imported through
`--plugins`; do not assume plugin-only classes are implemented in this tree.

## Active Kron-Schur Research Contract

The current sources of truth are:

- `refine-logs/FINAL_PROPOSAL.md` for the method and claim boundary.
- `refine-logs/EXPERIMENT_PLAN.md` for constants, estimands, budgets, and run
  rules.
- `refine-logs/EXPERIMENT_TRACKER.md` for implementation and experiment gates.
- `refine-logs/NOVELTY_REPORT.md` for the source-backed novelty boundary.
- `MANIFEST.md` for generated research artifacts.

Proposal readiness is not implementation readiness. The tracker currently
blocks treatment runs until implementation items I001-I010 and freeze gate
R014 pass; R014 also requires the calibration, profiling, and smoke gates
listed in the tracker. It is the sole authorization for E001-E020. Existing
M0/M1 code, caches, configurations, and results are legacy prototype evidence
only.

Preserve these load-bearing invariants in research implementation work:

- The implementation target is explicitly the `cardef/gridfm-graphkit`
  research fork of upstream `gridfm/gridfm-graphkit`. Record the fork commit,
  upstream reference, merge base, environment, and worktree state for every
  confirmatory artifact.
- Communication geometry is the sole treatment. All headline arms share the
  encoder, exactly `L_pre` local blocks, one communication-core call, exactly
  `L_post` local blocks, direct PF decoder, known-value projector, objective,
  optimizer family, data policy, and checkpoint schedule.
- The hierarchy is latent communication only. The confirmatory entry point has
  no `v_aff`, `cbus_x`, boundary-voltage tensor, affine physical unpool, HELM
  feature, or exact physical-reconstruction claim.
- Latent geometry and sparsity are pure functions of topology plus one frozen
  source-only policy. Never use scenario labels, solver bus classes, target
  results, or per-grid overrides to choose a partition or operator.
- Kron and Quotient share the partition, adapter, channel schema, coarse-node
  count, and sparsity cap. Do not pad Quotient to imitate Schur fill or claim
  equal realized edge counts; report both supports and charge their actual
  training and inference FLOPs.
- Use case-declared `baseMVA`. Optional standardization is fitted on source
  training data once and frozen. Per-grid and per-sample fitted normalizers are
  forbidden in the confirmatory path.
- Target outputs must be unreadable to normalization, partitioning, geometry,
  sparsity, checkpoint selection, and hyperparameter code. Known quantities
  are projected before PF error and physical-residual evaluation.
- The final objective has no coarse-voltage, reconstruction, or
  hierarchy-specific auxiliary unless the canonical proposal and plan are
  explicitly revised.
- Start the subprocess `MetaPathFinder` denial of the legacy hierarchy,
  `GNS_hetero_hier`, HELM reconstruction, and fitted per-grid normalizers at
  the first I007/F2 communication-seam commit and keep it in the test suite.
- Confirmatory jobs fail closed on missing manifests, hashes, budgets, or gate
  evidence. Failures remain in the denominator. Do not launch from the legacy
  M1 configuration glob or inspect target efficacy to alter the fixed matrix.

The current implementation in `datasets/hierarchy.py`,
`models/gnn_hetero_hier.py`, and the fitted normalizers does not satisfy this
contract. It couples the legacy REF/PV hierarchy, sparsified physical
reconstruction, scenario-dependent choices, and HELM/affine heads that the
confirmatory method excludes. Treat those paths as historical evidence; do not
import or describe them as the final method.

## Contributing

Follow PEP 8, retain project license headers on new files, and use DCO sign-off
for commits (`git commit -s`). See `CONTRIBUTING.md`.

## ARIS Skills

`.aris/installed-skills.txt` records 80 project-local Claude skill links under
`.claude/skills/`. Those entries are symlinks into
`/Users/carmine/Auto-claude-code-research-in-sleep`; never edit or delete files
through the symlinks. Codex has corresponding Codex-native skills installed
separately, so use the skills exposed in the current Codex session.

To install or reconcile project-local Codex skills explicitly, use the Codex
platform rather than relying on auto-detection in this mixed Claude/Codex repo:

```bash
bash /Users/carmine/Auto-claude-code-research-in-sleep/tools/install_aris.sh \
  "$PWD" --platform codex
```
