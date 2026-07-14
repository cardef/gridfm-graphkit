# Repository Instructions for Codex

These instructions apply to the whole repository. A fresh clone is the source
of truth for builds and tests; Syncthing folders and agent runtime state are
never deployment dependencies.

## Environment

Python 3.12 is the primary development version. A checkout may use either its
own virtual environment or the optional shared environment at `../.venv`:

```bash
source ../.venv/bin/activate
uv pip install --python ../.venv/bin/python -e ".[dev,test]"
```

This repository pins `gridfm-datakit`. When developing against a sibling
checkout, verify which package is imported before assuming the editable source
is active:

```bash
python -c 'import gridfm_datakit, pathlib; print(pathlib.Path(gridfm_datakit.__file__).resolve())'
```

`torch-scatter` and `torch-sparse` are not dependencies. Native PyTorch
implementations live in `gridfm_graphkit/utils/scatter.py` and are covered by
`tests/test_native_scatter.py`.

## Verification

Run from the repository root:

```bash
MLFLOW_ALLOW_FILE_STORE=true pytest tests/
pre-commit run --all-files
bandit --severity-level high .
python -m build
python tools/check_syncthing_boundary.py --require-local-config
```

Integration tests are slower and may require GPUs:

```bash
MLFLOW_ALLOW_FILE_STORE=true pytest integrationtests/test_base_set.py
```

## CLI and Architecture

The console entry point is `gridfm_graphkit/__main__.py`, with command handling
in `gridfm_graphkit/cli.py`:

```bash
gridfm_graphkit {train|finetune|evaluate|predict|benchmark} \
  --config path/to/config.yaml [--data_path data]
```

Configuration is YAML-based and registry-driven. Factories in
`gridfm_graphkit/io/param_handler.py` resolve names through registries in
`gridfm_graphkit/io/registries.py`. Keep task, transform, and physics-decoder
registrations aligned under the same configured name.

The main source areas are:

- `tasks/`: Lightning task definitions and diagnostic baselines.
- `models/`: registered GNS and GRIT implementations.
- `datasets/`: parquet/PyG ingestion, transforms, normalizers, positional
  encodings, and the optional consolidated mmap dataset.
- `training/`: registered losses and callbacks.

External plugins may register additional components through `--plugins`; do
not treat plugin-only classes as part of this package.

## Branch and Artifact Ownership

- `main` must remain buildable, tested, and deployable.
- `research/*` branches hold durable research methods and canonical plans.
- `exp/*` branches hold short-lived code hypotheses, not individual run data.
- Annotated `run/*` tags identify exact executed revisions.
- MLflow/Syncthing own ignored results, checkpoints, traces, and agent memory.

Code and configuration must use repository-relative paths or explicit
environment variables. Do not commit credentials, caches, sessions, generated
skill links, machine-specific absolute paths, or mutable run outputs.

## Contributing

Follow PEP 8, retain project license headers on new files, and use DCO sign-off
for commits (`git commit -s`). See `CONTRIBUTING.md`.
