# Artifact ownership and sync layout

The operational Mac/Linux branch workflow is documented in
[`GIT_SYNCTHING_WORKFLOW.md`](GIT_SYNCTHING_WORKFLOW.md).

A fresh clone must build, test, and run without Syncthing or agent tooling.
Git is the source of truth for every deployment input and repository contract.
Syncthing is optional replication for high-churn research artifacts that are
explicitly excluded from Git.

## Git-owned files

- Source, tests, packaging, CI, examples, and deployment configuration.
- Experiment scripts, canonical configurations, and SLURM launchers below
  `experiments/`.
- `AGENTS.md` and `CLAUDE.md`, because they are portable repository policy.
- `MANIFEST.md` and the promoted Kron-Schur research snapshot under
  `research/kron-schur/`.

No Git branch may track files below a Syncthing root. ARIS continues to use
`refine-logs/` as its working directory, including its expected canonical
filenames; those files are ignored and synchronized. Promote a reviewed
snapshot to Git with:

```bash
python tools/promote_kron_schur_contract.py
git diff -- research/kron-schur/
git add research/kron-schur/
git commit -s -m "Update Kron-Schur research contract"
```

The promotion is intentionally one-way: ARIS writes the working copy, and Git
records the reviewed snapshot. Do not edit the synchronized copy from Linux
while ARIS is running on the Mac.

## Syncthing-owned files

The saved Mac configuration may replicate these ignored directories:

| Folder | Repository path | Purpose |
| --- | --- | --- |
| `gridfm-idea-stage` | `idea-stage/` | Discovery and prototype notes |
| `gridfm-papers` | `papers/` | Paper working artifacts |
| `gridfm-refine-logs` | `refine-logs/` | Timestamped history and review records |
| `gridfm-research-wiki` | `research-wiki/` | Working research wiki |
| `gridfm-mlruns` | `mlruns/` | Experiment tracking artifacts |

Syncthing is replication, not version control. Do not use it to move tracked
files between Git worktrees.

## GPFS bridge

The current deployment has three distinct storage domains:

1. the Git checkout on GPFS, where HPC jobs execute;
2. the workstation artifact workspace managed by Syncthing;
3. the Mac Syncthing peer.

Syncthing does not manage the GPFS checkout directly. An explicit, one-way
`rsync` bridges one artifact root at a time. The direction depends on which
side produced the files:

| Content | Authoritative writer | Bridge direction |
| --- | --- | --- |
| `idea-stage/`, `refine-logs/`, `research-wiki/`, `papers/` | Mac/workstation ARIS workflow | workstation → GPFS |
| `mlruns/` | Linux/HPC execution | GPFS → workstation |
| `experiments/` | Git branch | no rsync; use Git pull/push |

Use environment variables for machine-specific roots; do not commit absolute
paths:

```bash
export GRIDFM_GPFS_REPO=/path/to/gpfs/gridfm-graphkit
export GRIDFM_SYNCTHING_REPO=/path/to/local/gridfm-graphkit
```

For an HPC-produced MLflow update, preview, copy, and verify in that order:

```bash
rsync -ain --itemize-changes \
  "$GRIDFM_GPFS_REPO/mlruns/" \
  "$GRIDFM_SYNCTHING_REPO/mlruns/"

rsync -a \
  "$GRIDFM_GPFS_REPO/mlruns/" \
  "$GRIDFM_SYNCTHING_REPO/mlruns/"

rsync -ainc \
  "$GRIDFM_GPFS_REPO/mlruns/" \
  "$GRIDFM_SYNCTHING_REPO/mlruns/"
```

Reverse the source and destination only for the four ARIS working roots in
the table. Never use `--delete` for this bridge. Do not run opposing rsync
directions concurrently, and do not edit the same logical artifact while a
copy or Syncthing transfer is active.

The repository-level `sync-layout.json` records these routes using relative
paths plus the `GRIDFM_GPFS_REPO` and `GRIDFM_SYNCTHING_REPO` environment
variables.

## Agent state

- Install Claude and Codex skills from the ARIS Git checkout; do not synchronize
  generated skill installations or project-local symlink farms.
- `~/.codex/memories/` may be synchronized only with a single-writer policy.
  Concurrent writers can create conflict copies and corrupt the routing index.
- Never synchronize an entire `~/.codex`, `~/.claude`, `.codex`, `.claude`, or
  `.aris` tree. Exclude credentials, API-key-bearing configuration, databases
  and WAL/SHM files, sessions, logs, caches, shell snapshots, and worktrees.

Repository code must use paths relative to the repository root or explicit
environment variables. Machine-specific absolute paths are not deployment
configuration.
