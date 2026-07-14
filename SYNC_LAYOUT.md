# Branch and sync ownership

The operational Mac/Linux branch workflow is documented in
[`GIT_SYNCTHING_WORKFLOW.md`](GIT_SYNCTHING_WORKFLOW.md).

A fresh clone must build, test, and run without Syncthing or agent tooling.

## Git

Git owns every deployment input and durable code contract:

- `main`: upstream-aligned package, tests, CI, and container configuration.
- `research/*`: research implementations and canonical plans.
- `exp/*`: short-lived code variants.
- Annotated `run/*` tags: immutable pointers to executed revisions.

`AGENTS.md` and `CLAUDE.md` are tracked because they describe repository
policy. Tool installations, transcripts, caches, and mutable memories are not
repository policy.

## Syncthing and MLflow

Syncthing or MLflow may own ignored, high-churn state such as:

- `idea-stage/`, `papers/`, `refine-logs/`, and `research-wiki/`;
- `mlruns*/`, checkpoints, run logs, and result directories;
- selected global agent memories under an explicit single-writer policy.

Syncthing is replication, not version control. Never synchronize tracked files
between worktrees, and never use a synchronized result directory as the sole
record of which code ran. Each run manifest should record at least the Git SHA,
configuration hash, data hash or immutable data identifier, environment lock or
image digest, and output location.

The allowed repo-local Syncthing roots are declared in `sync-layout.json`.
`main` must track no files below those roots. Verify both the Git tree and the
live machine configuration with:

```bash
python tools/check_syncthing_boundary.py --require-local-config
```

The check fails if a Syncthing root contains the repository, if an undeclared
folder is synchronized inside it, if a declared mapping drifts, or if `main`
starts tracking a file below a synchronized root. It runs in pre-commit and CI.

## Agent state

Install skills from their source repository. Do not synchronize generated
project-local symlink farms. Never synchronize an entire `~/.codex`,
`~/.claude`, `.codex`, `.claude`, or `.aris` tree. Credentials, API-key-bearing
configuration, databases and WAL/SHM files, sessions, logs, caches, shell
snapshots, and worktrees are machine-local.

Repository code must use relative paths or explicit environment variables;
machine-specific absolute paths are not deployment configuration.
