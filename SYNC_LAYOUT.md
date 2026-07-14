# Branch and sync ownership

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

## Agent state

Install skills from their source repository. Do not synchronize generated
project-local symlink farms. Never synchronize an entire `~/.codex`,
`~/.claude`, `.codex`, `.claude`, or `.aris` tree. Credentials, API-key-bearing
configuration, databases and WAL/SHM files, sessions, logs, caches, shell
snapshots, and worktrees are machine-local.

Repository code must use relative paths or explicit environment variables;
machine-specific absolute paths are not deployment configuration.
