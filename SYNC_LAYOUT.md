# Artifact ownership and sync layout

A fresh clone must build, test, and run without Syncthing or agent tooling.
Git is the source of truth for every deployment input and repository contract.
Syncthing is optional replication for high-churn research artifacts that are
explicitly excluded from Git.

## Git-owned files

- Source, tests, packaging, CI, examples, and deployment configuration.
- `AGENTS.md` and `CLAUDE.md`, because they are portable repository policy.
- `MANIFEST.md` and the canonical Kron-Schur research contract under
  `research/kron-schur/`.

No Git branch may track files below a Syncthing root. `refine-logs/` contains
only timestamped logs and scratch artifacts; canonical files are kept in the
Git-owned research directory.

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
