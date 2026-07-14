# Sync layout

This repository uses Git for source code and Syncthing for research artifacts.
The saved Syncthing configuration on the Mac currently maps these repository
directories to the paired devices:

| Folder | Local path | Purpose |
| --- | --- | --- |
| `gridfm-idea-stage` | `idea-stage/` | Discovery and prototype notes |
| `gridfm-papers` | `papers/` | Paper working artifacts |
| `gridfm-refine-logs` | `refine-logs/` | Proposal/refinement records |
| `gridfm-research-wiki` | `research-wiki/` | Durable research wiki |
| `gridfm-mlruns` | `mlruns/` | Experiment tracking artifacts |

The corresponding portable Codex mappings are:

| Folder | Mac source | Recommended HPC destination |
| --- | --- | --- |
| `codex-skills` | `~/.codex/skills/` | `~/sync/aris/codex/skills/` |
| `codex-memories` | `~/.codex/memories/` | `~/sync/aris/codex/memories/` |

The HPC destinations are a convention, not a claim about the remote device's
current configuration. Accept the folders on the HPC device at those paths,
or use another path consistently on that device.

Do not sync the whole `~/.codex` directory. In particular, exclude credentials
(`auth.json`, API-key-bearing configuration), SQLite databases and WAL/SHM
files, live sessions, logs, caches, shell snapshots, and worktrees. These are
machine-local state and can either leak secrets or corrupt when two clients
write them concurrently.

Repository code must refer to artifact directories relative to the repository
root or through an explicit environment variable. Absolute paths such as
`/Users/carmine/Code/FM/gridfm-graphkit` are legacy-only and are not portable
to the HPC checkout.
