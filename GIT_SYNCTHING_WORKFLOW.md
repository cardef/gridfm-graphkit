# Git, Syncthing, and ARIS workflow

Git owns code and reproducibility. Syncthing owns mutable research artifacts.
Syncthing is bidirectional, but it does not understand branches, commits,
merges, or tags.

## Ownership

| Item | Owner | Purpose |
| --- | --- | --- |
| `main` | Git | Clean, tested, deployable repository |
| `research/<paper>` | Git | Durable research line for one improvement/paper |
| `exp/<paper>-<variant>` | Git | Short-lived code variant or ablation |
| `run/<id>` tag | Git | Immutable pointer to an executed revision |
| `refine-logs/` | ARIS + Syncthing | Working proposals, plans, reviews, checkpoints |
| `idea-stage/` | ARIS + Syncthing | Idea discovery and pilot notes |
| `research-wiki/` | ARIS + Syncthing | Working literature and claim knowledge base |
| `papers/` | Syncthing | Paper drafts, PDFs, and working figures |
| `mlruns/` | Linux/HPC + Syncthing | MLflow runs, checkpoints, logs, results |

Git must not track files below a Syncthing root. The boundary is checked by
`tools/check_syncthing_boundary.py` in pre-commit and CI.

## Mac, workstation, and HPC roles

- Mac: research coordination, ARIS runs, contract review, branch integration.
- Linux workstation: bidirectional Syncthing peer and the local artifact
  workspace used to bridge files to and from GPFS.
- HPC/GPFS: checkout of the committed revision and expensive runs; GPFS is not
  directly managed by Syncthing in the current deployment.

Mac and workstation may write synchronized files, but they must not edit the
same logical file concurrently. Run ARIS refinement on one machine at a time.
Use the explicit one-way routes in `SYNC_LAYOUT.md` to bridge artifacts
between the workstation and GPFS.

## Start a research line

Create one research branch per improvement or paper, not one branch per run:

```bash
git switch main
git pull --ff-only origin main
git switch -c research/<paper-slug>
```

Use an `exp/` branch for a temporary code hypothesis:

```bash
git switch -c exp/<paper-slug>-<variant> research/<paper-slug>
```

## ARIS contract flow

ARIS and the research skills expect their working files in `refine-logs/`.
Those files are ignored by Git and synchronized by Syncthing.

After a proposal, plan, or review passes its gate, promote a reviewed snapshot
into the Git-owned research directory. On a research branch with the helper:

```bash
python tools/promote_kron_schur_contract.py
git diff -- research/kron-schur/
git add research/kron-schur/
git commit -s -m "Update research contract"
git push -u origin research/<paper-slug>
```

Promotion is one-way: ARIS writes the working copy, and Git records the
reviewed snapshot.

## Run on Linux/HPC

Pull the research branch and record the exact revision before launching a run:

```bash
git fetch origin
git switch research/<paper-slug>
git pull --ff-only origin research/<paper-slug>
git rev-parse HEAD
```

For an immutable checkout, detach at the recorded SHA:

```bash
git switch --detach <git-sha>
```

Write logs, checkpoints, and results under the GPFS checkout's `mlruns/`
artifact root. The run manifest must record the Git SHA, configuration hash,
data identifier/hash, environment or image digest, and output location. After
the run stops writing, copy `mlruns/` GPFS → workstation with the preview,
copy, and checksum-verification sequence in `SYNC_LAYOUT.md`; Syncthing then
replicates the workstation copy to the Mac.

## Fixes discovered on Linux

Do not edit the same research branch concurrently on Mac and Linux. Create a
focused fix branch and let the Mac integrate it:

```bash
git switch -c fix/<slug> research/<paper-slug>
git commit -s -am "Fix <slug>"
git push -u origin fix/<slug>
```

## Update branches

Keep a published research line current without rewriting its public history:

```bash
git switch research/<paper-slug>
git fetch origin
git merge origin/main
git push origin research/<paper-slug>
```

Update the fork from upstream only on `main`:

```bash
git switch main
git fetch upstream
git merge upstream/main
# run tests and build checks
git push origin main
```

Never push to `upstream`.

## Promote validated research to `main`

Only code, configuration, tests, manifests, and reviewed documentation move
into the deployable branch. Do not merge checkpoints, MLflow databases, raw
logs, or bulk result directories.

```bash
git switch main
git pull --ff-only origin main
git merge --no-ff research/<paper-slug>
# run repository verification commands
git push origin main
```

## Parallel papers

Syncthing folders are branch-blind. Do not run two independent papers in the
same checkout while both use `refine-logs/`, `idea-stage/`, or `research-wiki/`.
Work sequentially or give each paper its own checkout and artifact namespace.

## Safety checks

Before committing code:

```bash
git status -sb
python tools/check_syncthing_boundary.py --require-local-config
```

Before a confirmatory run, verify the Git SHA and manifest. Before merging to
`main`, run tests, build checks, and security checks.

`CLAUDE.md` is repository policy and is not part of the Syncthing workflow.
