# Research Artifact Manifest

This manifest records current durable artifacts, not every intermediate file
produced by an ARIS run. Git history is the version history for text artifacts.

## Git-owned research text

| Path | Role | Status |
|---|---|---|
| `idea-stage/IDEA_REPORT.md` | Historical idea-discovery decision report | Superseded by the refined contract but retained for provenance |
| `idea-stage/HELM_UNPOOL_NOTE.md` | Legacy HELM pilot interpretation used by historical code | Retained as scoped prototype evidence |
| `idea-stage/pilot_cpu.py` and `pilot_results.txt` | CPU locality and Kron-fill pilot | Retained as unique evidence |
| `idea-stage/helm_unpool_pilot.py` and `helm_unpool_results.json` | Legacy HELM pilot | Retained as unique evidence |
| `refine-logs/FINAL_PROPOSAL.md` | Sole method and claim boundary | Canonical G28 amendment; targeted re-review pending |
| `refine-logs/EXPERIMENT_PLAN.md` | Constants, estimands, budgets, and run rules | Canonical |
| `refine-logs/EXPERIMENT_TRACKER.md` | Implementation, freeze, campaign, and analysis gates | Canonical; treatment runs blocked until R014 |
| `refine-logs/REVIEW_SUMMARY.md` | Current review verdict and amendment status | Canonical |
| `refine-logs/NOVELTY_REPORT.md` | Source-backed novelty boundary | Canonical |
| `refine-logs/REFINEMENT_REPORT.md` | Current refinement and feasibility audit | Canonical |
| `refine-logs/REVIEW_ARCHIVE.md` | Consolidated earlier reviewer responses | Historical provenance |
| `refine-logs/EXPERIMENT_RESULTS.md` | Current performed-evidence summary | Zero confirmatory treatment runs |
| `refine-logs/EXPERIMENT_CODE_REVIEW.md` | Current implementation review | Retained gate evidence |
| `refine-logs/score-history.md` | Reviewer score trail | Historical provenance |
| `research-wiki/` | Literature, idea, claim, and graph knowledge base | Git-owned living knowledge base |
| `research/kron-schur/runs/*.json` | Immutable legacy run receipts | Historical evidence only |

`refine-logs/REFINE_STATE.json` is a machine-local resume checkpoint. It is not
a durable artifact and is intentionally ignored.

## Syncthing-owned payloads

| Path | Receipt | Rule |
|---|---|---|
| `papers/` | `research/papers-manifest.json` | Immutable PDFs; verify hashes after transfer |
| `mlruns/` | Per-run receipts under `research/kron-schur/runs/` | Sync only quiescent run artifacts; never use as deployment input without a committed receipt |

## Removed intermediates

Timestamped copies of canonical documents, Syncthing conflict files, stale
resume checkpoints, raw brainstorm bundles, cache files, and obsolete recovery
documents were consolidated or removed on 2026-07-16. They are not sources of
truth.
