# Refinement Report

## Task

Review the actual canonical proposal, make the implementation as general and simple as the causal question permits, recenter the work on scalability for foundational grid models, make the `cardef/gridfm-graphkit` fork explicit, audit novelty, use Fable 5 Max as the final reviewer, and remove intermediate proposal and plan artifacts after consolidation.

## Result

The canonical proposal specifies a controlled test of deterministic Kron–Schur communication inside one parameter-shared multi-topology model. Fable 5 Max assigned **9.05 / 10, READY** to the earlier G32 contract, and a cross-family review assigned **9.05 / 10, READY** to the superseded G28 contract. The current G26 source-development repair is pending external review because the available Claude session failed authentication before a verdict. Novelty remains **7.0 / 10, PROCEED WITH CAUTION**. No confirmatory empirical result is claimed.

## Principal Refinements

- Reframed the primary question around source-topology diversity, unseen-grid size, and cumulative training FLOPs at fixed capacity and total scenarios. The 2026-07-17 topology-only repair changes the endpoint from `G28` to `G26` after adding the required two disjoint whole-provenance source-development groups; exhaustive enumeration finds G26 maximal.
- Made `cardef/gridfm-graphkit` the explicit implementation and reproducibility boundary relative to upstream `gridfm/gridfm-graphkit`.
- Reduced the architecture to a common encoder/local-backbone/communication-slot/decoder contract.
- Removed HELM reconstruction, affine physical unpooling, per-scenario hierarchy quantities, target-fitted normalization, and recursive hierarchy from the confirmatory path.
- Defined a conservative real-latent transport pair and retained complex circuit information only as edge attributes and Schur support.
- Added a pinned GridSFM-style global control and a same-partition quotient control.
- Corrected the quotient control: it shares a sparsity cap but is not padded to imitate Schur fill. Realized bandwidth is reported and charged through common cumulative-FLOP checkpoints.
- Bounded all source-only choices: at most twelve geometry policies, at most three common loss candidates, deterministic capacity matching, and one common communication slot.
- Replaced topology/seed pseudo-replication with exact provenance-group inference and inverted one-sided bounds.
- Added a source-only MDE/power calculation that freezes the independent target-group count or blocks the campaign.
- Moved forbidden-legacy-import enforcement to the first F2 commit.
- Demoted few-shot adaptation to an unbudgeted exploratory appendix and reduced build-cost amortization to 1 and 1000 scenarios.

## Reviewer Provenance

| Review | Transport and model | Result | Immutable response |
|---|---|---|---|
| Initial independent pass | Fable 5 Max | 8.325, NOT READY; novelty 6.5 | [response](../.aris/traces/novelty-check/2026-07-13_run02/002-fable5-max-final-review.response.md) |
| Round-5 re-review | Fable 5 Max | 8.6875, NOT READY; novelty 7.0 | [response](../.aris/traces/novelty-check/2026-07-13_run02/003-fable5-max-round5-rereview.response.md) |
| Structural final gate | Fable 5 Max | 8.9375, NOT READY; all structural gates passed | [response](../.aris/traces/novelty-check/2026-07-13_run02/004-fable5-max-round5-final-gate.response.md) |
| Disclosure audit | Fable 5 Max | 9.05, READY; novelty 7.0 | [response](../.aris/traces/novelty-check/2026-07-13_run02/005-fable5-max-round5-sentence-audit.response.md) |
| G28 feasibility-amendment re-review | Claude via `claude-review` | 9.05, READY; no proposal-level blocker | [verdict](../.aris/traces/research-review/2026-07-16_run01/002-g28-verdict-followup.response.md), [addendum](../.aris/traces/research-review/2026-07-16_run01/003-g28-maximality-addendum.response.md) |
| G26 source-development amendment re-review | Claude via `claude-review` | no verdict; OAuth refresh failed before tokens | local failed trace, session `c7f48ef5-62ab-4a78-a74f-aad973e2db90` |

All Fable calls used session `f5411f61-3db6-4c81-87e1-0020d7fcbc5c`, model route `claude-fable-5`, maximum reasoning effort, calibration none, and read-only repository access. Earlier AGY/Antigravity rounds are consolidated in `REVIEW_ARCHIVE.md` and summarized in `score-history.md`.

The G28 re-review used cross-family `claude-review` thread
`3cd19e69-999a-4814-a4dc-a0f0d059650d`. Its first turn produced no verdict
and is retained as a failed trace; the bounded follow-up produced the review,
and the final addendum accepted the pre-amendment G32 BLOCKED records plus the
deterministic 28--32 maximality check. No treatment result was supplied or
credited.

The G26 re-review attempt was bounded to the source-development inconsistency,
maximality proof, deterministic PSERC/ACTIV tie-break, and claim preservation.
Authentication failed before model output, so no independent judgment is
attributed to that attempt and the amendment remains pending review.

## Novelty Boundary

The search found direct prior art for multi-topology grid learning, GridFMs, generic Kron pooling, multiscale message passing, global graph summaries, and hierarchical PF predictors. It did not locate the complete same-partition Kron-versus-quotient intervention with matched flat and domain-global controls under common cumulative FLOPs. This is an absence-of-evidence conclusion, not a priority claim.

The components are LOW novelty. The controlled experimental combination is MEDIUM novelty. A clean non-contracting Kron-over-quotient result could be HIGH finding novelty. A quotient tie collapses the electrical claim to generic hierarchy.

## Cleanup

The final proposal, review summary, refinement report, novelty report, review archive, score history, and current experiment evidence retain the durable decision record. Superseded timestamped copies, Syncthing conflict files, and stale resume checkpoints were removed after consolidation. `REFINE_STATE.json` remains an ignored machine-local checkpoint. Legacy experiment evidence remains explicitly scoped and cannot authorize confirmatory runs.
