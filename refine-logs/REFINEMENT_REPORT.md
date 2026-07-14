# Refinement Report

## Task

Review the actual canonical proposal, make the implementation as general and simple as the causal question permits, recenter the work on scalability for foundational grid models, make the `cardef/gridfm-graphkit` fork explicit, audit novelty, use Fable 5 Max as the final reviewer, and remove intermediate proposal and plan artifacts after consolidation.

## Result

The canonical proposal now specifies a controlled test of deterministic Kron–Schur communication inside one parameter-shared multi-topology model. Fable 5 Max assigns **9.05 / 10, READY** for proposal readiness. Novelty is **7.0 / 10, PROCEED WITH CAUTION**. No implementation or empirical result is claimed.

## Principal Refinements

- Reframed the primary question around source-topology diversity `G8/G16/G32`, unseen-grid size, and cumulative training FLOPs at fixed capacity and total scenarios.
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

All Fable calls used session `f5411f61-3db6-4c81-87e1-0020d7fcbc5c`, model route `claude-fable-5`, maximum reasoning effort, calibration none, and read-only repository access. Earlier AGY/Antigravity rounds remain represented in `score-history.md` and immutable historical artifacts.

## Novelty Boundary

The search found direct prior art for multi-topology grid learning, GridFMs, generic Kron pooling, multiscale message passing, global graph summaries, and hierarchical PF predictors. It did not locate the complete same-partition Kron-versus-quotient intervention with matched flat and domain-global controls under common cumulative FLOPs. This is an absence-of-evidence conclusion, not a priority claim.

The components are LOW novelty. The controlled experimental combination is MEDIUM novelty. A clean non-contracting Kron-over-quotient result could be HIGH finding novelty. A quotient tie collapses the electrical claim to generic hierarchy.

## Cleanup

The final proposal, review summary, refinement report, novelty report, state, score history, and immutable reviewer traces retain the full decision record. The working round-5 proposal is removed after its proposal body is published canonically. Legacy experiment-result files remain because they are evidence from the earlier prototype, not intermediate proposals or plans. Historical timestamped final artifacts remain immutable.

