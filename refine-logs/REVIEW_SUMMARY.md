# Review Summary

## Outcome

- **Last external verdict:** READY for the pre-amendment `G8/G16/G32` contract
- **Current amendment status:** `G8/G16/G28`; targeted external re-review pending
- **Final weighted score:** 9.05 / 10
- **Reviewer of record:** Fable 5 Max
- **Model route:** `claude-fable-5`
- **Reasoning effort:** max
- **Session:** `f5411f61-3db6-4c81-87e1-0020d7fcbc5c`
- **Calibration:** none
- **Readiness scope:** the 9.05 score applies to the pre-amendment proposal only; implementation and submission remain unverified
- **Novelty verdict:** 7.0 / 10 — PROCEED WITH CAUTION

## Review Evolution

| Stage | Reviewer | Score | Verdict |
|---|---|---:|---|
| 1 | AGY / Antigravity, Claude Opus 4.6 Thinking | 7.85 | REVISE |
| 2 | AGY / Antigravity, Claude Opus 4.6 Thinking | 8.70 | REVISE |
| 3 | AGY / Antigravity, Claude Opus 4.6 Thinking | 9.10 | READY |
| 4 | Fable 5 Max independent review | 8.325 | NOT READY |
| 5 | Fable 5 Max re-review | 8.6875 | NOT READY |
| 5a | Fable 5 Max structural gate | 8.9375 | NOT READY |
| 5b | Fable 5 Max disclosure audit | 9.05 | READY |

Fable reopened the earlier READY result because the quotient control claimed an impossible equality of nonzero counts and the six-group design lacked a power/MDE specification. The final candidate removes the impossible equality, narrows Claim 2 accordingly, and makes insufficient power a pre-campaign block.

## Final Dimension Scores

| Dimension | Weight | Score |
|---|---:|---:|
| Problem Fidelity | 15% | 9.00 |
| Method Specificity | 25% | 9.50 |
| Contribution Quality | 25% | 9.00 |
| Frontier Leverage | 15% | 8.75 |
| Feasibility | 10% | 8.50 |
| Validation Focus | 5% | 9.50 |
| Venue Readiness | 5% | 8.75 |
| **Weighted total** | **100%** | **9.05** |

## Load-Bearing Decisions

1. The implementation boundary is explicitly the `cardef/gridfm-graphkit` research fork of upstream `gridfm/gridfm-graphkit`.
2. The primary question is foundation-model scalability across source-topology diversity, unseen-grid size, and cumulative FLOPs at fixed learned capacity and total scenarios.
3. One common communication slot is the sole learned treatment seam. The encoder, local backbone, decoder, objective, normalization, data, and checkpoint schedule remain common.
4. The confirmatory path excludes HELM reconstruction, affine physical unpooling, per-grid fitted normalization, target-derived preprocessing, and legacy hierarchy imports.
5. Kron and quotient arms share partition, adapter, channel schema, coarse-node count, and sparsity cap. They are not falsely edge-count-matched; realized support density is reported and charged in common cumulative-FLOP comparisons.
6. The primary unit is a provenance group after topology and seed aggregation. Exact one-sided sign-flip inference uses the unweighted mean group contrast.
7. Target-group count is frozen by a source-only 80%-power calculation for a preregistered 5% minimum relevant error reduction. Failure to obtain enough independent groups blocks the campaign.
8. Geometry policy, loss weights, widths, and local depth are selected by bounded source-only rules before treatment runs.
9. The forbidden-import `MetaPathFinder` test runs from the first F2 commit onward.
10. Component novelty is not claimed. The defensible contribution is the controlled scaling study and, conditionally, its empirical finding.

## Residual Risks

- The power design assumes hierarchy-arm group dispersion does not exceed Flat-HGNS and paired arm errors are nonnegatively correlated. Violating this can under-provision groups; it cannot invalidate the exact test, but it can cause an honest failed gate.
- Dense Kron construction may fail host-memory or build-time limits on the largest targets. Such failures remain in the denominator.
- Novelty depends on a clean Kron-over-quotient result with non-contracting size behavior. A quotient tie reduces the result to generic hierarchy.
- The 2026 GridFM literature is moving quickly; adjacent learned-AMG, effective-resistance, and Ward/Kron-equivalent work requires another primary-source sweep before submission.

## Next Boundary

The label-blind 2026-07-16 feasibility audit replaced `G32` with `G28` while preserving the 20-run matrix, six-group inference, target-size envelope, and size-extrapolation claim. This amendment requires a targeted proposal re-review before reviewed status can be restored. Implementation must still pass F0–F5 in the fork before the campaign.
