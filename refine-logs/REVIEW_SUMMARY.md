# Review Summary

## Outcome

- **Current external verdict:** PENDING for the `G8/G16/G26` contract
- **Current amendment status:** topology-only feasibility repair complete; re-review unavailable because the Claude OAuth session expired before a verdict
- **Latest applicable weighted score:** none for G26; the historical G28 review scored 9.05 / 10
- **Reviewer of record:** pending
- **Model route:** `claude-review`
- **Reasoning effort:** high-rigor
- **Failed G26 session:** `c7f48ef5-62ab-4a78-a74f-aad973e2db90`
- **Calibration:** none
- **Readiness scope:** the historical READY verdict covers G28 only and does not transfer to G26
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
| 6 | Claude cross-family G28 amendment re-review | 9.05 | READY |
| 7 | Claude cross-family G26 amendment re-review | — | AUTHENTICATION FAILED; NO VERDICT |

Fable reopened the earlier READY result because the quotient control claimed an impossible equality of nonzero counts and the six-group design lacked a power/MDE specification. The final candidate removes the impossible equality, narrows Claim 2 accordingly, and makes insufficient power a pre-campaign block.

The historical G28 re-review separately tested whether that feasibility
amendment changed the estimand, causal control, statistical design, or claim
boundary. It found no proposal-level blocker. The outcome-blind audit reads
topology metadata only; the existing G32 BLOCKED records and a deterministic
28--32 enumeration confirm that G28 is the largest feasible integer source count
under the constraints used at that time. The full review is in
`G28_EXTERNAL_REREVIEW.md`; raw interaction traces are under
`.aris/traces/research-review/2026-07-16_run01/`.

That audit omitted the already-preregistered requirement for at least two
disjoint whole-provenance source-development groups. A corrected exhaustive
topology-only enumeration finds G26 maximal and deterministically reserves
PSERC and ACTIV. The attempted G26 review produced no tokens or verdict because
the Claude OAuth session could not be refreshed. Treatment remains blocked.

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
11. The current diversity endpoint is G26. The G8/G16/G26 matrix supports only the preregistered endpoint and non-contraction claims, not a universal scaling exponent; external review of this amendment is pending.

## Residual Risks

- The power design assumes hierarchy-arm group dispersion does not exceed Flat-HGNS and paired arm errors are nonnegatively correlated. Violating this can under-provision groups; it cannot invalidate the exact test, but it can cause an honest failed gate.
- Exactly six held-out provenance groups leave no R008 power slack. If six groups do not provide at least 80% source-only design power, the campaign blocks.
- Size and provenance are partly entangled because the largest extrapolative targets concentrate in three held-out groups. Group-balanced analysis protects the treatment contrast but not a provenance-free size interpretation.
- Dense Kron construction may fail host-memory or build-time limits on the largest targets. Such failures remain in the denominator.
- Novelty depends on a clean Kron-over-quotient result with non-contracting size behavior. A quotient tie reduces the result to generic hierarchy.
- The 2026 GridFM literature is moving quickly; adjacent learned-AMG, effective-resistance, and Ward/Kron-equivalent work requires another primary-source sweep before submission.

## Next Boundary

The label-blind 2026-07-17 repair replaces G28 with G26 so that two whole-provenance source-development groups are disjoint from training and target groups while preserving the 20-run matrix, six-group inference, target-size envelope, and size-extrapolation claim. External review is pending; the exact balanced `S_total=11,655` audit passes, but clean-commit evidence and R014 remain BLOCKED.
