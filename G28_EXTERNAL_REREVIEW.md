# G28 Amendment External Re-Review

**Date:** 2026-07-16

**Scope:** proposal readiness of the amended `G8/G16/G28` Kron-Schur
GridFM contract. This review does not assess implementation completion,
submission readiness, or treatment efficacy.

**Reviewer route:** `claude-review`, cross-family from the Codex executor

**Reviewer thread:** `3cd19e69-999a-4814-a4dc-a0f0d059650d`

**Reviewed repository state:** clean `research/kron-schur` at
`5e16b1f5438545c32a93947378de8173e648845d`; executable amendment at
`c690d6d6fd6e71187f0f4659c8daf52becbba69a`

## Final Decision

- **Verdict:** READY
- **Weighted score:** 9.05 / 10
- **Reviewed endpoint:** `G8/G16/G28`
- **Proposal-level blockers:** none
- **Campaign authorization:** unchanged; every treatment launch remains blocked
  until R014 and all of its prerequisites pass
- **Treatment evidence credited:** none; no confirmatory treatment result exists

The reviewer accepted the G32-to-G28 change as an outcome-blind feasibility
repair. The amendment preserves the 20-run matrix, six-group exact inference,
0.5k--13.7k target envelope, size-extrapolation requirement, Quotient control,
common-FLOP comparisons, and claim boundary. It changes the tested diversity
range from 8--32 to 8--28 and makes no universal scaling-law claim.

## Dimension Scores

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

## Amendment Audit

| Question | Decision | Failure condition |
|---|---|---|
| Outcome-blind repair or cherry-picking? | Legitimate repair. Selection reads topology metadata and frozen constraints only; no treatment output was available or inspected. | Any PF target, operating scenario, model output, or treatment result influencing the amendment would invalidate this conclusion. |
| Does `8 -> 16 -> 28` support the diversity claim? | Yes, for the proposal's narrow endpoint/non-contraction claim. Irregular spacing would not support a fitted exponent, but no such law is claimed. | Claiming a universal scaling exponent or a geometric-spacing law would exceed the design. |
| Does metadata-based split selection bias the paired treatment contrast? | No. Bus count and provenance determine one common source/target split for every arm. | Arm-dependent membership or outcome-dependent extrapolation would bias the comparison. |
| Is the fixed G28 split coherent with R008? | Yes, because R008 fails closed. Exactly six held-out groups either achieve the preregistered 80% design power or the campaign blocks. | Launching below 80% design power would violate the contract. |
| Is the unchanged 20-run matrix adequate? | Yes at proposal-design level. The target groups, two seeds, treatment arms, and exact group inference are unchanged. | A formal cross-level slope claim would require a different design; the proposal makes only a conjunctive endpoint/non-contraction claim. |
| Are novelty, causal attribution, or statistical validity weakened? | No. The same Flat, Global, Kron, and same-partition Quotient intervention remains. | Removing Quotient, changing target outcomes/groups after inspection, or broadening the claim would reopen review. |
| Are the canonical artifacts internally consistent? | Yes after maximality evidence was checked. | Drift among proposal, plan, tracker, executable matrix, or R002 evidence would reopen review. |

## Maximality Check

The review initially questioned whether the phrase "largest feasible" was
supported because the current PASS record fixes `source_count=28`. The concern
was retracted after checking the pre-amendment R002 records and re-running the
same deterministic audit over the same pinned inventory:

| Source count | Audit status | Feasible assignments at minimum group count |
|---:|---|---:|
| 28 | PASS | 1 |
| 29 | BLOCKED | 0 |
| 30 | BLOCKED | 0 |
| 31 | BLOCKED | 0 |
| 32 | BLOCKED | 0 |

The earlier `R002-split-audit-1127527.json` and
`R002-split-audit-bf22c01.json` records independently preserve the G32 BLOCKED
state. G28 is therefore the largest feasible integer source count under the
frozen whole-group, target-count, target-envelope, and size-extrapolation
constraints.

## Non-Blocking Residual Risks

1. **No independent-group slack.** The split has exactly six target provenance
   groups. If source-only dispersion implies that six groups provide less than
   80% design power at R008, the campaign cannot run.
2. **Size/provenance entanglement.** The largest extrapolative targets are
   concentrated in `epigrids`, `pegase`, and `rte`. Group-balanced analysis
   protects the arm contrast but does not make size independent of provenance;
   the paper must disclose this interpretation limit.
3. **Fixed source cap.** PGLib cases above the selected 4,917-bus source maximum
   are outside the G28 source set. This defines the tested regime rather than a
   universal source-size claim.
4. **Nested membership must remain frozen.** The campaign builder enforces
   `G8 subset G16 subset G28` and sizes 8/16/28. R013 must still hash the exact
   membership and source-only construction rule.

These are risks, not permissions to relax gates after observing efficacy.

## Results-to-Claims Matrix

| Outcome | Allowed claim |
|---|---|
| R008 or any R014 prerequisite fails | No confirmatory campaign and no treatment claim. Report a blocked protocol or pilot only. |
| C1 passes every error, residual, checkpoint, diversity, and size gate | Kron-Schur geometry improved the tested held-out PF error--physics--compute frontier over Flat and Global across the preregistered G8--G28 and target-size ranges. |
| Endpoint superiority passes but diversity or size non-contraction fails | No scalability claim; report the narrower endpoint result only if its prespecified gate remains valid. |
| C2 also beats Quotient under its paired gate | Attribute the measured advantage to the full tested electrical-operator family, not to coefficient values alone. |
| Quotient ties or beats Kron | Electrical-specific claim fails; any supported result reduces to generic multiscale communication. |
| Flat or Global ties or beats Kron | The electrical hierarchy is not justified relative to the simpler tested mechanism. |

## Required Next Work

No extra experiment was added by the re-review. The existing fail-closed queue
remains controlling:

1. generate and audit admissible source-development PF data;
2. close R003/R004 source-only geometry and capacity calibration;
3. close I010 and the GPU-dependent calibration, power, profile, and budget
   gates;
4. freeze exact memberships, configs, and hashes at R013; and
5. authorize E001--E020 only through R014.

## Review Rounds and Trace

1. **Initial external call:** completed without a substantive verdict after the
   reviewer attempted unavailable internal delegation. It is retained as a
   failed trace rather than hidden.
2. **Constrained verdict follow-up:** READY, initially 9.04 / 10, with one
   non-blocking concern that source-count maximality had not been demonstrated
   in the supplied packet.
3. **Maximality addendum:** the reviewer accepted the existing G32 BLOCKED
   records and the 28--32 deterministic enumeration, retracted that concern,
   and restored the final 9.05 / 10 score.

Raw requests, responses, metadata, and the cross-family route are under
`.aris/traces/research-review/2026-07-16_run01/`:

- `001-g28-amendment-review.*`
- `002-g28-verdict-followup.*`
- `003-g28-maximality-addendum.*`

## Reviewer-of-Record Statement

> **G28 amendment re-review (proposal scope only).** The label-blind feasibility
> amendment from G8/G16/G32 to **G8/G16/G28** is a legitimate, outcome-blind
> protocol repair. The selection logic reads topology metadata and frozen
> constraints, never treatment outcomes, and deterministic audits confirm G28
> is the largest feasible integer source count under the frozen constraints.
> The amendment preserves the 20-run matrix, six-group exact inference, target
> envelope, size-extrapolation requirement, Quotient control, and claim
> boundary. **Verdict: READY; 9.05 / 10; reviewed status restored for the G28
> proposal.** This is proposal readiness only: implementation and submission
> remain unverified, no treatment result is credited, and all treatment launches
> remain gated by R014.
