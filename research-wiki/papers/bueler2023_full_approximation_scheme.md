---
type: paper
node_id: paper:bueler2023_full_approximation_scheme
title: "A full approximation scheme multilevel method for nonlinear variational inequalities"
authors: ["Ed Bueler", "Patrick E. Farrell"]
year: 2023
venue: "SIAM Journal on Scientific Computing, 2024, Vol. 46, No. 4, pp. A2421--A2444"
external_ids:
  arxiv: "2308.06888"
  doi: null
  s2: null
tags: ["multigrid", "fas", "nonlinear"]
added: 2026-07-07T17:20:58Z
---

# A full approximation scheme multilevel method for nonlinear variational inequalities

## One-line thesis
Modern Full Approximation Scheme (FAS) multilevel method: coarse problem keeps full nonlinearity, tau-correction makes the coarse equation consistent with the fine residual.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> We present the full approximation scheme constraint decomposition (FASCD) multilevel method for solving variational inequalities (VIs). FASCD is a common extension of both the full approximation scheme (FAS) multigrid technique for nonlinear partial differential equations, due to A.~Brandt, and the constraint decomposition (CD) method introduced by X.-C.~Tai for VIs arising in optimization. We extend the CD idea by exploiting the telescoping nature of certain function space subset decompositions arising from multilevel mesh hierarchies. When a reduced-space (active set) Newton method is applied as a smoother, with work proportional to the number of unknowns on a given mesh level, FASCD V-cycles exhibit nearly mesh-independent convergence rates, and full multigrid cycles are optimal solvers. The example problems include differential operators which are symmetric linear, nonsymmetric linear, and nonlinear, in unilateral and bilateral VI problems.
