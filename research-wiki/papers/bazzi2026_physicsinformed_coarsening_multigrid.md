---
type: paper
node_id: paper:bazzi2026_physicsinformed_coarsening_multigrid
title: "Physics-Informed Coarsening for Multigrid Graph Neural Surrogates"
authors: ["Amir Bazzi", "David Cardinaux", "Ramy Nemer", "Jose Alaves", "Arjun Kalkur Matpadi Raghavendra", "Elie Hachem"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.31013"
  doi: null
  s2: null
tags: ["gnn", "multigrid", "coarsening"]
added: 2026-07-07T17:21:00Z
---

# Physics-Informed Coarsening for Multigrid Graph Neural Surrogates

## One-line thesis
Physics-informed coarsening for multigrid GNN surrogates (solid mechanics): encoder-processor-decoder with physics-derived pooling - nearest published neighbor to Kron-Schur U-Net.

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

> Learning-based surrogates for partial differential equations have recently matched the accuracy of classical solvers while achieving orders-of-magnitude speedups, predominantly in fluid settings and structured geometries. In contrast, robust surrogates for deformable solids remain underexplored, despite the presence of nonlinear elasticity, plasticity, and transient behavior that challenge standard architectures. We introduce a multigrid graph neural network for solid mechanics that couples an encoder-processor-decoder backbone with a physics-informed coarsening strategy. Instead of downsampling via geometric heuristics, our method scores nodes using a residual-based measure of local physical activity and preferentially retains regions of high strain or stress concentration, allocating multiscale capacity where it is most needed. This preserves long-range interactions through hierarchical message passing while improving stability over long rollouts. We evaluate on multiple datasets covering linear, nonlinear, and transient regimes, and observe consistent gains in accuracy and rollout stability compared to standard sampling baselines. Our results highlight the importance of physics-informed coarsening for scalable surrogate modeling in solid mechanics.
