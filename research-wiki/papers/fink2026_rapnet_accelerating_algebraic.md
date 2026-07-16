---
type: paper
node_id: paper:fink2026_rapnet_accelerating_algebraic
title: "RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections"
authors: ["Yali Fink", "Ido Ben-Yair", "Lars Ruthotto", "Eran Treister"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.26854"
  doi: null
  s2: null
tags: ["learned-operators", "amg"]
added: 2026-07-07T17:21:00Z
---

# RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections

## One-line thesis
RAPNet: accelerates algebraic multigrid with learned sparse corrections - a learned closure on top of a fixed linear coarse solver.

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

> The scalable solution of large sparse linear systems is a bottleneck in scientific computing and graph analysis. While algebraic multigrid (AMG) offers optimal linear scaling, its performance is severely constrained by the trade-off between the sparsity and convergence quality of coarse-grid operators. Classical AMG heuristics struggle to balance these objectives, often sacrificing stability or performance for sparsity. We propose RAPNet, a graph neural network (GNN) framework that resolves this trade-off by learning to generate sparse, robust coarse operators directly from the sparse algebraic system. Key to our approach is a level-wise training strategy that enables learning from small subgraphs and generalization to million-node domains, bypassing the bottlenecks of prior neural AMG attempts. RAPNet executes exclusively during the solver setup phase, ensuring that the solve phase retains its favorable computational properties. We show that our method outperforms classical non-Galerkin baselines on diverse PDE discretizations and graph Laplacians, making it particularly effective for multi-query tasks such as eigenproblems, time-dependent simulations, and inverse or design problems.
