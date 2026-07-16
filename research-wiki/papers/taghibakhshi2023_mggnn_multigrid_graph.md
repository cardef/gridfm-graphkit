---
type: paper
node_id: paper:taghibakhshi2023_mggnn_multigrid_graph
title: "MG-GNN: Multigrid Graph Neural Networks for Learning Multilevel Domain Decomposition Methods"
authors: ["Ali Taghibakhshi", "Nicolas Nytko", "Tareq Uz Zaman", "Scott MacLachlan", "Luke Olson", "Matthew West"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2301.11378"
  doi: null
  s2: null
tags: ["learned-operators", "multigrid", "gnn"]
added: 2026-07-07T17:20:59Z
---

# MG-GNN: Multigrid Graph Neural Networks for Learning Multilevel Domain Decomposition Methods

## One-line thesis
MG-GNN: GNN that learns restriction/prolongation for two-level domain decomposition methods - learned transfer operators instead of fixed algebraic ones.

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

> Domain decomposition methods (DDMs) are popular solvers for discretized systems of partial differential equations (PDEs), with one-level and multilevel variants. These solvers rely on several algorithmic and mathematical parameters, prescribing overlap, subdomain boundary conditions, and other properties of the DDM. While some work has been done on optimizing these parameters, it has mostly focused on the one-level setting or special cases such as structured-grid discretizations with regular subdomain construction. In this paper, we propose multigrid graph neural networks (MG-GNN), a novel GNN architecture for learning optimized parameters in two-level DDMs\@. We train MG-GNN using a new unsupervised loss function, enabling effective training on small problems that yields robust performance on unstructured grids that are orders of magnitude larger than those in the training set. We show that MG-GNN outperforms popular hierarchical graph network architectures for this optimization and that our proposed loss function is critical to achieving this improved performance.
