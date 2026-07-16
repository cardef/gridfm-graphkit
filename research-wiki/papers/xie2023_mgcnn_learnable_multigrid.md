---
type: paper
node_id: paper:xie2023_mgcnn_learnable_multigrid
title: "MGCNN: a learnable multigrid solver for sparse linear systems from PDEs on structured grids"
authors: ["Yan Xie", "Minrui Lv", "Chensong Zhang"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2312.11093"
  doi: null
  s2: null
tags: ["learned-operators", "multigrid"]
added: 2026-07-07T17:20:59Z
---

# MGCNN: a learnable multigrid solver for sparse linear systems from PDEs on structured grids

## One-line thesis
MGCNN: learnable multigrid solver for linear PDEs on structured grids; NN parameterizes smoothers/transfer inside a classical MG cycle.

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

> This paper presents a learnable solver tailored to iteratively solve sparse linear systems from discretized partial differential equations (PDEs). Unlike traditional approaches relying on specialized expertise, our solver streamlines the algorithm design process for a class of PDEs through training, which requires only training data of coefficient distributions. The proposed method is anchored by three core principles: (1) a multilevel hierarchy to promote rapid convergence, (2) adherence to linearity concerning the right-hand-side of equations, and (3) weights sharing across different levels to facilitate adaptability to various problem sizes. Built on these foundational principles and considering the similar computation pattern of the convolutional neural network (CNN) as multigrid components, we introduce a network adept at solving linear systems from PDEs with heterogeneous coefficients, discretized on structured grids. Notably, our proposed solver possesses the ability to generalize over right-hand-side terms, PDE coefficients, and grid sizes, thereby ensuring its training is purely offline. To evaluate its effectiveness, we train the solver on convection-diffusion equations featuring heterogeneous diffusion coefficients. The solver exhibits swift convergence to high accuracy over a range of grid sizes, extending from $31 \times 31$ to $4095 \times 4095$. Remarkably, our method outperforms the classical Geometric Multigrid (GMG) solver, demonstrating a speedup of approximately 3 to 8 times. Furthermore, our numerical investigation into the solver's capacity to generalize to untrained coefficient distributions reveals promising outcomes.
