---
type: paper
node_id: paper:chen2023_physicsguided_residual_learning
title: "Physics-guided Residual Learning for Probabilistic Power Flow Analysis"
authors: ["Kejun Chen", "Yu Zhang"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2301.12062"
  doi: null
  s2: null
tags: ["residual-learning", "power-flow"]
added: 2026-07-07T17:21:01Z
---

# Physics-guided Residual Learning for Probabilistic Power Flow Analysis

## One-line thesis
Physics-guided residual learning for probabilistic PF: linear physics shortcut layer (model-based init) + NN residual - the 'learn only the nonlinear correction' pattern at whole-solver level.

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

> Probabilistic power flow (PPF) analysis is critical to power system operation and planning. PPF aims at obtaining probabilistic descriptions of the state of the system with stochastic power injections (e.g., renewable power generation and load demands). Given power injection samples, numerical methods repeatedly run classic power flow (PF) solvers to find the voltage phasors. However, the computational burden is heavy due to many PF simulations. Recently, many data-driven based PF solvers have been proposed due to the availability of sufficient measurements. This paper proposes a novel neural network (NN) framework which can accurately approximate the non-linear AC-PF equations. The trained NN works as a rapid PF solver, significantly reducing the heavy computational burden in classic PPF analysis. Inspired by residual learning, we develop a fully connected linear layer between the input and output in the multilayer perceptron (MLP). To improve the NN training convergence, we propose three schemes to initialize the NN weights of the shortcut connection layer based on the physical characteristics of AC-PF equations. Specifically, two model-based methods require the knowledge of system topology and line parameters, while the purely data-driven method can work without power grid parameters. Numerical tests on five benchmark systems show that our proposed approaches achieve higher accuracy in estimating voltage phasors than existing methods. In addition, three meticulously designed initialization schemes help the NN training process converge faster, which is appealing under limited training time.
