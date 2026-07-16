---
type: paper
node_id: paper:ma2023_graph_inductive_biases
title: "Graph Inductive Biases in Transformers without Message Passing"
authors: ["Liheng Ma", "Chen Lin", "Derek Lim", "Adriana Romero-Soriano", "Puneet K. Dokania", "Mark Coates", "Philip Torr", "Ser-Nam Lim"]
year: 2023
venue: "PMLR 202 (2023) 23321-23337"
external_ids:
  arxiv: "2305.17589"
  doi: null
  s2: null
tags: ["graph-transformer", "rrwp", "architecture"]
added: 2026-07-06T07:25:45Z
---

# Graph Inductive Biases in Transformers without Message Passing

## One-line thesis
GRIT replaces message passing with learned RRWP-initialized relative positional encodings and joint node/node-pair attention updates, reaching SOTA on graph benchmarks.

## Problem / Gap
Graph Transformers with message-passing modules inherit MPNN pathologies (over-smoothing, over-squashing, expressivity limits) and diverge architecturally from Transformers in other domains, while those without message-passing lack sufficient inductive bias and underperform MPNNs on small datasets like ZINC (12k graphs) — even though non-message-passing Transformers dominate large-scale benchmarks like PCQM4Mv2 (3.7M graphs).

## Method
GRIT initializes relative positional encodings from K-step Relative Random Walk Probabilities P_{i,j}=[I,M,...,M^{K-1}] (M=D⁻¹A), then evolves them end-to-end via an elementwise MLP that is updated jointly with node/pair states inside the attention layer, rather than fixed as in prior PEs. Its attention computes node-pair states ê_{i,j}=σ(ρ((W_Q x_i ⊙ W_K x_j) ⊙ W_Ew e_{i,j} + W_Eb e_{i,j})) with a signed-square-root activation ρ that stabilizes large-magnitude inputs, then updates both node representations x_i and pair representations e_{i,j} every layer — unlike attention mechanisms that use positional encodings only as a static bias. A per-layer degree scaler (θ₁·x + θ₂·log(1+dᵢ)·x) injects degree information, and BatchNorm replaces LayerNorm because the paper proves (Prop. 3.3) LayerNorm makes sum-aggregated, degree-scaled, and mean-aggregated node representations numerically indistinguishable, erasing the degree signal.

## Key Results
- Best mean performance on 4 of 5 BenchmarkingGNNs datasets with statistical significance (ZINC MAE 0.059±0.002, CIFAR10 76.468±0.881%, PATTERN 87.196±0.076%, CLUSTER 80.026±0.277%); on MNIST it is 2nd-best (98.108±0.111%), not significantly different from EGT's best 98.173±0.087%.
- Best mean on both LRGB tasks (Peptides-func AP 0.6988±0.0082, Peptides-struct MAE 0.2460±0.0012) and on ZINC-full (MAE 0.023±0.001 vs. SignNet's 0.024±0.003); on PCQM4Mv2 (3.7M graphs) matches GraphGPS-medium (0.0859 vs. 0.0858 valid MAE) using fewer parameters (16.6M vs. 19.4M) and beats Graphormer (0.0864 MAE, 48.3M params) — no hyperparameter search was run for this dataset.
- Provably more expressive than shortest-path-distance (SPD) encodings: GD-WL with RRWP distinguishes the Dodecahedron and Desargues graphs (Prop. 3.2), which GD-WL with SPD cannot; in a synthetic k-hop-attention probe GRIT reaches MAE ≤0.007 / R²≥0.961 for k=1,2,3 — an order of magnitude better than the next-best baseline (Graphormer+SPDPE, R² 0.67–0.80).

## Assumptions
- Per-graph size stays small-to-medium: across all 8 benchmarks (Table 8), average nodes/graph range 14.1–150.9, including the 3.7M-graph PCQM4Mv2 (avg. 14.1 nodes/graph) — the paper's "large-scale" means many graphs, not large individual graphs.
- Dense O(|V|²) node-pair attention and O(K|V||E|) RRWP precompute (App. B.6) are assumed affordable per training batch (batch size ≤256 in all experiments).
- Batched training with meaningful batch statistics is assumed, since BatchNorm (not LayerNorm) is required to keep degree information from being normalized away (Prop. 3.3).

## Limitations / Failure Modes
- The conclusion explicitly names GRIT's O(N²) scaling for updating pair representations as an open limitation, alongside a "lack of upper bounds on expressive power" — only lower-bound expressiveness results (SPD/propagation-matrix approximation, GD-WL separation) are proven.
- No hyperparameter search was run for PCQM4Mv2 ("due to the limit of time"); values were inherited from GraphGPS, so the reported PCQM4Mv2 number may understate GRIT's ceiling.
- Every evaluated dataset has small per-graph node counts (max avg. 150.9 nodes, Peptides); there is no evaluation at the thousands-of-nodes scale relevant to real transmission grids.

## Reusable Ingredients
- RRWP (P=[I,M,...,M^{K-1}], M=D⁻¹A) as a single positional-encoding family: Prop. 3.1 gives exact θ-parameterizations for an MLP over P to recover shortest-path distances, sum/mean aggregation, K-truncated PageRank, or K-truncated heat kernels.
- Joint per-layer update of node AND pair/edge representations (not a static PE-as-bias) — a general attention design pattern portable to any structure-conditioned attention model.
- Degree-scaler + BatchNorm recipe (Prop. 3.3 proof) for preserving degree information through normalization — applicable to any GNN/Transformer where LayerNorm would otherwise erase it.

## Open Questions
- Can the O(N²) pair-representation update be made sub-quadratic (sparsified/hierarchical) without losing the expressiveness that dense pair states provide?
- What are the upper bounds on GRIT's expressive power? The paper proves lower bounds (SPD and propagation-matrix approximation, GD-WL separation) but explicitly leaves upper bounds open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This is the architecture graphkit implements verbatim in `models/grit_transformer.py`; the paper's own admitted O(N²) pair-update scaling and its benchmarks' small per-graph node counts (never exceeding ~151 nodes, Table 8) are exactly the wall gap G4 identifies for 10k+-bus transmission grids, and the paper's explicit lack of an expressiveness upper bound leaves no theoretical guarantee to fall back on when approximating a sub-quadratic variant.

## Abstract (original)

> Transformers for graph data are increasingly widely studied and successful in numerous learning tasks. Graph inductive biases are crucial for Graph Transformers, and previous works incorporate them using message-passing modules and/or positional encodings. However, Graph Transformers that use message-passing inherit known issues of message-passing, and differ significantly from Transformers used in other domains, thus making transfer of research advances more difficult. On the other hand, Graph Transformers without message-passing often perform poorly on smaller datasets, where inductive biases are more crucial. To bridge this gap, we propose the Graph Inductive bias Transformer (GRIT) -- a new Graph Transformer that incorporates graph inductive biases without using message passing. GRIT is based on several architectural changes that are each theoretically and empirically justified, including: learned relative positional encodings initialized with random walk probabilities, a flexible attention mechanism that updates node and node-pair representations, and injection of degree information in each layer. We prove that GRIT is expressive -- it can express shortest path distances and various graph propagation matrices. GRIT achieves state-of-the-art empirical performance across a variety of graph datasets, thus showing the power that Graph Transformers without message-passing can deliver.
