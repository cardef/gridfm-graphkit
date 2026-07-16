---
type: paper
node_id: paper:gao2019_graph_unets
title: "Graph U-Nets"
authors: ["Hongyang Gao", "Shuiwang Ji"]
year: 2019
venue: "arXiv"
external_ids:
  arxiv: "1905.05178"
  doi: null
  s2: null
tags: ["pooling", "hierarchical-gnn", "architecture"]
added: 2026-07-06T07:25:51Z
---

# Graph U-Nets

## One-line thesis
Graph U-Nets: gPool selects top-k nodes via a trainable projection score and gUnpool restores them by stored index, enabling U-Net encoder-decoders on graphs.

## Problem / Gap
Existing graph pooling was either too coarse (global pooling collapses a graph to a single node) or connectivity-inconsistent (k-max pooling per feature map can retain different underlying nodes across channels), so no operation gave graphs a structure-preserving down/up-sampling analog to U-Net's.

## Method
gPool projects node features onto a trainable vector p (y = X p/‖p‖), keeps the k highest-scoring nodes and their induced subgraph, and multiplies the kept features by a sigmoid gate ỹ = σ(y(idx)) so the otherwise-discrete top-k selection stays differentiable and p receives gradient. gUnpool inverts this by scattering the k retained feature rows back into a zero-initialized N×C matrix at their original positions using the stored indices. Because pruning nodes also prunes their edges, the model takes the 2nd graph power (A² = A·A) after each gPool to reconnect surviving nodes within 2 hops before the next GCN aggregates over them. Four gPool+GCN encoder blocks and four gUnpool+GCN decoder blocks, joined by additive skip connections and a modified GCN self-loop weight (Â = A+2I instead of A+I), form the g-U-Nets architecture.

## Key Results
- Transductive node classification accuracy: 84.4±0.6% (Cora), 73.2±0.5% (Citeseer), 79.6±0.2% (Pubmed) — beats GCN by 2.9/2.9/0.6 points and GAT by 1.4/0.7/0.6 points.
- Inductive graph classification: 82.43% (D&D) and 77.68% (PROTEINS) beat DiffPool by 1.79 and 1.43 points; on COLLAB g-U-Nets scores 77.56%, below DiffPool-DET's 82.13% (which uses an auxiliary link-prediction task the authors say DiffPool needs for training stability).
- Ablations: removing gPool/gUnpool costs 2.3/1.6/0.5 points (Cora/Citeseer/Pubmed); removing the graph-power connectivity augmentation costs up to 0.7 points; the gPool/gUnpool layers add only 0.12% more parameters (75,737 vs 75,643 on Cora) for the +2.3-point Cora accuracy gain.

## Assumptions
- A single scalar projection per node (onto one trainable vector p) suffices to rank importance — no multi-head or context-dependent scoring.
- A GCN layer immediately before each gPool already aggregates first-order neighbor info, so 2-hop reconnection via the 2nd graph power is assumed sufficient to prevent isolated nodes after pruning.
- Pooling ratios are fixed and hand-chosen per dataset (absolute counts 2000/1000/500/200 for transductive; 90/70/60/50% for inductive), not learned or adaptive.

## Limitations / Failure Modes
- Evaluated only on classification — node classification under transductive learning, graph classification under inductive learning — with no regression, forecasting, or physical-simulation task, despite the introduction motivating link prediction as an application.
- Node removal in gPool is a hard, irreversible drop: unselected nodes reappear in gUnpool's output as zero feature rows, not reconstructed values: the only mitigation is the graph-power reconnection trick, not recovery of dropped information.
- On COLLAB the strongest baseline (DiffPool-DET, 82.13%) beats g-U-Nets (77.56%) by using an auxiliary link-prediction task; the paper attributes DiffPool's need for that auxiliary task to training instability but doesn't close the resulting accuracy gap without it.

## Reusable Ingredients
- Canonical learned top-k pooling baseline (gPool/gUnpool: scalar projection + sigmoid gate for differentiability + index-based restore) for hierarchy ablations.
- Graph connectivity augmentation via the k-th graph power (Aᵏ) after coarsening — a cheap, generic fix for the isolated-node problem any node-dropping pooling scheme faces, portable to other coarsening pipelines.
- Self-loop reweighting (Â = A+2I instead of A+I) as a near-free knob to bias GCN aggregation toward a node's own features.

## Open Questions
- Does learned task-driven pooling recover physically meaningful clusters (zones), or arbitrary ones — the paper never inspects which nodes gPool actually selects.
- Would learned (rather than fixed, hand-tuned) per-layer pooling ratios shift the depth-4 optimum found in the ablation, or the accuracy-vs-depth trade-off?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Serves as the generic learned-pooling baseline that any physics-grounded (Kron/Ward, zonal) coarsening must beat in a graphkit hierarchy ablation (gap G1); its homogeneous, untyped-graph design (single adjacency matrix, no bus/gen/load/line typing) also means it doesn't itself address heterogeneous-grid coarsening (gap G2) without adaptation.

## Abstract (original)

> We consider the problem of representation learning for graph data. Convolutional neural networks can naturally operate on images, but have significant challenges in dealing with graph data. Given images are special cases of graphs with nodes lie on 2D lattices, graph embedding tasks have a natural correspondence with image pixel-wise prediction tasks such as segmentation. While encoder-decoder architectures like U-Nets have been successfully applied on many image pixel-wise prediction tasks, similar methods are lacking for graph data. This is due to the fact that pooling and up-sampling operations are not natural on graph data. To address these challenges, we propose novel graph pooling (gPool) and unpooling (gUnpool) operations in this work. The gPool layer adaptively selects some nodes to form a smaller graph based on their scalar projection values on a trainable projection vector. We further propose the gUnpool layer as the inverse operation of the gPool layer. The gUnpool layer restores the graph into its original structure using the position information of nodes selected in the corresponding gPool layer. Based on our proposed gPool and gUnpool layers, we develop an encoder-decoder model on graph, known as the graph U-Nets. Our experimental results on node classification and graph classification tasks demonstrate that our methods achieve consistently better performance than previous models.
