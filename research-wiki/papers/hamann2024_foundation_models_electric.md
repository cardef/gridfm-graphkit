---
type: paper
node_id: paper:hamann2024_foundation_models_electric
title: "Foundation Models for the Electric Power Grid"
authors: ["Hendrik F. Hamann", "Thomas Brunschwiler", "Blazhe Gjorgiev", "Leonardo S. A. Martins", "Alban Puech", "Anna Varbella", "Jonas Weiss", "Juan Bernabe-Moreno", "Alexandre Blondin Massé", "Seong Choi", "Ian Foster", "Bri-Mathias Hodge", "Rishabh Jain", "Kibaek Kim", "Vincent Mai", "François Mirallès", "Martin De Montigny", "Octavio Ramos-Leaños", "Hussein Suprême", "Le Xie", "El-Nasser S. Youssef", "Arnaud Zinflou", "Alexander J. Belyi", "Ricardo J. Bessa", "Bishnu Prasad Bhattarai", "Johannes Schmude", "Stanislav Sobolevsky"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2407.09434"
  doi: null
  s2: null
tags: ["foundation-model", "power-grid", "position-paper"]
added: 2026-07-06T07:25:43Z
---

# Foundation Models for the Electric Power Grid

## One-line thesis
Argues for GridFM: GNN-based foundation models pretrained via masked autoencoding on power-flow data, with a phased roadmap toward GridFM–v0.

## Problem / Gap
The energy transition (DERs, inverter-based resources, electrification-driven demand/weather shifts), aging infrastructure, and cybersecurity threats are widening the gap between required grid computational capabilities (complexity/uncertainty) and what's available today (Fig. 1); wider AI/ML adoption in power systems has been impeded by scarce training data and the limited transferability of bespoke, task-specific models across applications and system configurations.

## Method
GridFM–v0 represents the grid as a graph — buses as nodes carrying (active power, reactive power, voltage magnitude, voltage angle) features, transmission lines/transformers as edges — and pretrains an encoder-decoder with a Masked Auto-Encoder (MAE) task that reconstructs masked node features from power-flow solutions (Fig. 6). The pretraining loss combines a Scaled Cosine Error (SCE) reconstruction term with a physics-informed L_powerflow term that penalizes violation of the power-flow equations. Downstream adaptation attaches a small task-specific decoder head fine-tuned with limited labeled data per application (power flow, OPF, state estimation, contingency analysis), rather than training a bespoke model per task/utility. The paper lays out a 4-phase roadmap — (1) collect/generate training data, (2) develop the MAE architecture and loss, (3) validate on industry data behind utility firewalls, (4) implement power-flow-based downstream applications — rather than presenting a trained model.

## Key Results
- No original experiments on GridFM–v0 itself; the paper cites/projects numbers from prior narrower work to justify the roadmap: GNN-based power-flow surrogates (cited prior work) already replicate traditional solver output with <1% error, and a cited OPF surrogate trained on synthetic data achieves a 2–4 order-of-magnitude speedup while keeping objective-value error <1%.
- GridFM–v0 itself is *projected* (not measured here) to reach a 3–4 order-of-magnitude speed-up over conventional Newton-Raphson solvers, a figure demonstrated only on IEEE cases up to 118 buses in the cited prior work, not on GridFM–v0.
- Worked contingency-analysis example: N-1 on the Western Interconnection takes ~5 min/scenario with conventional solvers; a 1000×-faster solver would let N-2 analysis (499,500 scenarios for a 1000-line grid) run in the time conventional solvers take for ~500 scenarios.
- Phase-1 data earmarked: a cited dataset of real-load power-flow solutions on 4 grid topologies, plus a cited 300,000-sample OPF dataset spanning ten topologies from 14 to 14,000 buses (including N-1 contingency solutions) for fine-tuning.

## Assumptions
- A graph with bus nodes (p, q, v, θ features) and line/transformer edges, pretrained via masked-feature reconstruction, is the right substrate/task for a grid FM.
- Sufficient diverse, curated grid data/topologies (synthetic now, real utility data later) can be assembled despite grid data being heterogeneous, siloed across ~3,200 US utilities, and privacy/security-constrained.
- Power-flow reconstruction is a rich-enough pretraining task that its learned representations transfer to OPF, state estimation, contingency analysis, forecasting, stability, and control downstream tasks.
- GNN-based solvers' near-linear scaling in bus count (versus quadratic scaling for Newton-Raphson) will hold up at real transmission-system scale.

## Limitations / Failure Modes
- No implementation or empirical results for GridFM–v0 itself in this paper; all quantitative claims are either citations to prior, narrower GNN power-flow work or forward-looking projections (Phases 1-4 are future work).
- Real, sensitive utility data is regulatorily hard to access; validation against operational grids is deferred to Phase 3 (behind utility firewalls), so practical accuracy on real grids is unverified at publication time.
- Grids evolve (network reinforcement, new DER-driven business models change topologies/operating profiles), which the paper says requires continual learning of the FM — but no continual-learning mechanism is specified.
- Trust/interpretability is flagged as unresolved: operators prefer interpretable models over black-box ones, and standards for evaluating AI-model performance on out-of-sample scenarios (e.g., ISO/IEC 24029-2) are only just emerging.

## Reusable Ingredients
- MAE-style pretraining formulation for power-flow graphs: bus nodes with (p, q, v, θ) features, line/transformer edges, masked-node-feature reconstruction as the self-supervised objective (Fig. 6).
- Hybrid loss pattern combining a generic reconstruction term (Scaled Cosine Error) with a physics-informed power-flow-equation term — a reusable recipe for grounding GNN surrogates in domain physics rather than pure data-fitting.
- Downstream-task taxonomy (Fig. 4: transient/dynamic stability, load flow, forecasting, control operations, electricity markets, system security, expansion planning, cybersecurity) mapped to a single pretrained backbone plus lightweight fine-tuned heads.
- Named candidate datasets for pretraining/fine-tuning: a real-load 4-topology power-flow dataset, PGLIB-OPF plus IEEE benchmarks under topology/load perturbation, and a 300k-sample OPF dataset (14–14,000 buses, with N-1 contingency solutions).

## Open Questions
- What architecture lets one pretrained model serve grids across the range spanned by the paper's own cited fine-tuning data (14 to 14,000 buses), when its own demonstrated speedup figures only go up to 118-bus IEEE cases — the exact question hierarchy answers? _(Note: the paper's full text supports a 14–14,000-bus range, not the "70,000" figure from a prior abstract-only pass of this page.)_
- Can a single MAE-pretrained backbone actually transfer, via lightweight fine-tuning heads, across the full task diversity in Fig. 4 (stability, forecasting, control, markets, planning, cybersecurity), or does each task cluster need its own pretraining/data modality?
- What continual-learning mechanism keeps a pretrained GridFM current as topology and DER-driven operating profiles evolve, without full retraining on proprietary utility data?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This repo (gridfm-graphkit) is a direct implementation attempt of the GridFM–v0 blueprint — GNN/GRIT-transformer backbones with PF/OPF/SE task heads and MVA normalizers mirror the paper's masked-autoencoder-plus-physics-loss-plus-fine-tuning-head recipe. The paper's own scale story (fine-tuning data up to 14,000 buses, but demonstrated speedups only on ≤118-bus IEEE cases) is exactly the untested size-transfer question in gap G3, and its flat (non-hierarchical) bus-and-line graph representation leaves gap G4's sub-quadratic/hierarchical-attention scaling problem for 10k+-bus grids fully open.

## Abstract (original)

> Foundation models (FMs) currently dominate news headlines. They employ advanced deep learning architectures to extract structural information autonomously from vast datasets through self-supervision. The resulting rich representations of complex systems and dynamics can be applied to many downstream applications. Therefore, FMs can find uses in electric power grids, challenged by the energy transition and climate change. In this paper, we call for the development of, and state why we believe in, the potential of FMs for electric grids. We highlight their strengths and weaknesses amidst the challenges of a changing grid. We argue that an FM learning from diverse grid data and topologies could unlock transformative capabilities, pioneering a new approach in leveraging AI to redefine how we manage complexity and uncertainty in the electric grid. Finally, we discuss a power grid FM concept, namely GridFM, based on graph neural networks and show how different downstream tasks benefit.
