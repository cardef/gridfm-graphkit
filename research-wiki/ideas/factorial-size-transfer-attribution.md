---
type: idea
node_id: idea:factorial-size-transfer-attribution
title: "Factorial attribution of zero-shot size-transfer failure in learned PF/OPF"
stage: proposed
outcome: pending
added: 2026-07-06T12:24:52Z
based_on: ["paper:rivera2025_benchmark_dataset_power", "paper:lin2023_powerflownet_power_flow", "paper:piloto2024_canos_fast_scalable"]
target_gaps: ["gap:G3"]
tags: []
---

# Factorial attribution of zero-shot size-transfer failure in learned PF/OPF

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Intervention-level attribution of the small-to-large zero-shot failure across normalization (incl. zero-gradient oracle stats refit), receptive range, input marginal shift, and Ward-interpolated topology family; plus fine-tuning decomposition (normalizer-refit vs affine vs full FT). Expect normalization+marginal shift to dominate, receptive field to matter little.

## Key risks
Factors interact (factorial design needed); oracle arms may be called artificial; must cite Yehudai ICML 2021 and spectral size-gen work early and avoid causal-proof claims. Codex triage rank 1, novelty 8/10.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
