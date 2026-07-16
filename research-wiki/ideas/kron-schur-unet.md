---
type: idea
node_id: idea:kron-schur-unet
title: "Kron-Schur Coarsening for Scalable Power-Grid GNNs"
stage: active
outcome: pending
added: 2026-07-06T13:50:22Z
based_on: ["paper:gao2019_graph_unets", "paper:pham2024_reduced_optimal_power"]
target_gaps: ["gap:G1", "gap:G2"]
tags: []
---

# Kron-Schur Coarsening for Scalable Power-Grid GNNs

> **Historical research-wiki snapshot.** The score, three-seed gate, budget ladder, errata, and removed technical-plan references below describe the superseded 2026-07-10 design. Current method and launch authority live only in `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/EXPERIMENT_PLAN.md`, and `refine-logs/EXPERIMENT_TRACKER.md`.

**stage:** `superseded`  ·  **outcome:** `archived`

## Thesis
v3.2 FINAL proposal (**READY 9.14/10**, research-refine run 3 of 2026-07-10: 3 rounds vs Codex gpt-5.6-sol at max reasoning effort, 7.50 → 8.82 → 9.14; supersedes v2 9.28 of 2026-07-09 and the un-reviewed v3 coherence revision): **two deliberately separate maps from one construction**. An *exact physical path* — unsparsified P_phys/Yred_phys, boundary-conditioned HELM series at a **fixed** per-model order (default helm2; no inference-time k-dial), coefficient equations to solver tolerance, BoundaryConditionProjector enforcing known PV/slack conditions, strict reciprocal with deterministic k=0 intention-to-treat fallback — used only for reconstruction, residuals, diagnostics. A *sparse latent path* — P_latent + thresholded coarse edges under a label-free topology-only policy — moves hidden features, no exactness claim. τ-consistent coarse supervision corrected to a masked boundary-POWER residual (Î_B(k)=Yred·V_B+Y_BI·Σc_n, Ŝ_B=V_B⊙conj(Î_B); the technical plan's version equated power with current — erratum pending). v1 affine = `affine_flat_exact_germ` (w0=1 control, NOT order-1); prototype = `affine_flat_legacy_sparse` (archival). Primary gate = anchor-faithful **speed superiority at a joint accuracy target vs the flat-frontier lower envelope** (capped time-to-target + warm latency, per-seed deterministic 3-seed rule, defined censoring). Mechanism claims from three clean contrasts: A1 physical channel, A2 Kron package vs matched METIS pooling, A3 Pareto vs coarse-only. Fail-closed R014 (mandatory minimum C1+A1–A3 must fit budget). Full spec: refine-logs/FINAL_PROPOSAL.md

## Key risks
Declared falsifier 1: the flat-frontier envelope (any measured depth/width) may tie KS on the primary speed gate at case2000 (pivot pre-committed to diagnostic paper). Declared falsifier 2 (A3): coarse-only + physical reconstruction may Pareto-tie the full model (pivot to the amortized boundary-value model). Prototype exactness claims were invalid (sparsified germ) and the R006 gate was label leakage — pilot ceilings (77x/10x) must be re-established with exact operators under strict policy; M0 = `affine_flat_legacy_sparse` engineering evidence only. Predicted-V_B error may dominate (attribution decomposition e_B / e_phys|B* / e_phys|B-hat / e_final preregistered; A3 decides the pivot). Standard-validation fallback-rate ceiling gates C1; stress-slice rate descriptive with alarm threshold. Mandatory minimum (C1+A1–A3) may not fit ~230 GPU-h — R014 fails closed, preregistered descoping ladder (trimmables → case2000-only manifest with scaling/foundation claims auto-disabled). A2 (METIS package control) or the Performer-HGNS baseline may tie → cheap global communication, not electrical reduction, explains the gain. Coarse-denser-than-fine / |B|/N > 40% grids excluded by rule, reported separately. Accuracy claims ≤2k; 9241 feasibility-only (k∈{0,1}, backend stated). Fixed-Y scoping (SMW/LODF successor paper, no O(n_I) cost claim without analysis). Open R014 preconditions: EXPERIMENT_PLAN amendment + two technical-plan errata (τ residual power/current; primary-gate direction).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
