---
type: paper
node_id: paper:fortunato2022_multiscale_meshgraphnets
title: "MultiScale MeshGraphNets"
authors: ["Meire Fortunato", "Tobias Pfaff", "Peter Wirnsberger", "Alexander Pritzel", "Peter Battaglia"]
year: 2022
venue: "2nd AI4Science Workshop at the 39th International Conference on Machine Learning (ICML), 2022"
external_ids:
  arxiv: "2210.00612"
  doi: null
  s2: null
tags: ["multiscale", "hierarchical-gnn", "physics-simulation", "mesh"]
added: 2026-07-06T07:25:33Z
---

# MultiScale MeshGraphNets

## One-line thesis
MS-MGN adds coarse-mesh message passing to MeshGraphNets, restoring spatial convergence and cutting error versus MGN at equal message-passing-step budgets.

## Problem / Gap
MeshGraphNets' flat message passing has a fixed propagation speed set by mesh edge length, so finer meshes need proportionally more MP steps to carry information the same physical distance; empirically its one-step error stagnates below a resolution threshold instead of tracking the classical solver's spatial-convergence curve. Adding more MP steps only shifts this threshold left and hits a practical ceiling (25 mps was the most the authors could train on a single GPU, and the error still did not converge), so the field lacked a way to model fast-acting/non-local effects on high-resolution meshes without unbounded compute growth.

## Method
MS-MGN keeps the Encode-Process-Decode framework (Sanchez-Gonzalez et al., 2020) but processes two graphs — the fine input mesh G^h and an auxiliary coarse mesh G^l of the same domain — linked by downsampling/upsampling graphs (G^down, G^up) built by locating, for each node on one mesh, the enclosing triangle on the other mesh and connecting it to that triangle's three corner nodes. All four graph types (high-res, low-res, downsample, upsample updates; Eqs. 1-4) have their own edge/node MLP update functions, and the processor schedules blocks of these updates as a V-cycle inspired by multigrid methods — e.g. 'p=1H 11L 1H (U=1,D=1)' (15 total steps) or two V-cycles in sequence, 'p=3H 6L 3H 6L 3H (U=2,D=2)' (25 steps) — motivated by treating MP-GNN updates as Gauss-Seidel-like local smoothing that needs a coarser level to fix global error components cheaply, since a coarse-graph update touches far fewer nodes/edges than a fine-graph one. Inputs, outputs, and the training loss remain defined only on the fine mesh G^h; the coarse mesh only encodes geometric/node-type features, not the simulated field variables, and is generated (via COMSOL) with the same adaptive mesh generator as the fine mesh, constrained to a larger minimum edge length. A separate, orthogonal fix (Section 3) trains the same MGN/MS-MGN architectures on "high-accuracy labels" — ground truth interpolated from a higher-resolution reference simulation rather than same-resolution solver output — to let the model learn subgrid closure effects.

## Key Results
- At matched message-passing budgets, MS-MGN (25 mps) tracks the reference COMSOL spatial-convergence curve down to edge_min ≈ 1.6×10⁻³, while MGN at the same 25 mps stagnates well above it (Fig. 5); MGN alone needed up to 25 mps — the largest the authors could train on a single GPU — just to shift, not close, its divergence point (Fig. 4).
- MS-MGN + high-accuracy labels (25 mps) stays below the reference solver's own error at every tested resolution (Fig. 8); MGN + high-accuracy labels alone only beats the solver above edge length 0.0016 and still diverges below it (Fig. 7).
- A uniform-grid coarse mesh (à la Lino et al., 2021) is strictly worse than the paper's default conformal/adaptive coarse mesh on both 1-step and rollout (N>1) error (Fig. 10); MS-MGN also reduces but does not eliminate rollout error accumulation for edge_min < 1.5×10⁻³ (Fig. 9).

## Assumptions
- A coarse mesh generated with the same adaptive mesh generator as the fine mesh (just a larger minimum edge length) is available, so it conforms to domain boundaries and mirrors the fine mesh's relative node density.
- Only Eulerian dynamics with mesh edges are modeled; world edges (contact/collision handling, as used in the original MGN for cloth) are explicitly omitted.
- Experiments use 2D triangular meshes; extension to hexahedral or tetrahedral elements is stated as straightforward but not evaluated.

## Limitations / Failure Modes
- Error accumulation over long rollouts is "substantially ameliorated" but explicitly "not resolved" — MS-MGN still drifts from ground truth for edge_min < 1.5×10⁻³ (Fig. 9); the authors call reducing rollout error accumulation "an important research direction" left open.
- Only 2 hierarchy levels (fine + coarse) are evaluated; the conclusion names exploring more levels in the mesh hierarchy as an "interesting extension" not attempted here.
- Evaluated on a single problem family — 2D incompressible CylinderFlow (Kármán vortex street past one cylinder) — with authors flagging "more complex geometries or dynamics" as future work; no generalization evidence beyond this domain.
- Models are trained on next-step prediction with a fixed time step; adaptive time-stepping, sequence models, and the role of training noise in error accumulation are named as unaddressed future directions.

## Reusable Ingredients
- The 4-graph V-cycle message-passing schedule (H/L/U/D block notation, e.g. `3H 6L 3H 6L 3H` for two stacked V-cycles) as a general multigrid-inspired template for scheduling fine/coarse GNN updates.
- "High-accuracy labels" — training on targets interpolated from a higher-resolution reference simulation rather than same-resolution solver output — is architecture-agnostic and demonstrably teaches subgrid closure effects; portable to any learned-simulator setting independent of MS-MGN.
- Graph Fourier analysis of the per-node error signal (via graph-Laplacian eigenvectors, à la Shuman et al. 2013) as a diagnostic to check whether an architectural change specifically reduces slowly-varying/long-range error components rather than just aggregate MSE.

## Open Questions
- Does stacking more than 2 hierarchy levels keep improving accuracy/cost at even higher resolutions, or does it hit diminishing returns the way adding plain MP steps did for baseline MGN?
- Can rollout error accumulation be fixed by adaptive time-stepping, sequence models, or training noise, independent of the multiscale architecture — and how to build a coarse level when there is no spatial mesh at all (e.g., heterogeneous, non-spatial grid graphs)?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Directly the template gap G2 flags: MS-MGN's V-cycle fine/coarse schedule and containment-based downsample/upsample graphs assume a spatially embedded, boundary-conforming coarse mesh — and the paper's own ablation (Fig. 10, uniform grid vs. conformal coarse mesh) shows *how* the coarse level is built matters, not just that one exists. Realizing this template for `GNS_heterogeneous` requires substituting an electrically grounded coarsening (zones, Kron/Ward reduction) for the triangle-containment maps, since graphkit's HeteroData grids have no spatial coarse mesh to fall back on.

## Abstract (original)

> In recent years, there has been a growing interest in using machine learning to overcome the high cost of numerical simulation, with some learned models achieving impressive speed-ups over classical solvers whilst maintaining accuracy. However, these methods are usually tested at low-resolution settings, and it remains to be seen whether they can scale to the costly high-resolution simulations that we ultimately want to tackle. In this work, we propose two complementary approaches to improve the framework from MeshGraphNets, which demonstrated accurate predictions in a broad range of physical systems. MeshGraphNets relies on a message passing graph neural network to propagate information, and this structure becomes a limiting factor for high-resolution simulations, as equally distant points in space become further apart in graph space. First, we demonstrate that it is possible to learn accurate surrogate dynamics of a high-resolution system on a much coarser mesh, both removing the message passing bottleneck and improving performance; and second, we introduce a hierarchical approach (MultiScale MeshGraphNets) which passes messages on two different resolutions (fine and coarse), significantly improving the accuracy of MeshGraphNets while requiring less computational resources.
