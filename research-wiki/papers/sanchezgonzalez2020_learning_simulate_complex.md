---
type: paper
node_id: paper:sanchezgonzalez2020_learning_simulate_complex
title: "Learning to Simulate Complex Physics with Graph Networks"
authors: ["Alvaro Sanchez-Gonzalez", "Jonathan Godwin", "Tobias Pfaff", "Rex Ying", "Jure Leskovec", "Peter W. Battaglia"]
year: 2020
venue: "arXiv"
external_ids:
  arxiv: "2002.09405"
  doi: null
  s2: null
tags: ["physics-simulation", "gnn", "gns"]
added: 2026-07-06T07:25:49Z
---

# Learning to Simulate Complex Physics with Graph Networks

## One-line thesis
A single encode-process-decode graph network (GNS), with one fixed hyperparameter set, learns to simulate fluids, rigid solids, and deformables across many material domains.

## Problem / Gap
Traditional simulators need separate hand-engineered algorithms per material (SPH, PBD, MPM), and prior learned simulators (DPI, CConv) were material-specific or needed extra rigidity-constraint machinery; no single learned model handled fluid, rigid, and deformable interactions together with one fixed architecture.

## Method
GNS builds a particle graph via a fixed connectivity radius (k-d tree neighbor search, recomputed each rollout step) and runs an encode-process-decode GN: a relative-position ENCODER (2-hidden-layer MLPs, 128-d latents; uses displacement + magnitude rather than absolute coordinates for spatial invariance) feeds a PROCESSOR of M=10 message-passing GN blocks with unshared per-step parameters, and a DECODER MLP predicts per-particle average acceleration, integrated with semi-implicit Euler. Training injects random-walk Gaussian noise (σ=3e-4) into input velocity histories so the one-step training distribution better matches noisy self-generated rollout inputs. Unlike DPI, which needs a specialized hierarchical mechanism to keep rigid bodies rigid, GNS treats material type as a plain per-particle input feature; unlike CConv's radially-interpolated convolution kernels, GNS's deeper MLP edge/node functions and multi-step message passing give it more modeling capacity.

## Key Results
- Trained on single-step transitions from ~1k-20k-particle, 300-2000-step trajectories; generalizes to 5000-step rollouts and domains 32x larger in spatial extent, reaching 85k particles (34x more than the ~2.5k used in training).
- One-step/rollout MSE (Table 1): WATER-3D 8.66e-9 / 10.1e-3, SAND-3D 1.42e-9 / 0.554e-3, GOOP-3D 1.32e-9 / 0.618e-3; rollout MSE beats the best-tuned CConv variant across all six domains compared head-to-head, and on CONTINUOUS a model trained with friction angle [30°,55°] held out still predicts accurately in that gap.
- Inference (incl. neighbor search) costs 51%-345% of the ground-truth simulator's per-step time across domains; most of that is k-d-tree neighbor computation, not GN inference.

## Assumptions
- Locality: particle interactions modeled only within a fixed connectivity radius (~10-20 neighbors per particle); longer-range effects must emerge from stacking M message-passing steps.
- Markovian one-step training objective (L1-step): the paper explicitly argues whole-trajectory training risks letting the model exploit implicit memory that wouldn't generalize (its pendulum thought experiment).
- One fixed hyperparameter set (10 MP steps, unshared PROCESSOR params, relative encoder, noise σ=3e-4) is assumed to transfer across all material domains without per-domain retuning.

## Limitations / Failure Modes
- FLUIDSHAKE-BOX: a rigid block vigorously shaken over 1500 rollout steps gradually deforms, because the model must infer/maintain the block's shape from only C=5 initial frames with no explicit rigidity constraint.
- A bad-seed model trained on GOOP sometimes predicts a blob sticking to the wall instead of falling — attributed to insufficient training exposure to stick-vs-fall (static friction/adhesion) transitions.
- Restricted to mesh-free particle representations; extension to meshes, incorporation of stronger physical priors (e.g., Hamiltonian mechanics), and efficient GNS implementations are left as future work, and no inverse-problem/differentiable-simulator results are shown despite being motivated in the conclusion.

## Reusable Ingredients
- Random-walk Gaussian noise injected into the input velocity history (best of 4 schemes tested) — a general recipe for closing the train/rollout distribution gap in any autoregressive, one-step-trained simulator.
- Relative/frame-invariant encoder (mask absolute position; encode displacement + magnitude) as a portable inductive bias for spatially-invariant dynamics — measurably beats an absolute-position encoder (Fig. 4i,j).
- Unshared per-step PROCESSOR parameters (deep-net-like rather than weight-tied/RNN-like) give better rollout accuracy than shared parameters at negligible overfitting cost — a concrete, transferable architecture default.

## Open Questions
- Does GNS's demonstrated extrapolation (34x particles, 32x spatial extent) hold when interactions are heterogeneous (typed nodes/edges) and globally coupled rather than local and materially homogeneous, as in power grids?
- Can the one-step training objective's implicit-memory pitfall (the paper's own pendulum argument) silently arise in other L1-step-trained domains, and how would one detect it before deployment?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Direct architectural ancestor of graphkit's GNS_heterogeneous; but its largest extrapolation result (34x particles, 32x spatial extent) is demonstrated only for spatially-local, single-node-type materials under a fixed connectivity radius — whether that transfers to graphkit's heterogeneous, non-spatial, globally-coupled grid graphs is exactly the open question in gap G3, and the paper's reliance on local connectivity radius rather than any coarsening/hierarchy is also why it offers no template for G2's multiscale problem.

## Abstract (original)

> Here we present a machine learning framework and model implementation that can learn to simulate a wide variety of challenging physical domains, involving fluids, rigid solids, and deformable materials interacting with one another. Our framework---which we term "Graph Network-based Simulators" (GNS)---represents the state of a physical system with particles, expressed as nodes in a graph, and computes dynamics via learned message-passing. Our results show that our model can generalize from single-timestep predictions with thousands of particles during training, to different initial conditions, thousands of timesteps, and at least an order of magnitude more particles at test time. Our model was robust to hyperparameter choices across various evaluation metrics: the main determinants of long-term performance were the number of message-passing steps, and mitigating the accumulation of error by corrupting the training data with noise. Our GNS framework advances the state-of-the-art in learned physical simulation, and holds promise for solving a wide range of complex forward and inverse problems.
