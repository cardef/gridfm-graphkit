---
type: paper
node_id: paper:pham2022_reduced_optimal_power
title: "Reduced Optimal Power Flow Using Graph Neural Network"
authors: ["Thuan Pham", "Xingpeng Li"]
year: 2022
venue: "2022 North American Power Symposium (NAPS)"
external_ids:
  arxiv: "2206.13591"
  doi: null
  s2: null
tags: ["opf", "power-grid", "gnn"]
added: 2026-07-06T07:25:52Z
---

# Reduced Optimal Power Flow Using Graph Neural Network

## One-line thesis
A 4-layer XENet GNN classifies congested lines from bus/branch features and topology, shrinking the OPF's monitored-line set and cutting solve time up to 18%.

## Problem / Gap
Prior ML-for-OPF models (NN, CNN, and one GNN attempt) either ignore network topology or omit power-flow/line-rating constraints from their formulation entirely, and operators today pick the monitored-line subset as a static list derived from historical solutions and load profiles rather than dynamically per load profile.

## Method
The model stacks 4 XENet graph-convolution layers (from the Spektral library) followed by a dense layer; it consumes bus features (nodal load, max/min generation, number of connected branches, bus type: load/generator/slack), branch features (line reactance, line rating), and the adjacency matrix, and outputs a per-branch one-hot congestion classification — framed as an edge-level classification task. Training labels come from solving classic OPF via Pyomo on the IEEE 73-bus system across 20,000 samples (load profile varied ±10% of the base case, 80/10/10 train/val/test split), with a branch labeled congested if its solved flow exceeds a tunable loading threshold (tested at 70-95% of line rating). Only branches predicted congested get their line-limit constraint (Eq. 4a) enforced in the resulting reduced OPF (ROPF), instead of imposing it on every line (Eq. 4) as in full OPF. As an ablation, separate NN and CNN models with matching 4-layer depth per bus/branch stream but no adjacency matrix (i.e., no topology signal) are trained identically for comparison.

## Key Results
- GNN beats NN and CNN baselines at every tested loading threshold (70-95% of line rating): e.g., 1.47% vs. 1.76% (NN) / 1.67% (CNN) prediction error at 70%, and 2.71% vs. 3.18% / 3.14% at 95% (Table I).
- At the paper's recommended 85% threshold: only 17.67% of lines are monitored, 2000-sample OPF solve time drops from 145.87s (full OPF) to 122.38s (16% saving), and just 0.4% of samples (8/2000) have a branch that violates the true line rating (Tables II-III).
- Pushing to a 95% threshold buys ~18% time saving (118.97s) but raises the violation rate to 41.35% of samples, so the authors explicitly reject that setting; where violations do occur, total generation cost actually drops by <1% (the ROPF solution is cheaper but infeasible against the true line limit).

## Assumptions
- Single fixed IEEE 73-bus topology; only nodal load levels vary (±10% around the base case) across all 20,000 samples — no topology or contingency changes are modeled.
- Congestion status can be predicted purely from pre-solve static features (nodal load, generation limits, bus type, branch reactance/rating, adjacency) without running the OPF itself.
- A single global loading-threshold percentage of line rating is an adequate proxy for "must be monitored," rather than a per-line or operator-specific tolerance.

## Limitations / Failure Modes
- Type-2 (false-negative) misclassifications — a heavily loaded line predicted as light — are shown to correlate almost exactly with the line-rating violations that leak through ROPF (Figs. 10-13); the method carries no feasibility guarantee.
- At a 95% loading threshold, which gives the largest monitored-line reduction and speedup, 41.35% of the 2000 test ROPF solutions still violate a true line rating — deemed unacceptable, which restricts the practical threshold range to around 85%.
- Evaluated only on a single fixed-topology IEEE 73-bus system with load-only perturbations; generalization to topology or contingency changes is left as future work (the paper's own stated next step is N-1 contingency screening), not tested here.

## Reusable Ingredients
- Framing constraint/monitored-line-subset selection as an edge-level GNN classification task (predict which constraints will bind, then reduce the optimization problem to just those) — a generalizable "active-constraint screening via GNN" pattern.
- XENet (Spektral) as the message-passing layer, which jointly propagates node and edge features while retaining the adjacency matrix at every layer — contrasted via a controlled ablation against NN/CNN baselines that process bus/branch streams separately with no topology signal at all.
- Data-generation recipe: resolve classic OPF via Pyomo across many perturbed load profiles (±10% of base case) to cheaply mint large labeled sets, then derive binary congestion labels from a tunable %-of-rating threshold.

## Open Questions
- How does screening accuracy degrade under topology changes or N-1 contingencies, which the paper's own future-work section flags as untested?
- Is there a principled way to select the loading threshold that balances speedup against the feasibility risk from type-2 errors, rather than manually sweeping it as done here?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This is the flat (non-hierarchical) precursor to [[pham2024_reduced_optimal_power]], the paper gap_map's G1 actually cites for virtual-node hierarchy — this 2022 version has no hierarchical or electrically-grounded reduction at all, only a plain XENet edge classifier over the full bus-branch graph, so it establishes the ROPF/constraint-screening problem rather than bearing on G1 directly. Its controlled ablation (GNN with adjacency vs. NN/CNN without) is still a useful data point for graphkit: it is direct evidence that topology-aware message passing beats topology-blind baselines on a grid-electrical prediction task, reinforcing the case for graphkit's own GRIT/GNS architectures over flat MLP/CNN baselines.

## Abstract (original)

> OPF problems are formulated and solved for power system operations, especially for determining generation dispatch points in real-time. For large and complex power system networks with large numbers of variables and constraints, finding the optimal solution for real-time OPF in a timely manner requires a massive amount of computing power. This paper presents a new method to reduce the number of constraints in the original OPF problem using a graph neural network (GNN). GNN is an innovative machine learning model that utilizes features from nodes, edges, and network topology to maximize its performance. In this paper, we proposed a GNN model to predict which lines would be heavily loaded or congested with given load profiles and generation capacities. Only these critical lines will be monitored in an OPF problem, creating a reduced OPF (ROPF) problem. Significant saving in computing time is expected from the proposed ROPF model. A comprehensive analysis of predictions from the GNN model was also made. It is concluded that the application of GNN for ROPF is able to reduce computing time while retaining solution quality.
