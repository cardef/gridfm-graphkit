---
type: paper
node_id: paper:lovett2024_opfdata_largescale_datasets
title: "OPFData: Large-scale datasets for AC optimal power flow with topological perturbations"
authors: ["Sean Lovett", "Miha Zgubic", "Sofia Liguori", "Sephora Madjiheurem", "Hamish Tomlinson", "Sophie Elster", "Chris Apps", "Sims Witherspoon", "Luis Piloto"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2406.07234"
  doi: null
  s2: null
tags: ["dataset", "opf", "power-grid", "scalability"]
added: 2026-07-06T07:25:38Z
---

# OPFData: Large-scale datasets for AC optimal power flow with topological perturbations

## One-line thesis
300k solved AC-OPF examples per grid across 10 PGLib-OPF grids (14–13,659 buses), with FullTop and N-1 topology-perturbation variants.

## Problem / Gap
No large-scale open AC-OPF dataset existed: OPF-Learn tops out at 118 buses and TAS-97 provides only 7,284 samples on a 97-bus grid, blocking training of high-capacity data-driven OPF solvers.

## Method
Each of 10 PGLib-OPF base cases (14 to 13,659 buses; Table 1) is solved 300k times with PowerModels.jl + Ipopt/MUMPS on the full ACPPowerModel formulation, producing two variants per grid: **FullTop** (fixed topology; active/reactive load at each bus scaled by an independent uniform draw from [0.8, 1.2]) and **N-1** (same load perturbation, plus with probability 0.5 drop one randomly chosen generator or with probability 0.5 drop one randomly chosen line/transformer — never a reference-bus generator, never a component whose removal disconnects the graph; infeasible results are discarded). Each example is stored as JSON with a heterogeneous graph schema: typed nodes (bus, generator, load, shunt) and typed edges (ac_line, transformer, generator_link, load_link, shunt_link), with PyTorch Geometric loading utilities provided. This covers 4 of the 5 "core variability factors" for ML-OPF datasets identified by Popli et al. [17] (load distribution, load power factor, generator outages, line outages), explicitly leaving generator-cost perturbation to future iterations.

## Key Results
- 300k examples per grid, 10 grids × 2 variants (FullTop, N-1) = 20 datasets, 14 to 13,659 buses (Table 1) — orders of magnitude larger than OPF-Learn (≤118 buses) or TAS-97 (7,284 samples, 97 buses).
- Canonical split is 90/5/5 (270k/15k/15k) train/validation/test, i.i.d., taken sequentially by example index.
- Perturbation set covers 4 of the 5 core variability factors identified by Popli et al. [17]; generator-cost perturbation is left to future work.

## Assumptions
- Ground truth is the PowerModels.jl AC-OPF solution via Ipopt/MUMPS on the nonconvex, non-mixed-integer ACPPowerModel formulation (no unit-commitment or discrete variables).
- Grids are PGLib-OPF synthetic benchmark cases (IEEE, GOC, SDET, RTE, PEGASE test systems), not real utility topologies.
- Load perturbation is a single independent uniform-in-[0.8,1.2] scale factor applied to each load's P and Q — not a learned or historically-calibrated load distribution.

## Limitations / Failure Modes
- N-1 perturbation drops at most a single generator or single line/transformer per example — no N-2+, no line-addition, and no reconfiguration perturbations are included.
- Only load and topology are varied; generator capacities, line properties, and generator costs are never perturbed (paper explicitly names this as future work).
- No dual solutions, LMPs, or active-constraint-set labels are provided — only the primal solution and scalar objective value, limiting use for constraint-classification or pricing research.
- Samples are i.i.d. under a simple uniform perturbation rather than a targeted exploration of the feasible/active-constraint space; the paper itself flags this and suggests OPF-Learn-style active-constraint-set sampling or truncated-normal load perturbations as improvements.

## Reusable Ingredients
- Heterogeneous typed-node/typed-edge JSON schema (bus/generator/load/shunt nodes; ac_line/transformer/generator_link/load_link/shunt_link edges) closely parallels graphkit's own HeteroData design and could be adapted as an ingestion format.
- The N-1 perturbation protocol (single random generator-or-branch drop, reference-bus/connectivity guards, infeasible-sample discarding) is a reusable recipe for generating topology-perturbed data at scale.
- PGLib-OPF base cases solved end-to-end with PowerModels.jl/Ipopt/MUMPS is a ready-made reference pipeline for generating AC-OPF ground truth up to 13,659 buses.

## Open Questions
- Does uniform-random [0.8,1.2] load scaling plus single-component N-1 drops give coverage of the active-constraint-set space rich enough for learned solvers to generalize, rather than undersampling rare/binding regimes?
- Would adding generator-cost perturbations, N-2+ outages, or dual/LMP labels (all named as future work) materially change what downstream models can learn from this dataset?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
OPFData's typed bus/generator/load/shunt node and ac_line/transformer/generator_link/load_link/shunt_link edge schema is structurally close to graphkit's HeteroGridDatasetDisk representation, making it a plausible external training corpus (up to 13,659 buses) for the OOD grid-size generalization experiments needed to close gap G3, alongside [[rivera2025_benchmark_dataset_power]] for evaluation.

## Abstract (original)

> Solving the AC optimal power flow problem (AC-OPF) is critical to the efficient and safe planning and operation of power grids. Small efficiency improvements in this domain have the potential to lead to billions of dollars of cost savings, and significant reductions in emissions from fossil fuel generators. Recent work on data-driven solution methods for AC-OPF shows the potential for large speed improvements compared to traditional solvers; however, no large-scale open datasets for this problem exist. We present the largest readily-available collection of solved AC-OPF problems to date. This collection is orders of magnitude larger than existing readily-available datasets, allowing training of high-capacity data-driven models. Uniquely, it includes topological perturbations - a critical requirement for usage in realistic power grid operations. We hope this resource will spur the community to scale research to larger grid sizes with variable topology.
