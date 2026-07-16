# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Open Gaps
# Gap Map

_Field gaps with stable IDs._

## G1 — No hierarchical GNN exploiting the grid's native electrical hierarchy
**Status:** unresolved
No published PF/OPF model coarsens along the domain-native hierarchy (voltage levels, substations, zones/areas) or uses electrically grounded reduction (Kron/Ward) as the pooling operator. **Correction (2026-07-06, full-text re-enrichment):** [[pham2024_reduced_optimal_power]] does *not* do virtual generator-node splitting or any graph-coarsening hierarchy — AHGNN is a nested two-stage prediction pipeline (a base-case GNN's output feeds a contingency-case GNN, fused with a historical-prior classifier) over the same flat bus/line graph, built for N-1 constraint screening, not structural coarsening. It is not prior art for this gap. The only adjacent work is generic, non-electrically-grounded learned pooling ([[gao2019_graph_unets]]). G1 remains fully open — no published PF/OPF model addresses it.

## G2 — Multiscale physics-simulation GNNs unadapted to power grids
**Status:** unresolved
[[fortunato2022_multiscale_meshgraphnets]] and [[cao2022_efficient_learning_meshbased]] show 2-level/bi-stride hierarchies fix long-range propagation on large
## Key Papers (25 total)
- [paper:bazzi2026_physicsinformed_coarsening_multigrid] Physics-Informed Coarsening for Multigrid Graph Neural Surrogates
- [paper:bueler2023_full_approximation_scheme] A full approximation scheme multilevel method for nonlinear variational inequalities
- [paper:cao2022_efficient_learning_meshbased] Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN
- [paper:chen2023_physicsguided_residual_learning] Physics-guided Residual Learning for Probabilistic Power Flow Analysis
- [paper:chevalier2022_towards_optimal_kronbased] Towards Optimal Kron-based Reduction Of Networks (Opti-KRON) for the Electric Power Grid
- [paper:dorfler2011_kron_reduction_graphs] Kron Reduction of Graphs with Applications to Electrical Networks
- [paper:fink2026_rapnet_accelerating_algebraic] RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections
- [paper:fortunato2022_multiscale_meshgraphnets] MultiScale MeshGraphNets
- [paper:gao2019_graph_unets] Graph U-Nets
- [paper:hamann2024_foundation_models_electric] Foundation Models for the Electric Power Grid
- [paper:ho2026_twolevel_nonlinear_schwarz] Two-level nonlinear Schwarz methods - a parallel implementation with application to nonlinear elasticity and incompressible flow problems
- [paper:lin2023_powerflownet_power_flow] PowerFlowNet: Power Flow Approximation Using Message Passing Graph Neural Networks
## Recent Relationships (52 total)
  idea:topology-vs-size-protocol --inspired_by--> paper:rivera2025_benchmark_dataset_power
  idea:topology-vs-size-protocol --inspired_by--> paper:piloto2024_canos_fast_scalable
  idea:topology-vs-size-protocol --addresses_gap--> gap:G3
  idea:hmatrix-zbus-attention --inspired_by--> paper:ma2023_graph_inductive_biases
  idea:hmatrix-zbus-attention --addresses_gap--> gap:G4
  idea:electrically-regularized-learned-coarsening --inspired_by--> paper:gao2019_graph_unets
  idea:electrically-regularized-learned-coarsening --addresses_gap--> gap:G1
  idea:ward-teacher-supervision --inspired_by--> paper:pham2022_reduced_optimal_power
  idea:ward-teacher-supervision --addresses_gap--> gap:G1
  idea:linear-probe-implicit-hierarchy --inspired_by--> paper:hamann2024_foundation_models_electric
  idea:linear-probe-implicit-hierarchy --addresses_gap--> gap:G1
  idea:oversmoothing-dirichlet-sweep --inspi
