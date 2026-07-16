# Gap Map

_Field gaps with stable IDs._

## G1 — No hierarchical GNN exploiting the grid's native electrical hierarchy
**Status:** unresolved
No published PF/OPF model coarsens along the domain-native hierarchy (voltage levels, substations, zones/areas) or uses electrically grounded reduction (Kron/Ward) as the pooling operator. **Correction (2026-07-06, full-text re-enrichment):** [[pham2024_reduced_optimal_power]] does *not* do virtual generator-node splitting or any graph-coarsening hierarchy — AHGNN is a nested two-stage prediction pipeline (a base-case GNN's output feeds a contingency-case GNN, fused with a historical-prior classifier) over the same flat bus/line graph, built for N-1 constraint screening, not structural coarsening. It is not prior art for this gap. The only adjacent work is generic, non-electrically-grounded learned pooling ([[gao2019_graph_unets]]). G1 remains fully open — no published PF/OPF model addresses it.

## G2 — Multiscale physics-simulation GNNs unadapted to power grids
**Status:** unresolved
[[fortunato2022_multiscale_meshgraphnets]] and [[cao2022_efficient_learning_meshbased]] show 2-level/bi-stride hierarchies fix long-range propagation on large meshes, but their coarsening assumes spatially embedded, homogeneous meshes. Power grids are heterogeneous (bus/gen/load/line types, as in graphkit's HeteroData) and non-spatial; no adaptation exists.

## G3 — OOD grid-size generalization of learned PF solvers
**Status:** unresolved
[[rivera2025_benchmark_dataset_power]] (PFΔ) shows SOTA GNN solvers (TypedGNN, PowerFlowNet, CANOS) degrade on unseen topologies and grid sizes. Whether hierarchical structure improves size transfer (train small, infer 10k+ buses) is untested.

## G4 — Sub-quadratic graph transformers for 10k+-bus grids
**Status:** unresolved
GRIT ([[ma2023_graph_inductive_biases]], used in graphkit) needs full O(N²) attention plus dense K-step RRWP precompute — prohibitive beyond a few thousand buses. No sub-quadratic (e.g., hierarchical/cluster attention) grid transformer demonstrated at real transmission-system scale.
