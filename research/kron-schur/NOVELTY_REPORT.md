# Novelty Assessment: Electrical Hierarchies for Grid Foundation-Model Scalability

**Search cutoff:** 2026-07-13
**External reviewer:** Fable 5 Max
**Overall score:** 7.0 / 10
**Recommendation:** PROCEED WITH CAUTION

## Proposed Contribution

The proposal tests whether a deterministic Kron–Schur communication operator improves zero-shot power-flow transfer per cumulative FLOP in one parameter-shared multi-topology model as source topology count and held-out grid size increase. The implementation target is explicitly the `cardef/gridfm-graphkit` research fork.

The contribution is a controlled study, not a claim to have invented Kron reduction, graph coarsening, multiscale message passing, or grid foundation models.

## Claim Assessment

| Claim | Assessment | Reason |
|---|---|---|
| C1. Same-partition Kron-versus-quotient mechanism study under matched capacity, data, and cumulative FLOPs | MEDIUM | No complete precedent was located, but it is an obvious composition of known ingredients unless the control and scaling result are unusually clean. |
| C2. Kron advantage persists from `G8` to `G32` and does not contract on larger unseen grids | HIGH only if observed | This would be the main novel empirical finding. It cannot be claimed before the preregistered experiment passes. |
| C3. Conservative real-latent transport derived from Kron support and coefficient magnitudes | LOW | Restriction/prolongation pairs, Kron pooling, and AMG-style transport are established. Treat this as an implementation contract. |
| C4. Content-addressed geometry registry and additive fork seam | LOW | Standard software-engineering patterns; no novelty claim is warranted. |

## Closest Primary Work

| Work | Occupied territory | Remaining distinction |
|---|---|---|
| [Node Decimation Pooling](https://arxiv.org/abs/1910.11436) | Kron reduction and sparsification for GNN pooling | Does not run the proposed GridFM scaling/control study. |
| [MultiScale MeshGraphNets](https://arxiv.org/abs/2210.00612) | Generic fine/coarse physical message passing | Does not use the proposed electrical-versus-quotient intervention on power grids. |
| [Yaniv and Beck](https://arxiv.org/abs/2309.01124) | Electrical-correlation hierarchy of cluster-specific shallow ANNs | Not one parameter-shared GNN, no Kron–Schur coarse operator, and no multi-topology compute-matched study. |
| [LUMINA](https://arxiv.org/abs/2603.04300) and [LUMINA-Bench](https://arxiv.org/abs/2605.02133) | Multi-topology pretraining, held-out systems, adaptation | Establish the GridFM setting, not this communication-mechanism isolation. |
| [GridSFM](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/05/GridFM_white_paper.pdf) and [v1.1](https://github.com/microsoft/GridSFM) | Shared model with typewise global summaries | Supplies the necessary domain-global baseline. |
| [HydraGNN OPF-GFM](https://arxiv.org/abs/2605.23194) | Distributed training over large multi-grid OPF datasets | Occupies scalable GridFM training, not the proposed PF mechanism comparison. |
| [Scaling Laws of Machine Learning for Optimal Power Flow](https://arxiv.org/abs/2601.02706) | Data/compute scaling for OPF surrogates | Does not isolate electrical hierarchy across source diversity and unseen size. |
| [Power Flow Balancing with Decentralized GNNs](https://arxiv.org/abs/2111.02169) and [topology-independent PF GNNs](https://arxiv.org/abs/2204.07000) | Earlier multi-topology and unseen-grid PF learning | Do not provide the proposed same-partition mechanism control. |

## Strongest Rejection Attack

The method can be described as an obvious composition:

1. use known Kron reduction to build a coarse graph;
2. place it in a standard multiscale GNN;
3. pretrain across topologies as recent GridFMs already do.

The proposal survives this attack only at the study level. Kron and quotient use the same partition, adapter, channel schema, coarse-node count, and sparsity cap; local and typewise-global alternatives share the learned backbone; all primary comparisons use common cumulative-FLOP checkpoints. Realized support density is intentionally not equalized, so a positive Claim 2 applies to the full electrical operator family, not coefficient values alone.

If Kron ties Quotient, the electrical contribution fails. If both beat Flat, the result becomes evidence for generic multiscale communication and is much less novel.

## Search Limits and Residual Risk

The search covered primary arXiv papers, official PDFs, and official repositories for recent GridFMs, OPF scaling, Kron pooling, hierarchical PF, multiscale graph learning, learned graph tokenization, and scientific foundation models. The evidence log is [search-evidence.md](../.aris/traces/novelty-check/2026-07-13_run02/search-evidence.md).

This remains an absence-of-evidence result. Before submission, repeat the sweep for:

- post-cutoff LUMINA and GridSFM follow-ups;
- learned AMG or neural multigrid for power flow;
- effective-resistance graph rewiring, because Kron reduction preserves inter-anchor electrical relationships;
- Ward/Kron-equivalent neural-network literature predating deep GNNs.

## Decision Rule

Proceed to implementation only under the proposal's fail-closed protocol. The work earns a strong novelty claim only if Kron beats Flat, the released global-summary control, and the same-partition quotient control at the preregistered statistical and compute gates, with a non-contracting trend across diversity and size. Otherwise publish only the narrower result supported by the observed outcome.
