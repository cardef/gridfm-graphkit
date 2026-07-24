# Research Proposal: Electrical Hierarchies as a Scaling Mechanism for Grid Foundation Models

> **Repository scope:** this proposal explicitly targets [`cardef/gridfm-graphkit`](https://github.com/cardef/gridfm-graphkit), a research fork of upstream [`gridfm/gridfm-graphkit`](https://github.com/gridfm/gridfm-graphkit). The fork is the implementation and reproducibility boundary; it is not a claimed contribution.

> **2026-07-17 source-development amendment (pending external re-review):** the previous `G28` feasibility audit omitted the requirement that calibration use at least two disjoint whole-provenance source-development groups. An exhaustive topology-only audit of the pinned PGLib v23 inventory found no `G28` assignment and found `G26` is the largest source set compatible with two source-development groups, at least six held-out target groups, the frozen 0.5k–13.7k target envelope, and at least four size-extrapolative targets across at least two groups. The deterministic tie-break reserves PSERC and ACTIV for source development. No PF outcome or treatment result informed this amendment. The earlier G28 review remains historical and does not authorize this G26 contract.
> **2026-07-24 topology-only partition amendment (pending external re-review):** three fail-closed all-topology diagnostic builds ran before any treatment job or target-efficacy read. The first exposed missing or disconnected contiguous-METIS labels on 36 frozen topologies; the initial repair then exposed singleton cells with no non-anchor transport support on 29 topologies; the second repair exposed shallow breadth-first trees that could not restore covered cardinality on 16 topologies. The final global rule preserves the preregistered `m = ceil(rho N)`, repairs the seeded membership to exactly `m` connected cells with at least two buses per cell, and accepts the mathematically valid one-cell coarse graph with zero coarse edges on the smallest grids. No policy candidate, topology membership, target label, operating point, or per-grid override informed this amendment; R003 and every affected implementation/system gate must be rerun at the final clean commit before R014.


## Problem Anchor

- **Bottom-line problem**: determine whether parameter-free, topology-specific Kron–Schur geometry improves the compute–transfer scaling frontier of one parameter-shared grid foundation model across unseen network topologies and sizes.
- **Must-solve bottleneck**: a shared grid model needs electrically nonlocal communication. Flat message passing obtains it by increasing depth with graph diameter. Dense attention is expensive, while linear-cost global summaries are cheap but compress all long-range structure into a small pooled state. Existing grid-foundation-model studies establish multi-topology pretraining and adaptation but do not isolate whether an electrical hierarchy is a better communication mechanism. The comparison is invalid unless every model uses the same learned weights across grids, target grids contribute no labels or fitted output statistics at zero shot, and parameters, data, objectives, and compute accounting are matched.
- **Non-goals**: not the first scalable grid foundation model; not a universal scaling law over model size, dataset size, and compute; not a universal multi-task GridFM; not OPF, state estimation, dynamics, or topology control in this paper; not a new Kron reduction, Schur complement, pooling rule, or holomorphic solver; not recursive multilevel software; not a claim that zero-shot transfer must succeed on every grid.
- **Constraints**: approximately 230 GPU-hours on remote 3090-class GPUs; open or synthetically generated power-flow data; implementation in the explicit [`cardef/gridfm-graphkit`](https://github.com/cardef/gridfm-graphkit) research fork of [`gridfm/gridfm-graphkit`](https://github.com/gridfm/gridfm-graphkit); one fixed model-size tier; one two-level hierarchy; at most three primary backbones and one mechanism control; no target-grid hyperparameter search.
- **Success condition**: at common cumulative-FLOP budgets and matched parameter/data budgets, the Kron model improves the held-out-grid error–physics–compute frontier over both a flat heterogeneous GNN and an established linear-cost global-summary GNN. The advantage must persist as the number of distinct source base topologies grows and must not contract on larger unseen grids. A generic hierarchy built on the same partition and common sparsity cap must not explain the result. Few-shot label efficiency is supporting evidence, not part of the primary pass condition.

### Operational definition

The paper uses *foundation model* in the domain-specific sense used by LUMINA: one model is pretrained across multiple system configurations and then evaluated zero-shot or adapted quickly to unseen systems. It does not imply internet-scale pretraining or billions of parameters. Grid-specific admittance matrices and deterministic hierarchy operators are input geometry, not learned per-grid parameters.

The primary question is:

> At fixed learned capacity, total training scenarios, and cumulative training FLOPs, does deterministic Kron–Schur geometry give one parameter-shared multi-topology model lower held-out PF error at non-inferior physical residual than matched flat-local and typewise-global-summary communication as source base-topology count grows from 8 to 26 and unseen grid size grows from roughly 0.5k to 13.7k buses?

The study measures three distinct axes: source-topology diversity, held-out graph size, and cumulative compute. It does not infer a universal model-size or data-size scaling law from three diversity levels.

## Technical Gap and Novelty Boundary

Grid foundation-model scalability is already an active area:

- [Power Flow Balancing with Decentralized GNNs](https://arxiv.org/abs/2111.02169) trains across multiple topologies and evaluates unseen grids.
- [Solving AC Power Flow with GNNs under Realistic Constraints](https://arxiv.org/abs/2204.07000) studies topology-independent PF learning over diverse distribution grids.
- [LUMINA](https://arxiv.org/abs/2603.04300) and [LUMINA-Bench](https://arxiv.org/abs/2605.02133) study multi-topology pretraining, held-out systems, and adaptation for ACOPF.
- [GridSFM](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/05/GridFM_white_paper.pdf) trains one global-mixing model over many grids; its [v1.1 implementation](https://github.com/microsoft/GridSFM) uses typewise mean-plus-max summaries and broadcast fusion.
- The [HydraGNN OPF-GFM study](https://arxiv.org/abs/2605.23194) scales distributed training to millions of graphs from systems up to 13,659 buses.
- [Scaling Laws of Machine Learning for Optimal Power Flow](https://arxiv.org/abs/2601.02706) measures data- and compute-scaling for OPF surrogates.
- [Node Decimation Pooling](https://arxiv.org/abs/1910.11436) combines graph reduction, Kron reduction, and sparsification for GNN pooling; [MultiScale MeshGraphNets](https://arxiv.org/abs/2210.00612) establishes generic fine/coarse message passing.
- [Yaniv and Beck](https://arxiv.org/abs/2309.01124) divide a distribution system into clusters, train one single-hidden-layer ANN per cluster, and arrange those ANNs as a tree whose upward routing follows electrical correlation. Their tests use IEEE-123 and EPRI Ckt5. This is not a parameter-shared GNN, does not derive coarse communication from a Kron–Schur map, does not pretrain across base topologies, and does not compare electrical, generic-hierarchy, local, and global mechanisms at matched compute.

These works occupy “first scalable GridFM,” “first multi-topology pretraining,” generic Kron pooling, and generic hierarchy claims. The source audit found no study that isolates a deterministic Kron–Schur communication operator inside one parameter-shared grid model while matching local, global, and same-partition generic hierarchy alternatives for parameters, data, and compute. This is an absence-of-evidence statement, not a priority claim. The architectural ingredients are prior art; the proposed contribution is the controlled scaling study.

### Current repository failure point

The current fork prototype tests a case2000 specialist and makes Texas2k adaptation conditional. It also couples several mechanisms that invalidate the proposed study: per-grid fitted normalizers, a fixed-topology cache, target-voltage-dependent sparsity escalation, REF/PV boundary selection, and a PF-specific HELM reconstruction head. Those files remain historical prototype code. They are not evidence for the new method and cannot be imported by the confirmatory entry point.

## Method Thesis

**Thesis.** A parameter-free adapter derived from each topology's Kron–Schur geometry can carry electrically nonlocal messages through a sparse coarse graph and thereby improve zero-shot PF transfer per unit of training and inference compute.

The intervention changes only communication geometry. Encoders, local stem and readout, direct PF decoder, projection of known quantities, objective, optimizer, normalization, training examples, and checkpoint schedule remain common. HELM reconstruction is excluded from the primary path because it would make the treatment task-specific and break causal attribution.

## Explicit Fork Contract

All confirmatory work is additive code in `cardef/gridfm-graphkit`. It is not described as code accepted by upstream. Every run records:

- fork URL, fork commit, and worktree cleanliness;
- upstream URL, upstream reference commit, and merge base;
- configuration, data-manifest, and geometry-schema hashes;
- environment lock and accelerator type.

The fork reuses upstream schemas, registries, data abstractions, loss definitions, trainer behavior, and the flat `GNS_heterogeneous` contract. Fork-only additions are limited to:

1. topology partition and geometry builders;
2. a content-addressed geometry registry;
3. topology-portable normalization;
4. interchangeable communication cores; and
5. experiment manifests and integrity tests.

A dedicated pytest subprocess installs a `MetaPathFinder` deny-list for the legacy hierarchy builder, `GNS_hetero_hier`, HELM reconstruction, and fitted per-grid normalizers, clears those names from `sys.modules`, then imports and constructs the confirmatory entry point. Any attempted forbidden import fails the test. An upstream-flat compatibility test must reproduce construction, tensor schemas, and checkpoint loading under an unmodified flat configuration. Clean-clone instructions are tested at the recorded fork commit. Upstreaming is considered only after the mechanism passes its falsifiers.

## Proposed Method

### Common topology partition

Both hierarchy arms use the same deterministic partition:

1. order buses by stable case bus ID;
2. set `m = ceil(rho N)` (the eligible frozen inventory has `N >= 3`) and run contiguous METIS on the unweighted bus graph with the fixed seed;
3. deterministically repair the seeded membership to exactly `m` connected cells with at least two buses per cell when METIS returns an empty, disconnected, or singleton cell;
4. choose one anchor per cell by maximum fine-graph degree, breaking ties by minimum bus ID; and
5. restore tensor order.

`rho` is selected by the bounded source-only geometry calibration defined below. The partitioner cannot read PF labels, operating points, solver bus classes, target results, or per-grid overrides. Let anchors be `B` and remaining buses be `I`.

The repair first splits every raw METIS label into connected components. It processes singleton cells by stable bus ID and first borrows a removable adjacent bus whose donor remains connected with at least two buses; deterministic ties use donor size, moved bus ID, and donor ID bounds. If no such move exists, it merges the singleton into the smallest adjacent cell with stable-ID tie-breaks. If there are too many components, it repeatedly merges the adjacent pair with the smallest combined size, breaking ties by stable-ID bounds. If there are too few, it repeatedly peels the smallest valid subtree with at least two buses from a deterministic depth-first spanning tree of the largest splittable cell; both sides must remain connected and contain at least two buses. A one-cell hierarchy retains all non-anchor buses and uses an empty coarse-edge set. The repair rule is versioned, topology-only, identical for every grid, and fails closed if exact connected cardinality with transport coverage cannot be achieved.

### Kron–Schur construction

With the complex admittance matrix ordered as `(B,I)`, form offline:

```text
P     = -solve(Y_II, Y_IB)
Y_red = Y_BB - Y_BI solve(Y_II, Y_IB).
```

These dense objects are construction intermediates. The runtime model receives only sparse operators and attributes; it makes no exact-reconstruction claim after sparsification.

The source-frozen sparsifier retains a deterministic union of per-row top-`k_P` entries of `P` and the strongest entry for every coarse column. This guarantees row and column coverage or fails construction. Off-diagonal entries of `Y_red` use source-frozen per-row top-`k_C`. The complete geometry must satisfy:

```text
nnz(P_sparse) + nnz(E_coarse) <= kappa * |E_fine|.
```

If mandatory coverage violates the budget, construction fails; `k_P`, `k_C`, `kappa`, or the partition cannot be relaxed on a target grid. The topology-only residual

```text
||Y_II P_sparse + Y_IB||_F / max(||Y_IB||_F, epsilon)
```

and a frozen conditioning/resource gate may reject a topology but cannot tune it. A target failure remains in the confirmatory denominator.

### Conservative latent transport

Let `A = |P_sparse|` over interior-to-anchor edges. Define positive diagonal masses

```text
D_I = diag(A 1_B),       D_B = diag(A^T 1_I),
U   = D_I^-1 A,          R   = D_B^-1 A^T.
```

Coverage makes both diagonal inverses well-defined. Then

```text
U 1_B = 1_I,             R 1_I = 1_B,
D_B R = U^T D_I = A^T.
```

Thus prolongation `U` and restriction `R` preserve constants and are adjoint under the induced fine/coarse masses. This is a conservative pair for real latent messages. It is deliberately not presented as a complex Galerkin projection: AC `Y` is generally complex and non-Hermitian, and the learned hidden states are real. Electrical specificity enters through the harmonic-map support and coefficients, `Re/Im/magnitude/phase` edge attributes, and the Schur coarse graph. The edge-conditioned message function can use those signed complex attributes without introducing complex hidden states.

For the fixed confirmatory adapter, cross-level transport uses the support and magnitudes through the conservative `U/R` pair; the stored four-channel cross-level coefficient is reported for schema parity but is not injected through a second learned edge function. Signed `[Re, Im, magnitude, phase]` conditioning is consumed by the coarse processor. This keeps constant preservation exact and keeps `phi_down` and `phi_up` as the only adapter-specific trainable modules.

The adapter computes

```text
h_B^down = h_B + phi_down(aggregate_R(h_I))
z_B      = CoarseProcessor(h_B^down, E_coarse)
h_I^out  = h_I + phi_up([h_I, aggregate_U(z_B)])
h_B^out  = z_B.
```

The hierarchy consumes no per-scenario physical reconstruction quantity. In particular, the confirmatory path has no `v_aff`, `cbus_x`, boundary-voltage tensor, affine physical unpool, or HELM feature. An operating scenario enters only through the common fine-level input encoder; the geometry registry is a pure function of topology and frozen geometry policy.

The adapter maps latent width `d` to `d`. Its two residual branches are fixed as `phi_down(x) = W_2 SiLU(W_1 LayerNorm(x))` with `W_1,W_2: d -> d`, and `phi_up([h,u]) = W_4 SiLU(W_3 LayerNorm([h,u]))` with `W_3: 2d -> d` and `W_4: d -> d`; there is no dropout or output activation. The coarse processor uses the same block family and initializer family as the fine processor, with independent weights and deterministic seed mapping; there is no cross-level weight sharing. These are the only adapter-specific trainable modules. The adapter predicts no voltage and owns no task loss.

### Generic hierarchy control

The quotient arm uses the identical partition, adapter code, coarse-node count, parameter budget, message primitive, and edge-attribute schema. Its cross-level coefficient is the cell-assignment value `1+0j`, encoded in the same `[Re, Im, magnitude, phase]` channels as a Kron coefficient. For distinct cells `a,b`, its coarse weight is the complex cut sum

```text
Q_ab = sum_{i in cell a, j in cell b} Y_ij,
```

so the quotient graph retains only fine-graph cell adjacencies and exposes the same four coefficient channels as the Schur graph; it has no elimination-induced fill. Both hierarchy arms obey the same `nnz <= kappa |E_fine|` cap, but the quotient arm is not padded to imitate Schur fill and realized nonzeros need not match. Cross-level and coarse nonzeros are reported separately for both arms. Kron may therefore carry more communication nonzeros; this treatment-inherent bandwidth is charged in profiled training and inference FLOPs, and primary comparisons use common cumulative-FLOP checkpoints. No equality of per-step edge counts is claimed. The treatment changes harmonic versus assignment transport, Schur versus cut-sum coarse coupling, and the support induced by those operators; it does not change the partition, channel schema, U-Net shape, task head, or training recipe.

### Minimal software contract

```python
@dataclass(frozen=True)
class HierarchyGeometry:
    topology_key: str
    partition: Partition
    restrict: SparseOperator
    prolong: SparseOperator
    coarse_graph: SparseGraph
    provenance: GeometryProvenance

class GeometryBuilder(Protocol):
    def build(self, topology: GridTopology, budget: GeometryBudget) -> HierarchyGeometry: ...

class CommunicationCore(Protocol):
    def forward(self, h: HeteroLatents, context: TopologyContext) -> HeteroLatents: ...
```

`LocalCore`, `GlobalSummaryCore`, and `HierarchyCore` implement one interface. `KronGeometryBuilder` and `QuotientGeometryBuilder` are the only builders consumed by `HierarchyCore`. A content-addressed registry owns one immutable geometry per topology/schema/device; samples carry a topology key, never copied operators or filesystem paths. Task models depend on `CommunicationCore`, not on a hierarchy implementation.

### Shared communication slot and baselines

All headline models apply the same encoder, exactly `L_pre` shared fine-graph local blocks, one communication-core call, exactly `L_post` shared fine-graph local blocks, and the same direct PF decoder and known-value projector. `L_pre`, `L_post`, masks, objective, optimizer family, data order, and checkpoints are frozen before any treatment run. The communication slot is invoked once in every arm:

1. **Flat-HGNS:** the slot contains a source-frozen number `q` of additional fine-graph relation-aware blocks.
2. **Global-HGNS:** the slot contains exactly one GridSFM v1.1-style typewise mean-plus-max pool, one projection and broadcast, and one fine-graph local block. There is no second global summary elsewhere.
3. **Kron-HGNS:** the slot contains one down/coarse/up adapter and exactly one coarse block.
4. **Quotient-HGNS:** the `G26` mechanism control uses the same single adapter slot and coarse-block count with quotient geometry.

Widths and `q` are selected by a deterministic source-only parameter-matching search to keep trainable parameters within 2% without dummy parameters. Exact parameters, profiled cumulative FLOPs, wall time, and memory are reported. Fine and coarse blocks use independent parameters because their graph domains differ; matching concerns total capacity, not artificial weight tying.

### Frozen source-only selection budget

Before any target geometry is built or any hierarchy/global treatment is trained:

- The manifest preregisters at most twelve joint `(rho, k_P, k_C, kappa)` policies. They are evaluated only on topology-only source-development cases. Policies failing row/column coverage, conditioning, host-memory, build-time, or `nnz <= kappa |E_fine|` gates are discarded. Among the rest, select the lowest projected sparse-message FLOPs whose worst-case harmonic residual is within 5% of the best feasible residual; break ties by lower nonzeros, then fewer coarse nodes. The full candidate list, measurements, and deterministic choice are hashed at F0.
- The common loss-weight vector is selected from at most three preregistered candidates using Flat-HGNS only on source-development cases. A treatment-blind Flat throughput probe first freezes `C_cal`, the largest common profiled-FLOP budget for which one seed-0 run per candidate fits within a 3 GPU-hour total upper bound. Train every candidate to `C_cal`; choose the lowest family-balanced source PF error, with ties within 1% going to the lower physical residual. This spend is part of the ten-hour pre-campaign allowance. The selected vector is then fixed for every arm, and its `C_cal` checkpoint is the sole checkpoint used for the power-design dispersion below.
- No arm receives a separate search budget. Any additional architecture-independent choice must fit the same source-only allowance and be frozen before the first treatment run; otherwise the confirmatory campaign blocks.

## Portable Data Contract

Every arm uses the same zero-shot-safe normalization:

- powers and admittances use `baseMVA` read from case metadata; missing or inconsistent metadata is a data-integrity failure, never replaced by an assumed 100 MVA or a fitted percentile;
- angles are radians and voltages are per unit;
- optional standardization is fitted once on source-training inputs, stored in the checkpoint, and frozen;
- all cases share a canonical bus/generator/branch schema with explicit missing-field masks;
- case IDs, trainable topology embeddings, and per-grid affine parameters are forbidden; and
- target outputs cannot affect normalization, partitioning, sparsity, geometry acceptance, checkpoint selection, or hyperparameters.

## Scaling Protocol

### Source and target topology sets

Construct nested source sets `G8 subset G16 subset G26` from distinct intact PGLib base cases. Line-outage variants do not count as distinct systems and are excluded from the core study. The exact sampler total is `S_total = 11,655` scenario exposures per epoch for each of `G8`, `G16`, and `G26`; diversity therefore trades exposures per topology rather than silently increasing training work. This is the smallest multiple balanced across the observed five- and seven-group source sets that gives every case in the 13-case endpoint provenance group at least 128 batches per epoch at batch size one. Generate 2,331 immutable PF scenarios per source topology, which prevents within-epoch reuse even for a singleton G8 provenance group, and 512 scenarios per source-development or target topology. Sampling without replacement is per case until its pool is exhausted; all arms at a diversity level see the same frozen splits.

Before PF generation, one source-only preprocessing policy removes a case-declared MATPOWER type-4 bus only when its load, shunt, in-service generator count, and in-service incident-branch count are all zero; any non-inert type-4 bus or disconnected energized remainder blocks. The manifest records the resulting energized bus count, while generation provenance records the policy and original bus IDs removed.

Related snapshots and variants share a `provenance_group`. Whole groups, not files, are held out. The confirmatory target pool contains at least twelve intact topologies over at least six held-out provenance groups and spans roughly 0.5k–13.7k buses; fewer than six groups blocks confirmation. After the source-only Flat-HGNS calibration fixes the power-gated group count, but before the first global or hierarchy treatment run, a frozen `topology_manifest.yaml` records case, group, bus count, source/target split, case `baseMVA`, and integrity status. It also records `N_source_min` and `N_source_max`. A target is called size-extrapolative only when `N_target > N_source_max`; at least four such targets across at least two held-out groups are required for a size-extrapolation claim.

### Optimization and run matrix

- Same-topology batches avoid dynamic shapes; sampling balances provenance groups and then cases.
- Loss is averaged per graph and predicted component so large grids do not dominate by node count.
- The common objective is masked VM/VA error plus the existing dimensionally valid power-balance residual with one frozen weight vector.
- There is no coarse loss, reconstruction loss, or hierarchy-specific regularizer.
- Models checkpoint when cumulative profiled work first crosses `{C/4, C/2, C}`. A sample-matched endpoint is secondary.
- Source-only development cases select architecture-independent training choices. No target-specific checkpoint is selected.
- Mandatory runs are `3 communication cores x 3 diversity levels x 2 seeds = 18`, plus two `G26` Quotient-HGNS seeds: 20 pretraining runs.

Zero-shot evaluation builds target geometry without labels and evaluates the shared checkpoint directly. Build time and peak host memory are reported separately and amortized over `1` and `1000` scenarios. Few-shot adaptation is an optional exploratory appendix only after the mandatory campaign and reserve close; it has no preregistered claim, cannot enter the title or abstract, and is not budgeted inside the 230 GPU-hour confirmatory study.

## Decisive Evaluation

The statistical unit is a topology, not a scenario or seed. For each arm, topology, and checkpoint, first aggregate scenarios, then average the two preregistered seeds. Seed-level results and between-seed dispersion are reported, but seeds are not treated as independent topology replicates. For each model contrast, average topology-level paired log differences within each provenance group, yielding at least six equal-weight group contrasts.

Let `d_e = log(error_Kron) - log(error_baseline)` and `d_r = log(residual_Kron) - log(residual_baseline)`. For each contrast, the sign-flip statistic is the unweighted arithmetic mean of the equal-weight provenance-group contrasts. The primary superiority and non-inferiority decisions use an exact one-sided sign-flip randomization test over those contrasts; the corresponding one-sided 95% bounds are obtained by inversion of that test. The exchangeable-sign assumption, every group contrast, and a wild-cluster-bootstrap-t sensitivity analysis are reported. A percentile hierarchical bootstrap is not used. All gates below are conjunctive, so there is no choice among favorable checkpoints or target subsets.

The group count is power-gated before any treatment run. Define the smallest scientifically relevant error improvement as `delta_min = -log(0.95)`, a 5% reduction in family-balanced error. No located frontier study reports this same held-out-PF, family-balanced contrast, so 5% is an adoption threshold rather than a literature-derived effect: a smaller gain would not justify the hierarchy's added static geometry and implementation surface at equal FLOPs. Evaluate the selected-loss Flat-HGNS seed-0 `C_cal` checkpoint without adaptation on every held-out source-development provenance group, using the confirmatory scenario-to-topology-to-group aggregation and error definition. Let `s_Flat` be the standard deviation of those group-level log errors and set `sigma_design = sqrt(2) s_Flat`, under the explicit design assumptions that every treatment arm's marginal group-level dispersion is no larger than Flat's and that paired arm errors are nonnegatively correlated by group. Under independent symmetric `Normal(-delta_min, sigma_design^2)` group contrasts, a fixed-seed Monte Carlo calculation with one million draws estimates power for the exact one-sided sign-flip test. Freeze the smallest available target-group count of at least six that gives at least 80% design power; if the target pool cannot meet it, the confirmatory campaign blocks. This normal model and its dispersion assumptions govern design only, not analysis validity. With exactly six groups the randomization distribution has 64 sign assignments and minimum one-sided `p = 1/64`, so rejection requires nearly unanimous aligned group evidence.

The target manifest freezes bus-count terciles before training, with at least four targets per tercile. The primary pass requires:

1. at `G26/C`, the inverted one-sided upper 95% bound for `d_e` is below zero against both Flat-HGNS and Global-HGNS;
2. the inverted upper 95% bound for `d_r` is at most `log(1.05)` against both baselines; 5% is a study-design margin, not an industry tolerance;
3. the family-balanced `d_e` point estimate is below zero against both baselines at both fixed checkpoints `C/2` and `C`;
4. the family-balanced `d_e` point estimate is negative in every frozen size tercile and the largest-tercile estimate is no closer to zero than the smallest-tercile estimate; this replaces an underpowered fitted slope claim; and
5. the family-balanced `d_e` point estimate is negative at the diversity endpoints `G8` and `G26`, and the `G26` estimate is no closer to zero than the `G8` estimate; `G16` remains a mandatory reported point but is not an additional pass gate.

For the electrical-mechanism claim, the same exact group-level test must put the upper 95% bound for `log(error_Kron)-log(error_Quotient)` below zero at `G26/C`, while the same 5% residual gate holds. No effect threshold is selected after pilots.

Reported systems metrics are cumulative training FLOPs, GPU-hours, warm inference latency, accelerator peak memory, static build time, dense-intermediate host peak memory, and amortized end-to-end latency. Any construction failure fails the primary claim on that topology; it is not removed or assigned a convenient imputation.

## Claim Boundaries

### Claim 1 — compute-transfer scaling

If the primary gate passes, the paper may claim that deterministic Kron–Schur geometry improved the tested held-out PF error–physics–compute frontier across the preregistered source-diversity and target-size ranges. It may not claim a universal scaling law.

### Claim 2 — electrical rather than generic hierarchy

If Kron beats Quotient under the paired gate, the paper may attribute the measured advantage to the full electrical operator family under the matched partition, adapter, channel schema, and sparsity cap. Because realized support density is treatment-inherent and not edge-count-matched, the paper may not attribute the result to coefficient values alone. If they tie, the claim reduces to generic multiscale communication.

### Exploratory appendix — target-label efficiency

If spare compute remains after the mandatory campaign and reserve, a frozen few-shot recipe may be reported descriptively. It carries no confirmatory claim and cannot rescue Claim 1.

## Failure Conditions

1. A source-only gain that disappears on held-out provenance groups is specialist performance, not FM transfer.
2. A Global-HGNS tie or win makes the electrical hierarchy unnecessary relative to the simpler mechanism.
3. A Quotient-HGNS tie removes the electrical-specific claim.
4. Failure to retain a negative contrast in every frozen size tercile, or a largest-tercile advantage smaller than the smallest-tercile advantage, rejects the scalability thesis.
5. A gain that vanishes at matched cumulative FLOPs rejects compute-efficiency claims.
6. Frozen sparsity, conditioning, or host-memory gate failures define a failed operating regime and remain visible.
7. Any target-label-dependent preprocessing invalidates the zero-shot block.
8. If one two-level coarse graph remains too large, the paper reports the limit; it does not add recursive levels post hoc.
9. If the common budget cannot support the preregistered learning horizon and full matrix, the study is a pilot, not a confirmatory result.

## Frontier Positioning Without Extra Treatments

[GraphFM](https://arxiv.org/abs/2407.11907) uses learned latent tokens for multi-graph compression; [Graph Mamba](https://arxiv.org/abs/2402.08678) uses neighborhood tokenization, ordering, and a selective state-space encoder; [GraphGPS](https://arxiv.org/abs/2205.12454) provides scalable local/global graph-transformer designs. These are legitimate generic routes to nonlocal communication. They are not the primary treatment here because each adds a tokenizer, ordering/positional design, or global mixer whose learned inductive bias would vary together with the electrical geometry. The deterministic adapter instead asks a narrower question: whether the known circuit operator is useful when the learned model is held fixed.

[DPOT](https://arxiv.org/abs/2403.03542) and [Poseidon](https://arxiv.org/abs/2405.19101) are the closest scientific-foundation-model analogs: both pretrain operator models across PDE data, and Poseidon uses a multiscale operator transformer. They establish that multiscale structure and broad physical pretraining can support transfer. They do not test irregular power-network topologies, admittance-derived geometry, or a Kron-versus-generic-hierarchy intervention. They motivate the question; they are not matched baselines for it.

Linear attention is also excluded from the mandatory matrix. Kernel choice, positional encoding, and attention normalization constitute a second method axis. GridSFM's released linear-cost summary is the necessary domain baseline. A later robustness study may replace the global core with a stronger linear mixer, but it cannot replace the flat, GridSFM-style global, or quotient controls.

## Implementation Acceptance Path in the Fork

1. **F0 — fork and protocol freeze:** add an upstream remote or otherwise record the upstream reference; after the Flat-only source calibration, freeze the power-gated target manifest with at least six groups, source-size range, exact sign-flip analysis, bounded source-only selection candidates, no-leak rules, package boundaries, and all hashes.
2. **F1 — geometry domain:** implement deterministic partitioning, label-blind Kron and quotient builders, conservative transport, immutable schemas, and a content-addressed registry.
3. **F2 — communication seam:** extract the shared backbone seam and implement local, global-summary, and hierarchy cores without importing the legacy hierarchical model; run the subprocess `MetaPathFinder` deny-list test from the first F2 commit onward.
4. **F3 — portable data:** implement case-declared `baseMVA`, source-frozen optional standardization, family-balanced batching, and a test that target outputs are unreadable during preprocessing.
5. **F4 — algebra and systems gates:** dense-reference Schur tests, constant-preservation and mass-adjoint tests, permutation tests, sparsity/resource gates, recheck subprocess `MetaPathFinder` legacy-import denial, fork/upstream-flat compatibility, clean-clone cache tests, multi-topology batching, FLOP calibration, and largest-grid host/GPU probes.
6. **F5 — pilot then campaign:** run one short treatment-blind `G8` smoke pilot only after F0-F4 pass; use it to validate execution and budget accounting, not to tune the primary hypothesis. Freeze `C`, then launch all 20 mandatory runs.

Conditional `torch.compile` use requires output-parity, gradient-parity, and FLOP-counter checks for every core. It is a shared systems setting, not a method contribution. Sparse factorization and sparse right-hand sides may reduce intermediate work but do not eliminate the potentially dense `P`; resource gates are therefore based on measurements, not asymptotic optimism.

## Compute and Timeline

- Spend at most 10 GPU-hours total before the campaign: at most 3 on Flat-HGNS loss-weight selection and the remainder on treatment-blind `G26` throughput profiles.
- Fit an upper runtime bound for each core as a function of counted FLOPs.
- Freeze the largest common `C` for which the 20 upper-bound run costs, profiling spend, and a 20% campaign reserve total at most 230 GPU-hours.
- If `C` falls below a source-only preregistered learning horizon, block the confirmatory campaign rather than deleting a seed, baseline, diversity level, or quotient control.
- Budget CPU data generation and hierarchy construction separately, but report both.
- Use a 10–12 week implementation estimate: roughly three weeks for F0-F3, two for F4 and profiling, four for the mandatory campaign, and two to three for controls and analysis. Exploratory adaptation begins only after these close.

## Evidence Package

The planned paper needs four load-bearing figures rather than a broad catalog:

1. the common-backbone/fork-boundary diagram with the communication core as the sole treatment;
2. held-out error versus cumulative FLOPs for `G8/G16/G26`, with physical-residual gates;
3. paired Kron advantage across frozen target-size terciles and the Kron-versus-quotient mechanism result; and
4. static build cost, warm inference cost, and failure coverage.

Tables report topology provenance, parameter/FLOP matching, integrity gates, and per-family results. The paper is ready for submission only after these artifacts exist. This proposal is ready for implementation when its contracts survive review; those are different thresholds.

## Final Contribution Statement

The most defensible contribution is not a new component. It is a controlled, falsifiable study of whether deterministic electrical hierarchy changes the compute-transfer scaling frontier of a domain-specific grid foundation model. The explicit `cardef/gridfm-graphkit` fork supplies a narrow implementation seam and reproducibility boundary. A positive result supports electrical hierarchy over matched local, global-summary, and generic-hierarchy alternatives within the tested regime. A negative result identifies the simpler mechanism or resource regime that dominates.
