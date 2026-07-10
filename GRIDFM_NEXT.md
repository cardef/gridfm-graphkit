# GridFM GraphKit Next

## Thesis

GridFM GraphKit should become the **operating system for learned power-system models**: one platform that learns reusable grid representations, adapts them to many tasks, quantifies when they are trustworthy, repairs small violations, and invokes numerical solvers when learning is not enough.

The repository already contains strong ingredients: heterogeneous graph models, PF/OPF/state-estimation tasks, physics-aware decoding, multi-grid data, and a Kron–Schur hierarchy. The next version should stop treating these as task-specific training paths and expose them as a coherent stack.

## Target stack

### 1. Canonical grid state

Introduce a versioned, unit-aware schema for buses, generators, branches, measurements, topology events, controls, dual variables, and solver status. Every dataset adapter maps into this schema. Models consume a `GridBatch`; tasks no longer infer semantics from global feature indices.

### 2. Topology-conditioned multiresolution backbone

Promote Kron–Schur reduction from one PF experiment to the central architectural primitive. Support multiple electrical scales, topology keys, cached sparse operators, and incremental updates after outages. Coarsening is electrical rather than purely geometric; learned corrections remain residual to auditable physical operators.

The research target is a **topology-conditioned neural multigrid operator** with electrical restriction/prolongation, shared fine/coarse processors, outage-conditioned operator deltas, gauge-aware voltage representations, and adaptive depth based on predicted difficulty.

### 3. Pretrained grid world model

Pretrain one backbone across PF, masked state reconstruction, topology recovery, contingency contrastive learning, OPF active-set prediction, and temporal forecasting. Use task heads and lightweight adapters instead of separate model classes.

### 4. Trustworthy runtime

Operational inference is a pipeline, not a raw neural forward pass:

```text
predict -> calibrate -> certify -> repair -> solver fallback -> audit
```

The first implementation lives in `gridfm_graphkit.runtime`: typed predictions, composable physical contracts, finite-sample conformal intervals, differentiable repair, fail-closed policy, and solver fallback.

### 5. Solver interoperability

Treat Newton, HELM, interior-point, and active-set solvers as tools behind stable interfaces. Models provide warm starts, active sets, trust regions, preconditioners, and contingency rankings. Solvers provide labels, certificates, correction steps, and hard fallbacks.

### 6. Deployment-grade evaluation

Standard evaluation must include unseen-grid and unseen-topology transfer, N-k outages, stressed regimes, feasibility, calibrated coverage, selective risk, solver iterations saved, end-to-end latency including fallback, matched-compute Pareto fronts, and adversarial scenario discovery.

## Architectural decomposition

```text
gridfm_graphkit/
  schema/        canonical state, units, topology identity
  operators/     Ybus, Kron/Schur, PTDF/LODF, sparse updates
  backbones/     multiresolution topology-conditioned encoders
  heads/         PF, OPF, SE, contingency, dynamics, active sets
  objectives/    supervised, masked, contrastive, physics, KKT
  runtime/       uncertainty, certificates, repair, fallback, audit
  solvers/       adapters and learned-solver coupling
  benchmarks/    OOD splits, stress tests, calibration, latency
```

Existing APIs remain available through compatibility adapters while new code migrates toward this structure.

## Research programme

- **M0 — trustworthy runtime:** model-independent contracts, certificates, calibration, repair, fallback.
- **M1 — structured outputs:** adapt PF, OPF, and SE models to `PredictionBundle`; move policy out of model internals.
- **M2 — topology-keyed operators:** immutable topology fingerprints and sparse low-rank contingency updates, removing the fixed-Y limitation.
- **M3 — multilevel neural operator:** adaptive V-cycle, shared processors, learned smoothers, physics transfer operators, residual stopping.
- **M4 — multi-task pretraining:** masked reconstruction, PF, SE, contingency, and OPF active-set objectives on one backbone.
- **M5 — calibrated selective solver:** choose among accept, repair, deeper model, high-fidelity surrogate, or solver under a risk/latency budget.

## Non-negotiable principles

- Physical structure is an inductive bias and certificate source, not a slogan.
- Every speedup includes preprocessing, calibration, repair, and fallback cost.
- Every approximation has a measurable gate and kill criterion.
- Failed or degenerate scenarios are never hidden.
- Models expose uncertainty and can abstain.
- Solver compatibility is a feature, not an admission of defeat.
- Architectural claims require matched-compute baselines.

## End state

A user loads a pretrained backbone, attaches a task head, runs on an unseen network, receives calibrated predictions with physical diagnostics, and automatically falls back to a trusted solver only for the difficult tail. That is more ambitious—and more useful—than merely reducing average RMSE on one power-flow benchmark.
