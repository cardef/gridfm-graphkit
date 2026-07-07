# M1 experiment pack (cluster)

Prepared 2026-07-07 after M0 passed all gates (`refine-logs/EXPERIMENT_RESULTS.md`).
Everything here is generated/runnable; GPU items need the cluster.

## Order of operations

1. **Data generation** — `datakit_configs/*.yaml` (from `r001_cluster_configs.py`):
   ~10k solved scenarios/grid, M0-measured yields baked into the request
   counts (case2000 requests 40960 at 25% fast-PF yield; decision and
   caveats in the script docstring). Run datakit from the repo root with
   `gridfm-datakit` as a sibling checkout.
2. **Hierarchy precompute** — re-run `experiments/m0/r002_precompute.py` on
   the regenerated data (per-grid Kron-Schur caches; the hard P-mass assert
   re-fires there). Per-grid boundary-fraction overrides, if a grid needs
   one, go in `data.hierarchy.per_network.<network>.target_frac`
   (see `datasets/hierarchy.py::AddHierarchy`; case118's 46% gen-bus floor
   is why the knob exists — not needed for case500/case2000 at 23%/17%).
3. **Consolidated store** — all training configs already set
   `data.consolidated: true` (E003: 1.9–2.3× loader throughput,
   byte-identical).
4. **R010 runs** — `configs/*.yaml` (from `r010_make_configs.py`).
5. **R011** — `r011_grit_memory.py` (phase A is CPU-runnable; phase B needs
   CUDA).

## R010 run matrix (per grid: case500_goc, case2000_goc)

| Arm | Model | What it answers |
|---|---|---|
| `flat_<grid>_d{8,16,32,48}` | GNS_heterogeneous h48 | natural-width depth sweep (R010 axis) |
| `flat_<grid>_d{8,16,32,48}_iso` | GNS_heterogeneous, width-matched | iso-FLOP falsifier arms vs KS |
| `ks_<grid>` | GNS_hetero_hier h48, 4/8/4, λ_v=0.2 | the treatment (fixed reference) |

Matched-FLOP pairing (computed from the pre-registered
`gridfm_graphkit/utils/flops.py`; full table in `results/r010_matrix.json`).
KS is held fixed; the flat side gets the width adjustment (fairness: the
baseline gets every chance). All eight iso arms land inside the [0.9, 1.1]
window that the natural-width sweep alone cannot reach (R003):

| Grid | KS GFLOP | d8 iso | d16 iso | d32 iso | d48 iso |
|---|---|---|---|---|---|
| case500_goc | 14.1 | h56 → 1.017× | h38 → 0.993× | h27 → 1.034× | h22 → 1.042× |
| case2000_goc | 46.1 | h52 → 0.989× | h36 → 1.004× | h25 → 0.999× | h20 → 0.972× |

Side observation: iso-FLOP is also roughly iso-parameter here (iso arms
15–18M params vs KS 18.2M; natural d48 is 85M).

- **Seeds**: seed 0 committed; `python r010_make_configs.py --seeds 0 1 2`
  emits seed-suffixed copies (the CLI has no seed override flag). Plan:
  3 seeds at case500, 2 at case2000.
- **Recipe**: repo example PF recipe (LayeredWeightedPhysics 0.1 +
  MaskedBusMSE 0.9, lr 5e-4); KS adds CoarseVoltageMSE 0.2. Batch 8/4
  (case500/case2000) from the E001 budgets — retune on the cluster GPU if
  memory allows, but keep flat/KS batch sizes equal within a pairing.
- All configs enable `data.same_grid_batches` (E005 sampler → static
  shapes; single-grid runs get exact-size batches, tail dropped). For
  torch.compile: `--compile max-autotune` at case2000,
  `--compile reduce-overhead` at ≤case118 (engineering plan E4).
- MUST set for the M2 gate decision: `ks_case2000` vs `flat_case2000_d*_iso`
  (deep arms d32/d48 are the declared falsifier); the rest of the sweep
  feeds the B1 frontier figure.

## R011 GRIT memory curve

`r011_grit_memory.py` — full-attention GRIT (example scale h496/7L, full
RRWP ksteps 21). Phase A (CPU, done locally, 2026-07-07): RRWP precompute
s/sample and relative-PE bytes (`results/r011_grit_memory.json`) — headline:
case2000 costs 0.52 s/sample and 250 MB of relative PE **per sample**
(~N² pairs × 21 steps), i.e. a batch of 4 carries ~1 GB of positional
edge attrs before any activations. Phase B (cluster GPU):
peak memory per batch size {1,2,4,8}, stop at first OOM, quadratic fit of
bs=1 peak vs n_bus. CPU hosts run a bs=1 wiring step instead of phase B
(validated on case14 locally).
