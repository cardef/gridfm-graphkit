# Piano A on the released GENCO artifacts (F0 of `AC_PF_SYMMETRIES_PLAN.md`) — revised 2026-09-29

Status: code ready and smoke-tested on our own checkpoint; **nothing from Hugging Face downloaded yet**.
Provenance tags: [C] confirmed (command run / file read this session), [I] inferred, [?] unknown until data is on disk.

## 1. What is public

- Paper arXiv:2608.09921; repro notes github.com/albanpuech/GENCO; code branches of `gridfm/gridfm-graphkit` `genco-paper-repro` (6edf7eb),
  `-pfdelta` (4fed754), `-pretraining` (d001694): SHAs equal our local refs [C].
- HF `gridfm/genco-pf-datakit`: IEEE 14/30/57/118 and GOC 500 x {base,small,tiny} x seeds {0,1,42}; GOC 2000/10000 small/tiny only. Per folder:
  `best_model_state_dict.pt`, `normalizer_stats.pt`, `metrics.csv`, training YAML [C]. No ckpt for IEEE 300, PEGASE 1354/2869, ACTIVSg [C: repo tree]
  (a `pf_small_case1354_pegase` dataset exists but has no ckpt). Also `genco-pfdelta` (24 ckpts, IEEE 118), `genco-pf-transfer`, OPF repos.
- **GENCO Base = the architecture of our M0**: released YAML `model:` block is identical to our base config (`GNS_heterogeneous`, hidden 48, 12 layers,
  8 heads, edge_dim 10) [C]; 20.1M params in the paper vs 20,069,187 counted on our checkpoint [C]. Released weights: trained on `pf_small_case14_ieee`
  = 219,606 scenarios (split 175,713/21,947/21,946; `n_scenarios.txt` = 219606) vs 1,640 train scenarios in our PoC [C]; patience 100, 200 epochs.

## 2. Compatibility

- Model: `GNS_heterogeneous` of `genco-paper-repro` loads our `main`-trained weights `strict=True` (416 keys), outputs bit-identical on a real batch
  [C]; the only forward difference is a last-layer `physics_mlp` update that the code comments call dead. Whether the released state dicts use the same
  key names: [?] (checked only after download; a `model._orig_mod.` remap is handled).
- Data: `main` reads the public Hive layout (`bus_data.parquet/scenario_partition=k/`, ~200 scenarios per partition) via `stream_partitions: auto`
  [C: code], not yet run on it [?].
- `normalizer_stats.pt` is `{network: stats}` (`baseMVA_orig`, `baseMVA`, `vn_kv_max`); a missing network only *warns and refits* [C]. `genco_audit.py`
  asserts the stats were applied (it failed loudly on my first wrongly-keyed smoke file).
- **The split depends on the checkpoint's seed** (`random.seed(args.seed)` in the datamodule) [C]: seeds 0/1/42 have different test sets. Exact reproduction:
  the datamodule supports `data.split_from_existing_files` (a folder with `train.pt/val.pt/test.pt`, mutually exclusive with
  `split_by_load_scenario_idx`) [C]; `genco_audit --splits-json` converts the mlflow `*_scenario_splits.json` and asserts the test size (smoke-tested with
  synthetic ids). Full data only; on partial subsets the split is ours and the audited scenarios may include a model's own train scenarios (EE unaffected,
  accuracy claims must say so).
- `.venv-screening` cannot import `gridfm_graphkit.cli` (`gridfm_datakit` missing, pre-existing) [C]: the official `evaluate` CLI is not runnable there.
- **Do not follow the model card's `--local-dir data/case14_ieee/raw`: it would overwrite the PoC's local 2048-scenario data.** Use `data/genco/...`.
- Disk: 12 GiB free (98% full) [C]. Processed cache ~8 KB/scenario (local: 16 MB / 2048) => ~1.8 GB for full IEEE14 [I], likely >10 GB for IEEE118 [I].
  Hence full data only for IEEE14; 10 Hive partitions (~2,000 scenarios) for the larger grids.

## 3. Already run (no download)

- **E0 local data audit** (`data_audit.py`, `results/data_audit_local.json`) [C], 7 local grids (IEEE 14/30/57/118, GOC 500/2000, Texas2k): REF angle
  exactly 0; **no phase shifter** (`shift` = 0 on every in-service branch); transformer taps exist (e.g. case14: 0.932/0.969/0.978; `Yff != Ytt` on them,
  `Yft == Ytf`); per-graph mean |Yff| constant (CV ~1e-16), i.e. fixed topology in these local sets. So S2 is untestable and `Canonicalize` is an
  identity in-distribution *on these grids* (public sets may include topology changes [?]: E0p).
- **`genco_audit.py` on our `m0_seed0`** (200 test graphs) [C]: post-hoc `Canonicalize` changes in-dist RMSE by relative 2e-7 (VA) ... 4e-5 (PBE);
  EE_S1(pi) raw 1.15 -> canon 7.7e-8; EE_S1(0.1) 0.60 -> 1.7e-6; EE_S3(k=100) 0.033 -> 2.1e-7; EE_S4 ~9.5e-7 both. So the PoC's "trained canon beats M0
  in-dist (0.28 vs 0.51 deg)" cannot come from the wrapper as a function; [I] it is training-trajectory noise (`fit_scale_ref` iterates the shuffled
  train loader before training and consumes RNG). Retire that claim; do not spend compute on it.
- **E5 preview** (`--target`, `m0_seed0` from case14, 200 graphs, physical units) [C]; one model, one seed, so sign is not established:

  | target | normalizer | baseMVA | VM (pu) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) |
  |---|---|---|---|---|---|---|---|
  | case30 | source (fixed) | 82.28 | 0.0231 | 1.82 | 41.2 | 26.7 | 8.17 |
  | case30 | refit (default) | 40.00 | 0.0494 | 1.49 | 27.4 | 51.9 | 11.57 |
  | case57 | source (fixed) | 82.28 | 0.0321 | 9.70 | 300.3 | 67.5 | 17.64 |
  | case57 | refit (default) | 110.81 | 0.0253 | 9.85 | 325.6 | 101.1 | 13.34 |

  The normalization convention moves zero-shot metrics by up to 2x, with inconsistent sign across targets. The PoC's zero-shot table used *refit*
  (`build_grid` fits a normalizer per target grid, reading target Pg/Qg): it is not zero-shot in the fixed-convention sense.

## 4. Corrections to the first draft of this plan (all found on review)

- G1 said "PBE x baseMVA vs metrics.csv": wrong. `accuracy()["PBE"]` is already in the normalizer's per-unit and equals the framework's "PBE Mean"
  (mean over graphs of the per-graph mean of sqrt(rP^2+rQ^2), `pf_task.py`) for fixed-size graphs. Compare directly. [C: code; equality of numbers [?]]
- G2 assumed one split for all seeds: false (see section 2). Now per-seed, via `--splits-json`.
- E1's kill criterion "EE_S1(0.1) < 0.05 changes the story" contradicted my own critique (zero-variance input); removed. S1 is now a control.
- `equivariance_error` docstring claimed to divide by ||f(u)||; the code divides by ||rho(g) f(u)|| (as does `rel`); the Piano A formula uses ||f(u)||.
  Docstring fixed; for S1 compare across alpha through `rmse` (rad).
- The G1 run used `--n-test 0` together with the full audit (~33 forward passes over 21,946 graphs per run): now `--no-audit` for the gate and the audit on 2,000.
- "S4 tested" was too strong: `act_flip` flips lines only (`is_line`) [C: code]; transformers are never flipped. Fixed 2026-09-30: probe
  `fliptrafo` (row swap + tap -> 1/tap; the MATPOWER reversal rule is asserted by a test). Trained `m0_seed0`: EE/RMSE ~0.02 [C].

## 5. Experiments

Gates first; nothing below is reportable unless G1-G3 pass.

| id | question | needs | prediction (pre-registered) | kill / interpretation |
|---|---|---|---|---|
| G1 | does our path reproduce the released numbers? | ckpt + full IEEE14 + splits JSON | `genco_audit --no-audit --n-test 0 --splits-json` PBE within 2% of `metrics.csv` "PBE Mean" | fails => loading/normalizer/metric path wrong, stop |
| G2 | official split reproduced, per seed | splits JSON | test size and ids match | assertion in the script |
| G3 | weights load strict | ckpt | keys match | fails => key remap / run on `genco-paper-repro` source via PYTHONPATH |
| E0p | REF != 0, shifters, topology variation in public sets? | 10 partitions/grid | same as local (0 / 0 / none) | any shift != 0 => S2 unlocked (`act_shift`, plan F1); CV(|Yff|) > 0 => canon not an exact identity in-dist |
| E1 | EE of released Base ckpts | 12 base ckpts | S3: EE small (<= 0.1 for k in 0.1..10), larger at 0.01/100 (PoC M0: 0.02-0.12); S4 on lines ~1e-6 | S3 EE >= 0.3 at k=10 => convention matters more than the PoC suggested |
| E1-S1 | control only | same | large by construction: REF-angle input has zero variance in training | not a finding about the architecture |
| E1c | capacity effect | case14 small/tiny | none pre-registered (3 sizes, 3 seeds: exploratory) | report as exploratory |
| E2 | training-free `Canonicalize` on released ckpts | same | in-dist RMSE within 1e-3 relative of raw (as on `m0_seed0`); EE <= 1e-5 on S1/S3/S4(lines); raw RMSE on T_g degrades, canon's does not | in-dist change > 1e-3 => wrapper not free (scale_ref fit) |
| E5 | true zero-shot: source-fixed vs target-refit normalizer; does EE predict zero-shot error (RQ2/H2)? | Tier B ckpts + partial data of the other grids | convention changes some physical-unit metric by >30% for most (ckpt, target) pairs (preview: up to 2x, sign varies) | all ratios within 10% => convention irrelevant, drop the leakage concern. H2: Spearman rho(EE_S3, zero-shot error) over (ckpt, target) pairs with partial correlation on (source, target, size); \|rho\| < 0.3 => H2 rejected in this population |
| E6a | is the wide-augmentation failure the 1/k physics loss? | local, training | aug wide with physics weight 0 trains (VA ~ M0's 0.5 deg) | still fails => cause is not the loss (needs a `--physics-weight` flag in `run.py`, not written) |
| E6b | fair M0+Aug baseline: covariant physics loss (residual x k) | local, training | aug wide then trains and matches `augmild`-level in-dist | implemented 2026-09-30 as the `augcov` arm (`CovariantAugment`: loss in the sample's frame, k per batch), not trained yet; until then M0+Aug (k 0.1..10) is a handicapped baseline [C: VA 5.6 vs 0.51 deg, 2/3 seeds stop at epoch 41; cause [I]] |

E1/E2 are run as raw and `--canon` on identical scenarios; `audit()` already records RMSE on every transformed set T_g. Canon at a different grid needs the
*source* constant (`--scale-ref` from the source run's json; the script refuses `--canon --target` without it).

**Deferred (needs external disk / GPU, decide after E1/E2/E5):**

- E3 slack-convention / cross-grid shift with `genco-pfdelta` ckpts (IEEE118 -> IEEE57/GOC500, PFDelta task 3.1; branch `genco-paper-repro-pfdelta`;
  ckpts 1.93 GB, `pfdelta_task3.1` 1.05 GB, cache unknown). Only informative if a PFDelta scenario has theta_ref != 0 [?]. `data/pfdelta` (20 GB, local)
  is the original PFDelta format, not readable by `data_audit.py`.
- E4 training: `canon` arm at the data sizes of `genco-pf-transfer` (100 ... 250k) against the released from-scratch baselines (no M0 retraining). Not
  feasible on MPS beyond ~10k scenarios.
- ~~S4 on transformers~~ (done 2026-09-30, `fliptrafo`); S2 needs a grid with phase shifters (none in 7/7 datakit grids).

### Amendment 2026-10-01 (cluster run, written before any audit result)

Execution moved to the abacus cluster (`cluster/*.sbatch`): full public datasets on GPFS, so every checkpoint is
audited on its *own* official test split. `subset_data.py official` copies, per grid, every seed's full test split
plus the first 5000 train / 500 val ids (renumbered; `splits/<size>_seed<s>.json`), so G1 runs on the complete
official test split and E1/E2 on its first 2000 graphs. All 36 released IEEE checkpoints (14/30/57/118 ×
base/small/tiny × seeds 0/1/42), not 12 + 6: H2 lists size as a covariate. GOC500 and the Tier B partial-data
shortcut are dropped.

Corrections found while preparing it:
- §4 G1 bullet is itself wrong: `pf_task.test_step` inverse-transforms batch and outputs *before* the residuals,
  so the released "PBE Mean" is in MVA. G1 compares `accuracy()["PBE"] × baseMVA` with it (pass within 2%).
- E5's kill criterion (|ρ| < 0.3 rejects H2) predates the 2026-09-30 revision of H2, which predicts ρ ≈ 0 when
  the shift has no orbit component. Replaced by the three checks below.
- The target refit is *exactly* an S3 action with k = baseMVA_refit / baseMVA_source on every model-visible input
  (Pd, Qd, Qg, Gs, Bs, gen Pg, Y; all other rescaled columns, `vn_kv` included, are masked in PF). Hence for every
  channel |RMSE_refit − RMSE_source| ≤ EE_S3(k) on the same graphs (triangle inequality), with no modelling
  assumption. It is a consistency check of the pipeline, not a test of H2.

E5 analysis, fixed now:
- E5a (must hold): |ΔRMSE_c| ≤ EE_S3,c(k_pair) · (1 + 1e-4) for all 108 (checkpoint, target) pairs and channels.
  A violation means the action, the normalizer or the scoring disagree: stop and debug.
- E5b (must hold): canon zero-shot is identical under source and refit normalizers (relative difference < 1e-4).
- E5c (H2, exploratory): Spearman ρ between EE_S3,VM(k = 10) on the target and the zero-shot VM RMSE (source
  normalizer), over the 108 pairs, raw and with ranks residualized on (source grid, target grid, size). Revised H2
  predicts |ρ_partial| < 0.3: the cross-grid shift is mostly physical (on the section), not along the S3 orbit.
- E2 on N-1 data: if E0p finds CV(mean |Yff|) > 1e-3, post-hoc `Canonicalize` is not the identity in distribution
  and its in-dist cost is reported, not assumed zero; its EE ≤ 1e-5 prediction is unaffected.

### Amendment 2026-10-01 22:00, after seeing the 27 pairs with a case14 source (not the other 81)

E5c on those 27 pairs: ρ = 0.99, residualized 0.99, against the predicted |ρ_partial| < 0.3. The assumption behind
the prediction ("the cross-grid shift is mostly on the section") does not hold for these checkpoints: post-hoc
canon lowers their zero-shot VM error in every case14-sourced row, i.e. the target sits far along the S3 orbit of
a model whose EE_S3 is 40-2000x its in-dist RMSE. Two readings remain: the orbit mechanism of the revised H2
(EE measures the part of the zero-shot error that the S3 frame removes) or generic instability (a model that is
sensitive to any input change is also wrong on any shifted input). E5d separates them on the **81 held-out pairs**
(sources case30/57/118), with EE = EE_S3,VM(k = 10) on the target and errors = zero-shot VM RMSE:
- E5d-1: ρ_partial(EE, raw error) > 0.5 (E5c replicates on new sources);
- E5d-2 (orbit): ρ_partial(EE, raw − canon error) > 0.5;
- E5d-3 (orbit vs instability): ρ_partial(EE, canon error) < ρ_partial(EE, raw error). Instability predicts the
  two are close (canon's input is just another shifted input); the orbit mechanism predicts the canon one is lower.

## 6. Consequences for the text of Piano A

Applied in the plan revision of 2026-09-30, together with the conceptual corrections (canonicalization, Prop. 5-6).

1. §3 GENCO row ("input p.u. a base fissa: rompe S1-S4") is not what the evidence says: S1 = untrained REF-angle input (theta_ref = 0 in 7/7 grids); S3 =
   the base is *per-grid, data-fitted* (baseMVA 82.3 case14, 40.0 case30, 110.8 case57 vs `baseMVA_orig` 100) and PoC EE_S3 is 2-12%; S4 = exact on lines,
   transformers untested, edges bidirectional [I: paper]; S2 = unobservable.
2. §9 risk "tutti i dataset a 100 MVA rendono S3 sintetica" holds for S1 and S2 as well: 3 of 4 symmetries are only synthetic inside datakit, so real
   evidence must come from ENGAGE or other conventions. H4 gains weight.
3. §6/§7.3 M0+Aug (alpha in [0,2pi), k in [1e-2,1e2]) is a handicapped baseline unless the physics loss is made covariant (E6): RQ3/H1 would otherwise
   compare equivariance against a baseline that fails for optimization reasons.
4. §4 H2 ("rho > 0.7 tra modelli e seed") needs a defined population and controls; E5 is one (36 released ckpts, 3 target grids).
5. §5 EE formula and code differ in the denominator; S1 should be reported in radians. §7.1 "stessi casi dei rilasci GENCO": released PF ckpts cover
   IEEE 14/30/57/118 and GOC 500/2000/10000 only.

## 7. Fetch manifest (sizes from the HF API, `?blobs=true`) [C]

Tier A, IEEE14 complete (G1, G2, E1, E1c, E2): checkpoints `case14_ieee/{base,small,tiny}/seed{0,1,42}` 318 MB (base 241.4, small 61.1, tiny 15.8);
dataset `pf_small_case14_ieee` bus+branch+gen 0.31 GB; splits JSON ~2 MB per seed; local processed cache ~1.8 GB [I]. **~2.5 GB.**
Tier B, larger grids on ~2,000 scenarios (E0p, E1, E2, E5): base ckpts x3 seeds for case30/57/118/GOC500 = 4 x 241.4 = 966 MB; 10 partitions of
bus+branch+gen: IEEE118 ~31 MB, GOC500 ~131 MB, IEEE30/57 not measured (smaller). **~1.2 GB + cache.**
Not fetched: GOC2000 (dataset 102 GB), GOC10000, PFDelta, transfer (6.84 GB), OPF repos.

```bash
# paths stay under the gitignored data/; hf flags copied from the model card, partition patterns untested
G=data/genco; C=data/genco_ckpt
hf download gridfm/genco-pf-datakit --include "case14_ieee/*/seed*/*" \
  --include "mlflow/eval/case14_ieee/*/seed*/*/artifacts/stats/*" --local-dir $C
mkdir -p $G/case14_ieee/raw
hf download gridfm/pf_small_case14_ieee --repo-type dataset --local-dir $G/case14_ieee/raw \
  --include "bus_data.parquet/*" --include "branch_data.parquet/*" --include "gen_data.parquet/*" --include "n_scenarios.txt"
# Tier B: N=118 (same for case30_ieee, case57_ieee, case500_goc); only partitions 0..9
inc=(); for p in $(seq 0 9); do for t in bus branch gen; do inc+=(--include "${t}_data.parquet/scenario_partition=$p/*"); done; done
hf download gridfm/pf_small_case118_ieee --repo-type dataset --local-dir $G/case118_ieee/raw "${inc[@]}"
hf download gridfm/genco-pf-datakit --include "case118_ieee/base/seed*/*" --local-dir $C
```

## 8. Run

```bash
D=$C/case14_ieee/base/seed0; R=experiments/ac_pf_symmetries/results/genco; SPL=$(ls $C/mlflow/eval/case14_ieee/base/seed0/*/artifacts/stats/*_scenario_splits.json)
A="-m experiments.ac_pf_symmetries.genco_audit --config $D/case14_ieee_base_seed0.yaml --data-path $G --model-path $D/best_model_state_dict.pt --normalizer-stats $D/normalizer_stats.pt"
# G1/G2: accuracy only, full official test split, compare PBE with $D/metrics.csv
.venv-screening/bin/python $A --no-audit --n-test 0 --splits-json $SPL --out $R/g1_case14_base_seed0.json
# E1/E2: audit on 2000 test graphs, raw and canon
for canon in "" "--canon"; do .venv-screening/bin/python $A $canon --n-test 2000 --splits-json $SPL --out $R/case14_base_seed0${canon:+_canon}.json; done
# E5 (Tier B data present): source-fixed vs refit on another grid; canon needs the source constant from the canon run's json
.venv-screening/bin/python $A --no-audit --target case118_ieee --target-stats source --out $R/e5_case14_to_case118_source.json
.venv-screening/bin/python $A --no-audit --target case118_ieee --target-stats refit  --out $R/e5_case14_to_case118_refit.json
# E0 on the partial public data
.venv-screening/bin/python -m experiments.ac_pf_symmetries.data_audit --root $G --max-partitions 10 --out experiments/ac_pf_symmetries/results/data_audit_public.json
```

Not done: timing of a full-test-split accuracy pass; an aggregation script for `results/genco/*.json` (extend `report.py` once results exist); E6 flags.
