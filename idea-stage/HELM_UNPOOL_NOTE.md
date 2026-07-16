# HELM-Unpool: a nonlinear upgrade path for the Kron-Schur prolongation

**Date:** 2026-07-07 · **Status:** derivation verified on paper + numerically validated locally (pilot: `idea-stage/helm_unpool_pilot.py`, results: `idea-stage/helm_unpool_results.json`) · **Sources read:** `papers/1706.06622.pdf` (MDHEM, §II–III), `papers/2403.00400.pdf` (nonlinear Kron, §1–2), `papers/1512.08250.pdf` (constant-power-load reduction, §I–III), `papers/2308.06888.pdf` (FASCD, §1–2).

## 1. Problem

The KS U-Net unpool is `V_i = P V_b + V_aff` with `P = −Y_ii⁻¹ Y_ib` (sparsified top-k) and
`V_aff = Y_ii⁻¹ conj(S_I)` (`hierarchy.py:243-250`). Exactness requires interior injections to be
*currents* known a priori; AC PF has constant-**power** interior injections `I_i = conj(S_i)/conj(V_i)`,
so `V_aff` implicitly approximates `conj(V_i) ≈ 1∠0` (flat). The question: is there a principled,
cheap, fixed-operator way past this linearization?

## 2. What the literature actually offers (papers read, scoped)

- **MDHEM (Liu et al., arXiv:1706.06622, §II Table I–II, eq. 4–10)** — canonical HELM: embed PQ
  injections as `s·S*/V*(s*)`, expand `V(s) = Σ c_n sⁿ`, reciprocal series `W*(s)` with recurrence
  `W*[0]=1/V*[0]`, `W*[n] = −Σ_{τ<n} W*[τ]V*[n−τ]/V*[0]` (eq. 7–8). One linear solve with the *same*
  matrix per order. Padé for convergence near collapse. **PV buses need 3 extra convolutions + a Q
  series (§III-D) — our PQ-only interior avoids all of that.** [confirmed by reading]
- **Nonlinear Kron (van der Schaft et al., arXiv:2403.00400, §1–2)** — exact interior elimination for
  networks with *nonlinear monotone edge* relations, **but assumes zero nodal currents at interior
  nodes and real potentials** (§2.1). Our nonlinearity is at *nodes* (complex constant-power): not a
  drop-in; theoretical anchor only. Their Remark 1.1 cites a companion result that nonlinear Kron is
  often *impossible* for non-smooth edge laws — the exactness door is narrow. [confirmed]
- **Monshizadeh et al. (arXiv:1512.08250, §II–III)** — exact reduction *with* constant-power loads via
  projected incidence matrix, but in the lossless, constant-|V|, sine-coupled swing model (nonlinear
  DC-like structure). Existence proof under special structure; not the full AC PF. [confirmed]
- **FAS/τ-correction (Brandt 1977; modern: Bueler & Farrell arXiv:2308.06888)** — the numerical-analysis
  remedy: keep the coarse problem nonlinear, correct its RHS with `τ = A_c(R u_f) − R A_f(u_f)` so the
  coarse equation is consistent with the fine residual. (τ formalism is textbook; the paper was read
  for currency of the FAS family, §1.) [standard result]

## 3. Derivation: boundary-conditioned HELM = the exact nonlinear unpool

**Setup.** Interior PQ-only (by boundary construction). True interior equations, given boundary
voltage `V_b`, with full `Y_ii` (shunts included), all p.u. on case base:

```
Y_ii V_i + Y_ib V_b = conj(S_I) ⊘ conj(V_i)          (*)
```

**Embedding.** `Y_ii V(s) + Y_ib V_b = s · conj(S_I) ⊙ W(s)`, `W(s) := 1/conj(V)(s)` (reciprocal
series, MDHEM eq. 7). At `s=0`: linear; at `s=1`: exactly (*). Matching orders:

```
c_0 = −Y_ii⁻¹ Y_ib V_b                                = P V_b   (harmonic extension = the GERM)
w_0 = 1 ⊘ conj(c_0);   w_n = −(1⊘conj(c_0)) Σ_{τ=0}^{n−1} w_τ ⊙ conj(c_{n−τ})
c_{n+1} = Y_ii⁻¹ (conj(S_I) ⊙ w_n)                    (one solve per order, SAME LU)
```

**Identifications** (all verified numerically, §5):
- Current `V_aff` = `c_1` with `w_0` replaced by `1` — i.e. the order-1 term evaluated at flat voltage.
- `c_0 + c_1` = exactly one implicit-Z-bus fixed-point step `V ← P V_b + Y_ii⁻¹(conj(S_I) ⊘ conj(V))`
  started at `c_0` (FP1 ≡ HELM1; FP-k ≠ HELM-k beyond first order).
- **τ-connection:** exact boundary equation after elimination is
  `Y_red V_b = I_B(V_b) − Y_bi Y_ii⁻¹ I_I(V_i)`. The cached coarse injection
  `I_c = conj(S_B_masked) − Y_bi V_aff` is this with `I_I` at flat voltage; the HELM series gives a
  computable τ-style correction `δI_c = −Y_bi (c_1 + c_2 + …)`-terms — same solves upgrade **both**
  unpool and the coarse physical channel.

**Sanity:** `S_I→0 ⇒ c_n≥1 = 0` (recovers harmonic extension) ✓; units `[Y⁻¹][S/V]=[V]` ✓; guard
`|c_0| ≫ 0` (measured min 0.909 across all grids/scenarios) ✓. Convergence at `s=1` is *inherited
plausibly* from HELM theory but Stahl's theorem is proven for the standard global embedding, **not**
this boundary-conditioned one → treated as an empirical question (answered below at nominal loading;
open near collapse).

## 4. Pilot design

`idea-stage/helm_unpool_pilot.py`, 7 local grids (case14/30/57/118/500/2000, Texas2k), ≤256
scenarios/grid, TRUE `V_b` (operator ceiling, matches R002's diagnostic convention). Hard sanity:
median residual of (*) on the datakit solutions is **1e-13…1e-15 on every grid** — conventions exact.
Cross-check: pilot's `REPO_Psp` (sparsified P + current `V_aff`) reproduces the *recorded* R002
`vhat_abs_err_median` for case2000 to 3 digits (3.09e-2), so the pilot sits on the registered baseline.

## 5. Results (median |V̂−V| p.u., interior; full table in JSON)

| grid | H0 (I=0) | REPO (=v_aff) | HELM1 | HELM2 | HELM3 | HELM2, P-sparse | Jacobi-32 | M-sparse k32 |
|---|---|---|---|---|---|---|---|---|
| case14 | 3.5e-2 | 8.8e-3 | 1.7e-3 | 1.1e-4 | 1.2e-5 | 1.1e-4 | 4.1e-4 | 1.1e-4 |
| case30 | 7.3e-2 | 1.7e-2 | 5.1e-3 | 4.4e-4 | 6.4e-5 | 4.4e-4 | 2.4e-2 | 4.4e-4 |
| case57 | 9.6e-2 | 9.2e-3 | 9.3e-3 | 1.2e-3 | 2.7e-4 | 1.2e-3 | 4.3e-2 | 9.8e-3 |
| case118 | 1.4e-2 | 2.2e-3 | 2.0e-4 | 3.5e-6 | 1.1e-7 | 3.5e-6 | 4.2e-6 | 3.5e-6 |
| case500 | 7.8e-2 | 2.6e-2 | 8.1e-3 | 7.7e-4 | 1.7e-4 | **5.4e-3** | 3.0e-2 | 9.5e-3 |
| case2000 | 6.9e-2 | 3.0e-2 | 5.3e-3 | 3.9e-4 | 6.6e-5 | **2.6e-2** | 4.4e-2 | 2.1e-2 |
| Texas2k | 1.0e-1 | 1.3e-2 | 1.2e-2 | 1.3e-3 | 3.2e-4 | **8.0e-3** | 4.7e-2 | 3.5e-2 |

**Findings:**

1. **F1 — the series is the right object.** ~1 order of magnitude per order, monotone on all 7 grids.
   HELM2 beats the current `V_aff` by 20–80× at operator level. Padé[1/1] ≈ plain truncation at this
   loading (no collapse proximity); FP2 ≈ HELM2.
2. **F2 — naive runtime paths fail.** (a) Top-k-sparsified `M = Y_ii⁻¹` (the P trick) does NOT
   concentrate: k=64 mass 0.84/0.87 on case2000/Texas2k; error floors at 2–4e-2, *worse than today*.
   (b) Unrolled damped Jacobi (K≤32 sparse matvecs) barely converges on ≥500-bus grids (4.4e-2 at
   case2000). Y_ii is not diagonally dominant enough; K message-passing rounds cannot emulate the solve.
3. **F3 — NEW architecture-level diagnostic, independent of HELM:** on case2000 the **P-sparsification
   error (~2.6e-2 median) already dominates the entire unpool error budget** — same order as the
   linearization error it coexists with. The 95%-|P|-mass gate (passed at 96.6%, k=16) is *not*
   sufficient for reconstruction fidelity there: with exact HELM2 tails the error stays at 2.6e-2 until
   P itself is improved. On ≤500-bus grids and Texas2k, P is faithful enough that HELM2 delivers
   3.5–350× end-to-end. → *The mass gate measures the wrong proxy on case2000; a reconstruction-error
   gate on `P_sp V_b` vs `P V_b` would have caught this at M0.* (Pre-existing flaw, now named.)
4. **F4 — the honest runtime path is exact triangular solves.** Per-grid dense LU factors of `Y_ii`,
   batched `solve_triangular` on GPU: case2000 `n_i≈1.5k` → ~2×36 MB complex64 buffers/grid, O(n_i²·B)
   flops ≪ one GNN layer O(E·h²·B). Feasible to ~case9241 (`n_i≈6.7k` → ~2×0.7 GB: heavy but storable;
   fork to decide there). E005 same-grid batches make this a single batched op.
5. **F5 — data artifact:** one persistent case2000 outlier (max ≈1.07 p.u. in *every* variant incl. H0,
   injection-insensitive) — smells like a dead/degenerate bus or non-converged fast-PF sample; ties into
   the known 25%-yield caveat. Follow-up: locate scenario/bus; do not gate on max.

## 6. Recommendation (ranked, falsifiable)

1. **R-A (do): HELM2 unpool + τ-corrected coarse injections** via precomputed dense triangular factors,
   as an M1 arm: swap only `v_aff`→(c₁+c₂ at runtime) and `I_c`→`conj(S_B) − Y_bi(c₁+c₂-terms)`;
   everything else frozen. Gate: does the fine-level VM/VA RMSE at matched FLOPs move? The GNN residual
   stream may already absorb the linear-prior error — that is exactly the discriminating question, and
   the R005 load-tercile instrument will show it (linearization error grows with load; HELM2 flattens it).
2. **R-B (do first, cheaper): fix the P floor on case2000** — raise k (or switch that grid to dense P:
   1460×540 complex64 ≈ 6 MB, trivially storable) and add a reconstruction-error assert next to the mass
   assert. Without R-B, R-A is capped at 2.6e-2 on the M2-gate grid and will falsely read "no gain".
   **Order matters: R-B before R-A, else R-A is unfalsifiable on case2000.**
3. **R-C (skip for now):** learned/sparsified M, unrolled Jacobi/Chebyshev — measured dead ends at this
   scale. Nonlinear-Kron-exact pooling (van der Schaft) — wrong nonlinearity class for PF.

**Caveats.** All numbers condition on TRUE `V_b`; with a predicted coarse `V_b` the unpool gain matters
only once coarse error ≲ unpool error (measure both in M1). Convergence near voltage collapse is
unverified (high-load tail of cluster data: check `|c_2|/|c_1|` ratio as a divergence canary; fall back
to Padé). Stahl-type guarantee for the boundary-conditioned embedding: open theory question — could be a
paper-side contribution if proven.

## 7. Cost accounting

Offline: unchanged + dense triangular factors per grid (case2000 ≈ 72 MB; case500 ≈ 4 MB). Runtime per
sample: 2 batched triangular-solve pairs + elementwise ops (HELM2) — ≪1% of forward FLOPs at h=48; must
enter `utils/flops.py` accounting (pre-registered) as an unpool-operator term.
