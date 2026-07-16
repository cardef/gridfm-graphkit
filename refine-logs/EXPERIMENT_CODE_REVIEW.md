# Experiment Code Review

**Date:** 2026-07-15
**Skill:** `/experiment-bridge`
**Verdict:** PASS for code readiness; R014 remains BLOCKED for experiment authorization.

## Scope

Independent review covered the complete communication-only confirmatory path:
geometry, model seam, target-safe data preparation, training/evaluation phase
separation, immutable evidence, campaign validation, and locked analysis.

## Blocking findings resolved

1. Main-CLI checkpoint/output overrides now create three distinct evaluation
   artifacts.
2. Training configs contain source networks only and use `--train_only`.
   Target-only evaluation is a separate mode that is unavailable until all 20
   hashed training seals exist.
3. Confirmatory datasets do not filter scenarios using solved outputs. Target
   membership is inherited from the pre-generation inventory hash; the fixed
   integrity rule rejects and records a whole topology before target freeze.
4. The real `python -m gridfm_graphkit` import path passes the legacy-module
   denial, and the confirmatory task path has no legacy hierarchy/normalizer.
5. Checkpoints and terminal artifacts use exclusive writes; the FLOP ledger is
   atomically replaced during training; launch/evaluation locks prevent
   concurrent reuse. Every terminal record binds its checkpoint, ledger,
   runtime, metric, campaign, data, split, and geometry hashes.
6. Evaluation requires the exact frozen target scenario set and rejects extra
   records. Analysis revalidates seals and hashes before reading metrics.
7. Geometry identity excludes measured timings. Partial target construction
   failures are typed, retained in the denominator, and force C1/C2 to FAIL.
8. Size terciles/extrapolation are derived and independently recomputed.
9. Datakit generation is bound to the exact sibling fork, uniform policy,
   executing-process sidecar, pre-generation inventory, and I001 commit.
10. Gate evidence is typed and rehashes underlying inputs. R003 additionally
    recomputes the `GeometryBudget` hash, deterministic candidate choice, and
    equality with the geometry bundle's sole policy hash.
11. A001--A004 retain component errors, scenario P95, seed dispersion, exact
    tests, wild-cluster sensitivity, resource evidence, and failures.

## Verification

- Independent final rereview: PASS.
- Full repository suite: 116 passed, 184 warnings, 311.28 s before the final
  contract-only hardening; changed bridge subsets were rerun afterward.
- Focused post-review bridge suite: 42 passed, 19 warnings.
- Final reviewer targeted suite: 26 passed.
- Pre-commit on changed bridge paths: Ruff, formatter, flake8, YAML,
  trailing-comma, secret scan, and Syncthing-boundary hooks passed.
- Current I001 preflight: BLOCKED as intended because `../.venv` imports a
  nested datakit worktree instead of the exact sibling root and both working
  trees are not yet clean confirmatory commits.

PASS means the experiment code and fail-closed gates are prepared. It does not
turn unexecuted I/R/C/P/S evidence into PASS, authorize E001--E020, or support
an efficacy claim.
