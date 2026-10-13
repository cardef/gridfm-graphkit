# X3: GENCO with the injection decoder on the 31 public subgrids, zero-shot (pianox_pt_a)

Seeds per (arm, normalizer): dcinjcanon/source 3, dcinjcanon/refit 3

## Training stability

| run | epochs | best val. loss (epoch) | max val. loss after epoch 5 | collapsed |
|---|---|---|---|---|
| dcinjcanon_seed0 | 100 | 0.0863 (97) | 0.422 | False |
| dcinjcanon_seed1 | 100 | 0.0792 (94) | 1.12 | True |
| dcinjcanon_seed2 | 100 | 0.0731 (97) | 0.262 | False |

## case118_ieee

References (VA deg): DC 1.95, fdpf inputs k=5 0.235, fdpf given k=5 0.000521

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcinjcanon | source | [1.51, 1.88] | [1.14, 1.53] | [0.997, 1.1] | [0.458, 0.472] | [0.0011, 0.00235] | [9.22, 9.86] | [0.778, 0.967] |
| dcinjcanon | refit | [1.51, 1.88] | [1.14, 1.53] | [0.997, 1.1] | [0.458, 0.472] | [0.0011, 0.00235] | [9.22, 9.86] | [0.778, 0.967] |
| public finetune100 | own stats | 4.56 | 3.04 | 3.41 | 0.812 | 0.00253 | 5.13 | 2.34 |
| public finetune1000 | own stats | 1.52 | 1.17 | 0.974 | 0.231 | 0.000373 | 1.3 | 0.78 |
| public pretrained | target | 11.3 | 8.46 | 7.5 | 1.98 | 0.00218 | 11 | 5.80 |

## case118_tr

References (VA deg): DC 2.3, fdpf inputs k=5 0.301, fdpf given k=5 0.00945

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcinjcanon | source | [1.83, 2.22] | [1.48, 1.89] | [1.07, 1.17] | [0.473, 0.492] | [0.00144, 0.00251] | [9.43, 10] | [0.798, 0.966] |
| dcinjcanon | refit | [1.83, 2.22] | [1.48, 1.89] | [1.07, 1.17] | [0.473, 0.492] | [0.00144, 0.00251] | [9.43, 10] | [0.798, 0.966] |
| public finetune100 | own stats | 7.77 | 5.89 | 5.07 | 1.19 | 0.00324 | 5.87 | 3.39 |
| public finetune1000 | own stats | 2.9 | 2.34 | 1.71 | 0.445 | 0.0011 | 2.08 | 1.27 |
| public pretrained | target | 13.3 | 11.2 | 7.2 | 2.11 | 0.00287 | 11.7 | 5.80 |

## case500_goc

References (VA deg): DC 2.52, fdpf inputs k=5 0.0709, fdpf given k=5 0.000537

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcinjcanon | source | [1.09, 3.71] | [0.489, 3.5] | [0.974, 1.23] | [0.267, 0.684] | [0.0148, 0.0258] | [17.4, 23.1] | [0.433, 1.47] |
| dcinjcanon | refit | [1.09, 3.71] | [0.489, 3.5] | [0.974, 1.23] | [0.267, 0.684] | [0.0148, 0.0258] | [17.4, 23.1] | [0.433, 1.47] |
| public pretrained | target | 6.28 | 5.5 | 3.02 | 0.661 | 0.00593 | 22.1 | 2.49 |

## case2000_goc

References (VA deg): DC 9.75, fdpf inputs k=5 0.329, fdpf given k=5 0.00611

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcinjcanon | source | [3.18, 6.95] | [2.62, 6.69] | [1.81, 2.31] | [0.241, 0.251] | [0.0294, 0.0412] | [21.5, 24.8] | [0.326, 0.713] |
| dcinjcanon | refit | [3.18, 6.95] | [2.62, 6.69] | [1.81, 2.31] | [0.241, 0.251] | [0.0294, 0.0412] | [21.5, 24.8] | [0.326, 0.713] |
| public pretrained | target | 13.4 | 10.6 | 8.26 | 1.04 | 0.0196 | 72.6 | 1.38 |

## case57_ieee

References (VA deg): DC 1.08, fdpf inputs k=5 0.115, fdpf given k=5 0.0691

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcinjcanon | source | [1.06, 1.08] | [0.362, 0.387] | [0.998, 1.01] | [0.527, 0.531] | [0.00787, 0.0118] | [6.83, 7.58] | [0.983, 0.996] |
| dcinjcanon | refit | [1.06, 1.08] | [0.362, 0.387] | [0.998, 1.01] | [0.527, 0.531] | [0.00787, 0.0118] | [6.83, 7.58] | [0.983, 0.996] |
| public pretrained | target | 2.6 | 1.06 | 2.38 | 1.23 | 0.0267 | 6.69 | 2.41 |

## case30_ieee

References (VA deg): DC 1.56, fdpf inputs k=5 0.565, fdpf given k=5 0.00869

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcinjcanon | source | [1.54, 1.57] | [1.48, 1.52] | [0.389, 0.399] | [0.337, 0.34] | [0.00918, 0.016] | [3.6, 4.4] | [0.984, 1.01] |
| dcinjcanon | refit | [1.54, 1.57] | [1.48, 1.52] | [0.389, 0.399] | [0.337, 0.34] | [0.00918, 0.016] | [3.6, 4.4] | [0.984, 1.01] |
| public pretrained | target | 9.41 | 9.18 | 2.06 | 1.68 | 0.0042 | 5.1 | 6.03 |

## case14_ieee

References (VA deg): DC 1.51, fdpf inputs k=5 0.429, fdpf given k=5 0.0148

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcinjcanon | source | [1.5, 1.51] | [1.46, 1.47] | [0.363, 0.367] | [0.444, 0.445] | [0.00276, 0.00432] | [4.26, 5.16] | [0.995, 1] |
| dcinjcanon | refit | [1.5, 1.51] | [1.46, 1.47] | [0.363, 0.367] | [0.444, 0.445] | [0.00276, 0.00432] | [4.26, 5.16] | [0.995, 1] |
| public pretrained | target | 8.36 | 8.04 | 2.3 | 2.21 | 0.00362 | 9.65 | 5.54 |

## Verdicts

**X3a-P0 (no dcinjcanon seed collapses: validation loss at most 0.5 after epoch 5): FAIL** dcinjcanon_seed0 max 0.422; dcinjcanon_seed1 max 1.12; dcinjcanon_seed2 max 0.262

**X3a-P1 (dcinjcanon below DC on both case118 sets, every seed): PASS** case118_ieee [1.51, 1.88] vs DC 1.95; case118_tr [1.83, 2.22] vs DC 2.3

**X3a-P2 (dcinjcanon below DC on case500_goc, every seed): FAIL** [1.09, 3.71] vs DC 2.52

**X3a-P4 (on case2000_goc at least one dcinjcanon seed above DC): FAIL** [3.18, 6.95] vs DC 9.75

**X3a-P5 (no learned arm below fast-decoupled from the same inputs after 5 iterations on 118 / 500 / 2000): PASS** case118_ieee/dcinjcanon/source min 1.51 vs 0.235; case500_goc/dcinjcanon/source min 1.09 vs 0.0709; case2000_goc/dcinjcanon/source min 3.18 vs 0.329
