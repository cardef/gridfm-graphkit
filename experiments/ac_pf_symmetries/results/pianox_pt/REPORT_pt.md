# X3: GENCO with the injection decoder on the 31 public subgrids, zero-shot (pianox_pt)

Seeds per (arm, normalizer): plain/source 3, plain/refit 3, dcinjcanon/source 3, dcinjcanon/refit 3, dcinjvref/source 3, dcinjvref/refit 3

## Training stability

| run | epochs | best val. loss (epoch) | max val. loss after epoch 5 | collapsed |
|---|---|---|---|---|
| dcinjcanon_seed0 | 41 | 0.15 (0) | 3.34 | True |
| dcinjcanon_seed1 | 46 | 0.115 (5) | 12.8 | True |
| dcinjcanon_seed2 | 100 | 0.094 (97) | 1.39 | True |
| dcinjvref_seed0 | 100 | 0.0948 (87) | 1.95 | True |
| dcinjvref_seed1 | 63 | 0.119 (22) | 1.11 | True |
| dcinjvref_seed2 | 44 | 0.126 (3) | 1.56 | True |
| plain_seed0 | 52 | 0.182 (11) | 0.401 | False |
| plain_seed1 | 100 | 0.146 (96) | 0.235 | False |
| plain_seed2 | 100 | 0.144 (97) | 0.26 | False |

## case118_ieee

References (VA deg): DC 1.95, fdpf inputs k=5 0.235, fdpf given k=5 0.000521

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| plain | source | [11, 20.4] | [8.42, 17.9] | [7.15, 9.78] | [1.84, 3.87] | [0.0033, 0.005] | [16.2, 49.8] | [5.67, 10.5] |
| plain | refit | [11.4, 26.7] | [9.07, 23.7] | [6.86, 12.3] | [2.05, 5.01] | [0.00365, 0.00628] | [17.3, 68.1] | [5.84, 13.7] |
| dcinjcanon | source | [1.57, 1.89] | [1.19, 1.54] | [1.02, 1.11] | [0.461, 0.475] | [0.00215, 0.00553] | [9.88, 10.7] | [0.804, 0.972] |
| dcinjcanon | refit | [1.57, 1.89] | [1.19, 1.54] | [1.02, 1.11] | [0.461, 0.475] | [0.00215, 0.00553] | [9.88, 10.7] | [0.804, 0.972] |
| dcinjvref | source | [1.49, 1.69] | [1.11, 1.33] | [0.999, 1.05] | [0.458, 0.467] | [0.00308, 0.00643] | [9.88, 12.6] | [0.765, 0.868] |
| dcinjvref | refit | [1.49, 1.69] | [1.11, 1.33] | [0.999, 1.05] | [0.458, 0.467] | [0.00308, 0.00643] | [9.88, 12.6] | [0.765, 0.868] |
| public finetune100 | own stats | 4.56 | 3.04 | 3.41 | 0.812 | 0.00253 | 5.13 | 2.34 |
| public finetune1000 | own stats | 1.52 | 1.17 | 0.974 | 0.231 | 0.000373 | 1.3 | 0.78 |
| public pretrained | target | 11.3 | 8.46 | 7.5 | 1.98 | 0.00218 | 11 | 5.80 |

## case118_tr

References (VA deg): DC 2.3, fdpf inputs k=5 0.301, fdpf given k=5 0.00945

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| plain | source | [13.5, 22] | [11.3, 19.2] | [7.37, 10.7] | [2.15, 4.03] | [0.00386, 0.00529] | [15.4, 46.5] | [5.89, 9.6] |
| plain | refit | [14.1, 28.3] | [12, 24.9] | [7.33, 13.6] | [2.39, 5.28] | [0.00417, 0.00645] | [16.4, 65.2] | [6.12, 12.3] |
| dcinjcanon | source | [1.87, 2.2] | [1.52, 1.87] | [1.09, 1.16] | [0.476, 0.493] | [0.0023, 0.00529] | [10.1, 11.1] | [0.817, 0.956] |
| dcinjcanon | refit | [1.87, 2.2] | [1.52, 1.87] | [1.09, 1.16] | [0.476, 0.493] | [0.0023, 0.00529] | [10.1, 11.1] | [0.817, 0.956] |
| dcinjvref | source | [1.81, 2.01] | [1.46, 1.66] | [1.07, 1.13] | [0.473, 0.484] | [0.0032, 0.00599] | [10.1, 12.8] | [0.789, 0.874] |
| dcinjvref | refit | [1.81, 2.01] | [1.46, 1.66] | [1.07, 1.13] | [0.473, 0.484] | [0.0032, 0.00599] | [10.1, 12.8] | [0.789, 0.874] |
| public finetune100 | own stats | 7.77 | 5.89 | 5.07 | 1.19 | 0.00324 | 5.87 | 3.39 |
| public finetune1000 | own stats | 2.9 | 2.34 | 1.71 | 0.445 | 0.0011 | 2.08 | 1.27 |
| public pretrained | target | 13.3 | 11.2 | 7.2 | 2.11 | 0.00287 | 11.7 | 5.80 |

## case500_goc

References (VA deg): DC 2.52, fdpf inputs k=5 0.0709, fdpf given k=5 0.000537

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| plain | source | [8.02, 9.37] | [7.2, 8.6] | [3.53, 3.73] | [0.697, 0.965] | [0.0151, 0.0239] | [22.8, 30.4] | [3.19, 3.72] |
| plain | refit | [7.81, 9.67] | [6.95, 8.69] | [3.57, 4.29] | [0.784, 1.52] | [0.0158, 0.0221] | [28.6, 42.4] | [3.1, 3.84] |
| dcinjcanon | source | [1.6, 3.21] | [1.42, 3.01] | [0.718, 1.1] | [0.286, 0.322] | [0.0195, 0.0517] | [22.8, 42.9] | [0.634, 1.27] |
| dcinjcanon | refit | [1.6, 3.21] | [1.42, 3.01] | [0.718, 1.1] | [0.286, 0.322] | [0.0195, 0.0517] | [22.8, 42.9] | [0.634, 1.27] |
| dcinjvref | source | [1.19, 3.1] | [1.01, 2.9] | [0.63, 1.08] | [0.252, 0.413] | [0.0143, 0.0264] | [22, 37.1] | [0.474, 1.23] |
| dcinjvref | refit | [1.19, 3.1] | [1.01, 2.9] | [0.63, 1.08] | [0.252, 0.413] | [0.0143, 0.0264] | [22, 37.1] | [0.474, 1.23] |
| public pretrained | target | 6.28 | 5.5 | 3.02 | 0.661 | 0.00593 | 22.1 | 2.49 |

## case2000_goc

References (VA deg): DC 9.75, fdpf inputs k=5 0.329, fdpf given k=5 0.00611

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| plain | source | [15.1, 16.5] | [12.4, 13.9] | [8.52, 8.84] | [0.826, 0.976] | [0.0328, 0.0461] | [24.8, 31.5] | [1.55, 1.69] |
| plain | refit | [15, 16.5] | [12.4, 13.9] | [8.49, 8.84] | [0.822, 0.977] | [0.0325, 0.0458] | [25, 30.8] | [1.54, 1.69] |
| dcinjcanon | source | [3.09, 15.8] | [2.88, 14.9] | [1.13, 5.26] | [0.229, 0.353] | [0.0304, 0.0587] | [31, 79.6] | [0.317, 1.62] |
| dcinjcanon | refit | [3.09, 15.8] | [2.88, 14.9] | [1.13, 5.26] | [0.229, 0.353] | [0.0304, 0.0587] | [31, 79.6] | [0.317, 1.62] |
| dcinjvref | source | [2.25, 6.72] | [1.81, 6.15] | [1.2, 2.71] | [0.229, 0.26] | [0.0291, 0.0401] | [27.5, 43.3] | [0.23, 0.69] |
| dcinjvref | refit | [2.25, 6.72] | [1.81, 6.15] | [1.2, 2.71] | [0.229, 0.26] | [0.0291, 0.0401] | [27.5, 43.3] | [0.23, 0.69] |
| public pretrained | target | 13.4 | 10.6 | 8.26 | 1.04 | 0.0196 | 72.6 | 1.38 |

## case57_ieee

References (VA deg): DC 1.08, fdpf inputs k=5 0.115, fdpf given k=5 0.0691

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| plain | source | [4.27, 8.55] | [3.05, 5.26] | [2.98, 7.54] | [1.58, 4.83] | [0.00895, 0.0326] | [11.4, 25.1] | [3.95, 7.91] |
| plain | refit | [4.29, 9.16] | [3.07, 5.84] | [3, 8.33] | [1.58, 5.3] | [0.00985, 0.0373] | [11.8, 25.5] | [3.97, 8.48] |
| dcinjcanon | source | [1.07, 1.08] | [0.385, 0.425] | [0.986, 0.996] | [0.527, 0.529] | [0.0127, 0.0291] | [6.89, 11] | [0.988, 0.996] |
| dcinjcanon | refit | [1.07, 1.08] | [0.385, 0.425] | [0.986, 0.996] | [0.527, 0.529] | [0.0127, 0.0291] | [6.89, 11] | [0.988, 0.996] |
| dcinjvref | source | [1.06, 1.07] | [0.375, 0.405] | [0.987, 0.999] | [0.525, 0.53] | [0.0139, 0.0486] | [6.94, 7.78] | [0.978, 0.992] |
| dcinjvref | refit | [1.06, 1.07] | [0.375, 0.405] | [0.987, 0.999] | [0.525, 0.53] | [0.0139, 0.0486] | [6.94, 7.78] | [0.978, 0.992] |
| public pretrained | target | 2.6 | 1.06 | 2.38 | 1.23 | 0.0267 | 6.69 | 2.41 |

## case30_ieee

References (VA deg): DC 1.56, fdpf inputs k=5 0.565, fdpf given k=5 0.00869

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| plain | source | [6.65, 19.4] | [4.29, 16.2] | [2.21, 10.8] | [2.02, 6.69] | [0.00994, 0.035] | [7.07, 35.2] | [4.26, 12.5] |
| plain | refit | [4.87, 19.1] | [4.64, 15.9] | [1.47, 10.6] | [1.02, 6.45] | [0.00678, 0.0126] | [6.07, 34.3] | [3.12, 12.3] |
| dcinjcanon | source | [1.53, 1.54] | [1.48, 1.49] | [0.386, 0.389] | [0.335, 0.337] | [0.00844, 0.0358] | [4.2, 6.07] | [0.981, 0.99] |
| dcinjcanon | refit | [1.53, 1.54] | [1.48, 1.49] | [0.386, 0.389] | [0.335, 0.337] | [0.00844, 0.0358] | [4.2, 6.07] | [0.981, 0.99] |
| dcinjvref | source | [1.54, 1.57] | [1.49, 1.52] | [0.39, 0.394] | [0.337, 0.34] | [0.0143, 0.0193] | [3.52, 4.59] | [0.986, 1.01] |
| dcinjvref | refit | [1.54, 1.57] | [1.49, 1.52] | [0.39, 0.394] | [0.337, 0.34] | [0.0143, 0.0193] | [3.52, 4.59] | [0.986, 1.01] |
| public pretrained | target | 9.41 | 9.18 | 2.06 | 1.68 | 0.0042 | 5.1 | 6.03 |

## case14_ieee

References (VA deg): DC 1.51, fdpf inputs k=5 0.429, fdpf given k=5 0.0148

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| plain | source | [4.04, 11.4] | [1.82, 11] | [2.92, 3.61] | [2.66, 3.08] | [0.00605, 0.00843] | [14.6, 30.9] | [2.68, 7.56] |
| plain | refit | [4.84, 11.1] | [3.99, 10.7] | [2.75, 3.85] | [2.13, 3] | [0.00318, 0.006] | [12.5, 17.9] | [3.21, 7.38] |
| dcinjcanon | source | [1.5, 1.52] | [1.46, 1.47] | [0.366, 0.367] | [0.444, 0.447] | [0.00444, 0.0219] | [5.23, 8.64] | [0.997, 1.01] |
| dcinjcanon | refit | [1.5, 1.52] | [1.46, 1.47] | [0.366, 0.367] | [0.444, 0.447] | [0.00444, 0.0219] | [5.23, 8.64] | [0.997, 1.01] |
| dcinjvref | source | [1.51, 1.53] | [1.46, 1.48] | [0.366, 0.367] | [0.446, 0.449] | [0.00265, 0.012] | [4.51, 5.82] | [1, 1.01] |
| dcinjvref | refit | [1.51, 1.53] | [1.47, 1.48] | [0.366, 0.367] | [0.446, 0.449] | [0.00265, 0.012] | [4.51, 5.82] | [1, 1.01] |
| public pretrained | target | 8.36 | 8.04 | 2.3 | 2.21 | 0.00362 | 9.65 | 5.54 |

## Verdicts

Stability (descriptive, X3 registered no stability prediction; X3a-P0's criterion): 3 of 3 dcinjcanon seeds collapse; dcinjcanon_seed0 max 3.34; dcinjcanon_seed1 max 12.8; dcinjcanon_seed2 max 1.39

**X3-P1 (dcinjcanon below DC on both case118 sets, every seed): PASS** case118_ieee [1.57, 1.89] vs DC 1.95; case118_tr [1.87, 2.2] vs DC 2.3

**X3-P2 (dcinjcanon below DC on case500_goc, every seed): FAIL** [1.6, 3.21] vs DC 2.52

**X3-P3 (plain above DC on case118_ieee and case500_goc, every seed, either normalizer): PASS** case118_ieee/source [11, 20.4]; case118_ieee/refit [11.4, 26.7]; case500_goc/source [8.02, 9.37]; case500_goc/refit [7.81, 9.67]

**X3-P4 (on case2000_goc at least one dcinjcanon seed above DC): PASS** [3.09, 15.8] vs DC 9.75

**X3-P5 (no learned arm below fast-decoupled from the same inputs after 5 iterations on 118 / 500 / 2000): PASS** case118_ieee/plain/source min 11 vs 0.235; case118_ieee/plain/refit min 11.4 vs 0.235; case118_ieee/dcinjcanon/source min 1.57 vs 0.235; case118_ieee/dcinjvref/source min 1.49 vs 0.000521; case500_goc/plain/source min 8.02 vs 0.0709; case500_goc/plain/refit min 7.81 vs 0.0709; case500_goc/dcinjcanon/source min 1.6 vs 0.0709; case500_goc/dcinjvref/source min 1.19 vs 0.000537; case2000_goc/plain/source min 15.1 vs 0.329; case2000_goc/plain/refit min 15 vs 0.329; case2000_goc/dcinjcanon/source min 3.09 vs 0.329; case2000_goc/dcinjvref/source min 2.25 vs 0.00611
