# X3: GENCO with the injection decoder on the 31 public subgrids, zero-shot (pianox_pt_loopg)

Seeds per (arm, normalizer): dcloopgcanon/source 3, dcloopgcanon/refit 3, dcloopgvref/source 3, dcloopgvref/refit 3

## Training stability

| run | epochs | best val. loss (epoch) | max val. loss after epoch 5 | collapsed |
|---|---|---|---|---|
| dcloopgcanon_seed0 | 100 | 0.169 (97) | 3.87 | True |
| dcloopgcanon_seed1 | 100 | 0.0808 (77) | 1.98 | True |
| dcloopgcanon_seed2 | 100 | 0.0688 (81) | 4.39 | True |
| dcloopgvref_seed0 | 85 | 0.137 (44) | 1.82 | True |
| dcloopgvref_seed1 | 100 | 0.0792 (76) | 2.43 | True |
| dcloopgvref_seed2 | 100 | 0.0654 (75) | 5.15 | True |

## case118_ieee

References (VA deg): DC 1.95, fdpf inputs k=5 0.235, fdpf given k=5 0.000521

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcloopgcanon | source | [0.544, 1.75] | [0.542, 1.74] | [0.0548, 0.378] | [0.108, 0.353] | [0.000978, 0.00369] | [0.785, 6.76] | [0.279, 0.9] |
| dcloopgcanon | refit | [0.544, 1.75] | [0.542, 1.74] | [0.0548, 0.378] | [0.108, 0.353] | [0.000978, 0.00369] | [0.785, 6.76] | [0.279, 0.9] |
| dcloopgvref | source | [0.0549, 0.115] | [0.0433, 0.0953] | [0.0338, 0.0646] | [0.0195, 0.0462] | [0.000589, 0.00115] | [0.749, 1.67] | [0.0282, 0.0591] |
| dcloopgvref | refit | [0.0549, 0.115] | [0.0433, 0.0953] | [0.0338, 0.0646] | [0.0195, 0.0462] | [0.000589, 0.00115] | [0.749, 1.67] | [0.0282, 0.0591] |
| public finetune100 | own stats | 4.56 | 3.04 | 3.41 | 0.812 | 0.00253 | 5.13 | 2.34 |
| public finetune1000 | own stats | 1.52 | 1.17 | 0.974 | 0.231 | 0.000373 | 1.3 | 0.78 |
| public pretrained | target | 11.3 | 8.46 | 7.5 | 1.98 | 0.00218 | 11 | 5.80 |

## case118_tr

References (VA deg): DC 2.3, fdpf inputs k=5 0.301, fdpf given k=5 0.00945

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcloopgcanon | source | [0.796, 2.16] | [0.759, 2.13] | [0.18, 0.383] | [0.176, 0.442] | [0.00129, 0.00574] | [1.69, 3.14] | [0.347, 0.942] |
| dcloopgcanon | refit | [0.796, 2.16] | [0.759, 2.13] | [0.18, 0.383] | [0.176, 0.442] | [0.00129, 0.00574] | [1.69, 3.13] | [0.347, 0.942] |
| dcloopgvref | source | [0.108, 0.137] | [0.0849, 0.101] | [0.0661, 0.0935] | [0.0414, 0.0519] | [0.000781, 0.00118] | [1.34, 1.62] | [0.0469, 0.0598] |
| dcloopgvref | refit | [0.107, 0.137] | [0.0847, 0.101] | [0.066, 0.0936] | [0.0413, 0.0519] | [0.000781, 0.00118] | [1.34, 1.62] | [0.0468, 0.0599] |
| public finetune100 | own stats | 7.77 | 5.89 | 5.07 | 1.19 | 0.00324 | 5.87 | 3.39 |
| public finetune1000 | own stats | 2.9 | 2.34 | 1.71 | 0.445 | 0.0011 | 2.08 | 1.27 |
| public pretrained | target | 13.3 | 11.2 | 7.2 | 2.11 | 0.00287 | 11.7 | 5.80 |

## case500_goc

References (VA deg): DC 2.52, fdpf inputs k=5 0.0709, fdpf given k=5 0.000537

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcloopgcanon | source | [0.463, 0.798] | [0.389, 0.654] | [0.252, 0.457] | [0.0713, 0.129] | [0.0178, 0.0188] | [10.9, 15.2] | [0.184, 0.317] |
| dcloopgcanon | refit | [0.463, 0.798] | [0.389, 0.654] | [0.251, 0.457] | [0.0713, 0.129] | [0.0178, 0.0188] | [10.9, 15.2] | [0.184, 0.317] |
| dcloopgvref | source | [0.596, 0.873] | [0.492, 0.719] | [0.337, 0.495] | [0.097, 0.155] | [0.0178, 0.0191] | [12.1, 19.2] | [0.237, 0.347] |
| dcloopgvref | refit | [0.596, 0.873] | [0.492, 0.719] | [0.337, 0.495] | [0.097, 0.155] | [0.0178, 0.0191] | [12.1, 19.1] | [0.237, 0.347] |
| public pretrained | target | 6.28 | 5.5 | 3.02 | 0.661 | 0.00593 | 22.1 | 2.49 |

## case2000_goc

References (VA deg): DC 9.75, fdpf inputs k=5 0.329, fdpf given k=5 0.00611

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcloopgcanon | source | [2.87, 4.14] | [2.74, 3.96] | [0.842, 1.43] | [0.128, 0.167] | [0.0293, 0.0333] | [16, 17.2] | [0.294, 0.425] |
| dcloopgcanon | refit | [2.87, 4.14] | [2.74, 3.96] | [0.842, 1.43] | [0.128, 0.167] | [0.0293, 0.0333] | [16, 17.2] | [0.294, 0.425] |
| dcloopgvref | source | [2.92, 3.72] | [2.8, 3.6] | [0.781, 1.13] | [0.127, 0.165] | [0.0316, 0.0369] | [13.7, 21.6] | [0.3, 0.382] |
| dcloopgvref | refit | [2.92, 3.72] | [2.8, 3.59] | [0.781, 1.13] | [0.127, 0.165] | [0.0316, 0.0369] | [13.7, 21.6] | [0.3, 0.382] |
| public pretrained | target | 13.4 | 10.6 | 8.26 | 1.04 | 0.0196 | 72.6 | 1.38 |

## case57_ieee

References (VA deg): DC 1.08, fdpf inputs k=5 0.115, fdpf given k=5 0.0691

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcloopgcanon | source | [0.447, 3.31] | [0.246, 3.22] | [0.29, 0.769] | [0.184, 0.975] | [0.011, 0.0212] | [2.76, 11] | [0.414, 3.06] |
| dcloopgcanon | refit | [0.447, 3.31] | [0.246, 3.22] | [0.29, 0.769] | [0.184, 0.975] | [0.011, 0.0212] | [2.75, 11] | [0.414, 3.06] |
| dcloopgvref | source | [0.325, 0.521] | [0.14, 0.276] | [0.272, 0.485] | [0.137, 0.239] | [0.00758, 0.00946] | [1.87, 3.64] | [0.301, 0.482] |
| dcloopgvref | refit | [0.325, 0.521] | [0.14, 0.276] | [0.272, 0.485] | [0.137, 0.239] | [0.00758, 0.00946] | [1.87, 3.63] | [0.301, 0.482] |
| public pretrained | target | 2.6 | 1.06 | 2.38 | 1.23 | 0.0267 | 6.69 | 2.41 |

## case30_ieee

References (VA deg): DC 1.56, fdpf inputs k=5 0.565, fdpf given k=5 0.00869

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcloopgcanon | source | [0.505, 4.52] | [0.436, 4.51] | [0.254, 0.341] | [0.239, 1.06] | [0.00672, 0.0151] | [3.27, 8.63] | [0.323, 2.9] |
| dcloopgcanon | refit | [0.505, 4.52] | [0.436, 4.51] | [0.254, 0.341] | [0.239, 1.06] | [0.00672, 0.0151] | [3.27, 8.63] | [0.323, 2.9] |
| dcloopgvref | source | [0.378, 1.24] | [0.34, 1.19] | [0.166, 0.349] | [0.148, 0.289] | [0.00276, 0.00676] | [2.17, 3.09] | [0.243, 0.795] |
| dcloopgvref | refit | [0.378, 1.24] | [0.34, 1.19] | [0.166, 0.349] | [0.148, 0.289] | [0.00276, 0.00676] | [2.17, 3.09] | [0.243, 0.795] |
| public pretrained | target | 9.41 | 9.18 | 2.06 | 1.68 | 0.0042 | 5.1 | 6.03 |

## case14_ieee

References (VA deg): DC 1.51, fdpf inputs k=5 0.429, fdpf given k=5 0.0148

| model | normalizer | VA (deg) | offset | rest | θ_f − θ_t | VM, PQ | PBE (MVA) | VA / DC |
|---|---|---|---|---|---|---|---|---|
| dcloopgcanon | source | [1.06, 4.26] | [1.03, 4.25] | [0.151, 0.326] | [0.427, 1.35] | [0.00368, 0.0104] | [6.28, 11.5] | [0.7, 2.82] |
| dcloopgcanon | refit | [1.06, 4.26] | [1.03, 4.25] | [0.151, 0.326] | [0.427, 1.35] | [0.00368, 0.0104] | [6.28, 11.5] | [0.7, 2.82] |
| dcloopgvref | source | [0.0932, 0.539] | [0.0648, 0.506] | [0.0669, 0.186] | [0.077, 0.246] | [0.000459, 0.00529] | [1.05, 3.83] | [0.0618, 0.357] |
| dcloopgvref | refit | [0.0932, 0.539] | [0.0649, 0.506] | [0.0669, 0.186] | [0.077, 0.246] | [0.000459, 0.00529] | [1.04, 3.83] | [0.0618, 0.357] |
| public pretrained | target | 8.36 | 8.04 | 2.3 | 2.21 | 0.00362 | 9.65 | 5.54 |

## Verdicts

**X4b-P0 (no closed-loop seed above 0.5 in validation after epoch 5): FAIL** dcloopgcanon_seed0 max 3.87; dcloopgcanon_seed1 max 1.98; dcloopgcanon_seed2 max 4.39; dcloopgvref_seed0 max 1.82; dcloopgvref_seed1 max 2.43; dcloopgvref_seed2 max 5.15

**X4b-P1 (dcloopgcanon below DC on 118, 118_tr, 500 and 2000, every seed): PASS** case118_ieee [0.544, 1.75] vs DC 1.95; case118_tr [0.796, 2.16] vs DC 2.3; case500_goc [0.463, 0.798] vs DC 2.52; case2000_goc [2.87, 4.14] vs DC 9.75

**X4b-P2 (dcloopgcanon on case500_goc below all six open-loop dcinjcanon seeds of X3 and X3a): PASS** [0.463, 0.798] vs open-loop min 1.09 (open-loop seeds 1.09, 1.6, 1.91, 2.84, 3.21, 3.71)

**X4b-P3 (no closed-loop seed below fast-decoupled from the same inputs after 5 iterations on 118 / 500 / 2000): PASS** case118_ieee/dcloopgcanon min 0.544 vs 0.235; case118_ieee/dcloopgvref min 0.0549 vs 0.000521; case500_goc/dcloopgcanon min 0.463 vs 0.0709; case500_goc/dcloopgvref min 0.596 vs 0.000537; case2000_goc/dcloopgcanon min 2.87 vs 0.329; case2000_goc/dcloopgvref min 2.92 vs 0.00611
