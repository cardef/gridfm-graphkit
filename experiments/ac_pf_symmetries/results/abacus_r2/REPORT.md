# R2: M0+Canon vs M1 on held-out grids

Runs: heldout_case118_ieee canon 3, heldout_case118_ieee m1fullcanon 3, heldout_case30_ieee canon 3, heldout_case30_ieee m1fullcanon 3, heldout_case57_ieee canon 3, heldout_case57_ieee m1fullcanon 3

### heldout_case118_ieee: zero-shot

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|---|
| case118_ieee | canon | 0.00632 ± 0.00064 | 11.5 ± 1 | 435 ± 2.1e+02 | 42.6 ± 22 | 16.7 ± 2.7 | 7.13 ± 1.6 | 8.89 ± 0.47 | 2.02 ± 0.19 |
| case118_ieee | m1fullcanon | 0.00731 ± 0.0011 | 11.2 ± 1.1 | 317 ± 1.7e+02 | 78.2 ± 53 | 21.2 ± 3.7 | 6.85 ± 1.5 | 8.84 ± 0.22 | 2.07 ± 0.11 |

### heldout_case118_ieee: in distribution

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|---|
| case14_ieee | canon | 0.00345 ± 0.0012 | 0.333 ± 0.049 | 2.17 ± 0.43 | 13.7 ± 4.9 | 0.15 ± 0.021 | 0.328 ± 0.05 | 0.0535 ± 0.0096 | 0.0939 ± 0.019 |
| case14_ieee | m1fullcanon | 0.00082 ± 0.00034 | 0.198 ± 0.087 | 2.44 ± 0.87 | 2.31 ± 1 | 0.161 ± 0.032 | 0.188 ± 0.083 | 0.0622 ± 0.025 | 0.0535 ± 0.022 |
| case30_ieee | canon | 0.0058 ± 0.00033 | 0.649 ± 0.009 | 4.06 ± 0.67 | 22.2 ± 2 | 0.183 ± 0.0086 | 0.641 ± 0.0073 | 0.0979 ± 0.013 | 0.112 ± 0.0014 |
| case30_ieee | m1fullcanon | 0.00384 ± 9.3e-05 | 0.244 ± 0.046 | 2.69 ± 0.66 | 4.09 ± 0.21 | 0.182 ± 0.018 | 0.233 ± 0.044 | 0.0749 ± 0.013 | 0.0491 ± 0.0077 |
| case57_ieee | canon | 0.00413 ± 0.00032 | 0.249 ± 0.03 | 4.38 ± 0.92 | 16.5 ± 0.82 | 0.207 ± 0.015 | 0.203 ± 0.029 | 0.145 ± 0.013 | 0.0746 ± 0.0066 |
| case57_ieee | m1fullcanon | 0.00397 ± 0.00031 | 0.216 ± 0.021 | 3.89 ± 0.43 | 12.4 ± 1.1 | 0.186 ± 0.016 | 0.156 ± 0.03 | 0.147 ± 0.0039 | 0.0747 ± 0.00056 |

### heldout_case30_ieee: zero-shot

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|---|
| case30_ieee | canon | 0.0109 ± 0.0019 | 5.34 ± 1 | 31.3 ± 13 | 9.71 ± 3.8 | 5.56 ± 0.19 | 4.67 ± 1.1 | 2.55 ± 0.2 | 1.36 ± 0.14 |
| case30_ieee | m1fullcanon | 0.00817 ± 0.0027 | 3.45 ± 2.7 | 45.8 ± 39 | 8.04 ± 1.8 | 2.8 ± 0.63 | 3.21 ± 2.6 | 1.16 ± 0.58 | 0.727 ± 0.41 |

### heldout_case30_ieee: in distribution

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|---|
| case14_ieee | canon | 0.000689 ± 8.7e-05 | 0.158 ± 0.096 | 1.91 ± 1.2 | 2.33 ± 0.41 | 0.18 ± 0.054 | 0.151 ± 0.092 | 0.046 ± 0.026 | 0.043 ± 0.024 |
| case14_ieee | m1fullcanon | 0.000808 ± 0.0002 | 0.167 ± 0.034 | 1.96 ± 0.37 | 2.77 ± 0.55 | 0.145 ± 0.032 | 0.16 ± 0.032 | 0.0483 ± 0.011 | 0.0446 ± 0.0091 |
| case57_ieee | canon | 0.0042 ± 0.00033 | 0.289 ± 0.046 | 5.27 ± 1.3 | 17 ± 0.48 | 0.259 ± 0.029 | 0.242 ± 0.05 | 0.156 ± 0.0072 | 0.0817 ± 0.0034 |
| case57_ieee | m1fullcanon | 0.00407 ± 0.00037 | 0.217 ± 0.03 | 3.74 ± 0.21 | 13.5 ± 0.83 | 0.2 ± 0.013 | 0.154 ± 0.033 | 0.152 ± 0.011 | 0.0753 ± 0.0053 |
| case118_ieee | canon | 0.00156 ± 0.00027 | 1.65 ± 0.31 | 72.8 ± 13 | 9.26 ± 2 | 1.2 ± 0.18 | 1.4 ± 0.3 | 0.878 ± 0.12 | 0.251 ± 0.038 |
| case118_ieee | m1fullcanon | 0.00192 ± 0.00021 | 0.771 ± 0.08 | 27.3 ± 1.4 | 10.9 ± 0.04 | 0.607 ± 0.092 | 0.55 ± 0.06 | 0.539 ± 0.061 | 0.128 ± 0.012 |

### heldout_case57_ieee: zero-shot

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|---|
| case57_ieee | canon | 0.018 ± 0.0027 | 8.26 ± 0.63 | 285 ± 14 | 66.2 ± 22 | 10 ± 1.9 | 7.35 ± 0.88 | 3.72 ± 0.31 | 2.06 ± 0.067 |
| case57_ieee | m1fullcanon | 0.0177 ± 0.0018 | 5.55 ± 2.1 | 186 ± 73 | 70.1 ± 21 | 8.74 ± 1.5 | 4.84 ± 2.2 | 2.59 ± 0.4 | 1.35 ± 0.32 |

### heldout_case57_ieee: in distribution

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|---|
| case14_ieee | canon | 0.000474 ± 0.00013 | 0.112 ± 0.0046 | 1.46 ± 0.17 | 1.6 ± 0.51 | 0.127 ± 0.0065 | 0.106 ± 0.0046 | 0.0354 ± 0.0017 | 0.0311 ± 0.001 |
| case14_ieee | m1fullcanon | 0.00067 ± 0.00031 | 0.209 ± 0.047 | 2.47 ± 0.51 | 2.23 ± 0.73 | 0.155 ± 0.035 | 0.199 ± 0.045 | 0.0624 ± 0.014 | 0.0548 ± 0.012 |
| case30_ieee | canon | 0.00406 ± 0.00061 | 0.422 ± 0.038 | 4.74 ± 0.37 | 5.91 ± 3.2 | 0.227 ± 0.0084 | 0.409 ± 0.04 | 0.103 ± 0.0044 | 0.0753 ± 0.0039 |
| case30_ieee | m1fullcanon | 0.00443 ± 0.00038 | 0.241 ± 0.034 | 2.94 ± 0.48 | 3.49 ± 1.3 | 0.205 ± 0.023 | 0.229 ± 0.033 | 0.0748 ± 0.0092 | 0.0488 ± 0.0064 |
| case118_ieee | canon | 0.00178 ± 0.00048 | 1.87 ± 0.18 | 75.8 ± 4.1 | 11.1 ± 3.4 | 1.09 ± 0.036 | 1.57 ± 0.13 | 1 ± 0.14 | 0.27 ± 0.02 |
| case118_ieee | m1fullcanon | 0.00194 ± 0.00043 | 0.889 ± 0.08 | 33.8 ± 4.6 | 11.5 ± 2.7 | 0.626 ± 0.056 | 0.685 ± 0.062 | 0.565 ± 0.051 | 0.141 ± 0.014 |

**P17 (held-out grids, folds 1-2): PASS** (predicted no separation). case30_ieee VA m1 [0.7911, 7.07] vs canon [3.89, 6.169]; case30_ieee VM m1 [0.005591, 0.01188] vs canon [0.008638, 0.01324]; case57_ieee VA m1 [3.006, 8.083] vs canon [7.437, 8.971]; case57_ieee VM m1 [0.01588, 0.02022] vs canon [0.01489, 0.02155]

**P18 (training grids, folds 1-2): FAIL** (predicted no separation). case14_ieee VA m1 [0.1317, 0.2125] vs canon [0.08881, 0.294]; case57_ieee VA m1 [0.1872, 0.2575] vs canon [0.227, 0.3351]; case118_ieee VA m1 [0.6861, 0.8782] vs canon [1.233, 1.99] SEPARATED; case14_ieee VA m1 [0.1419, 0.2455] vs canon [0.1081, 0.118] SEPARATED; case30_ieee VA m1 [0.1942, 0.2701] vs canon [0.3837, 0.4738] SEPARATED; case118_ieee VA m1 [0.7903, 0.9855] vs canon [1.655, 2.086] SEPARATED

**P20 (case118 held out, VA): FAIL** (predicted m1 entirely below canon): m1 [10.37, 12.74] vs canon [10.06, 12.41] deg

**P21 (not scored: P20 failed): —** m1/canon ratio of the offset 0.96, of the rest 0.99 (predicted offset ratio below)
