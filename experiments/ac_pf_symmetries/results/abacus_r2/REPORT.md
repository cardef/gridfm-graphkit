# R2: M0+Canon vs M1 on held-out grids

Runs: heldout_case30_ieee canon 3, heldout_case30_ieee m1fullcanon 3, heldout_case57_ieee canon 3, heldout_case57_ieee m1fullcanon 3

### heldout_case30_ieee: zero-shot

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|
| case30_ieee | canon | 0.0109 ± 0.0019 | 5.34 ± 1 | 31.3 ± 13 | 9.71 ± 3.8 | 5.56 ± 0.19 | 4.67 ± 1.1 | 1.36 ± 0.14 |
| case30_ieee | m1fullcanon | 0.00817 ± 0.0027 | 3.45 ± 2.7 | 45.8 ± 39 | 8.04 ± 1.8 | 2.8 ± 0.63 | 3.21 ± 2.6 | 0.727 ± 0.41 |

### heldout_case30_ieee: in distribution

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|
| case14_ieee | canon | 0.000689 ± 8.7e-05 | 0.158 ± 0.096 | 1.91 ± 1.2 | 2.33 ± 0.41 | 0.18 ± 0.054 | 0.151 ± 0.092 | 0.043 ± 0.024 |
| case14_ieee | m1fullcanon | 0.000808 ± 0.0002 | 0.167 ± 0.034 | 1.96 ± 0.37 | 2.77 ± 0.55 | 0.145 ± 0.032 | 0.16 ± 0.032 | 0.0446 ± 0.0091 |
| case57_ieee | canon | 0.0042 ± 0.00033 | 0.289 ± 0.046 | 5.27 ± 1.3 | 17 ± 0.48 | 0.259 ± 0.029 | 0.242 ± 0.05 | 0.0817 ± 0.0034 |
| case57_ieee | m1fullcanon | 0.00407 ± 0.00037 | 0.217 ± 0.03 | 3.74 ± 0.21 | 13.5 ± 0.83 | 0.2 ± 0.013 | 0.154 ± 0.033 | 0.0753 ± 0.0053 |
| case118_ieee | canon | 0.00156 ± 0.00027 | 1.65 ± 0.31 | 72.8 ± 13 | 9.26 ± 2 | 1.2 ± 0.18 | 1.4 ± 0.3 | 0.251 ± 0.038 |
| case118_ieee | m1fullcanon | 0.00192 ± 0.00021 | 0.771 ± 0.08 | 27.3 ± 1.4 | 10.9 ± 0.04 | 0.607 ± 0.092 | 0.55 ± 0.06 | 0.128 ± 0.012 |

### heldout_case57_ieee: zero-shot

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|
| case57_ieee | canon | 0.018 ± 0.0027 | 8.26 ± 0.63 | 285 ± 14 | 66.2 ± 22 | 10 ± 1.9 | 7.35 ± 0.88 | 2.06 ± 0.067 |
| case57_ieee | m1fullcanon | 0.0177 ± 0.0018 | 5.55 ± 2.1 | 186 ± 73 | 70.1 ± 21 | 8.74 ± 1.5 | 4.84 ± 2.2 | 1.35 ± 0.32 |

### heldout_case57_ieee: in distribution

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | θ_f − θ_t (deg) |
|---|---|---|---|---|---|---|---|---|
| case14_ieee | canon | 0.000474 ± 0.00013 | 0.112 ± 0.0046 | 1.46 ± 0.17 | 1.6 ± 0.51 | 0.127 ± 0.0065 | 0.106 ± 0.0046 | 0.0311 ± 0.001 |
| case14_ieee | m1fullcanon | 0.00067 ± 0.00031 | 0.209 ± 0.047 | 2.47 ± 0.51 | 2.23 ± 0.73 | 0.155 ± 0.035 | 0.199 ± 0.045 | 0.0548 ± 0.012 |
| case30_ieee | canon | 0.00406 ± 0.00061 | 0.422 ± 0.038 | 4.74 ± 0.37 | 5.91 ± 3.2 | 0.227 ± 0.0084 | 0.409 ± 0.04 | 0.0753 ± 0.0039 |
| case30_ieee | m1fullcanon | 0.00443 ± 0.00038 | 0.241 ± 0.034 | 2.94 ± 0.48 | 3.49 ± 1.3 | 0.205 ± 0.023 | 0.229 ± 0.033 | 0.0488 ± 0.0064 |
| case118_ieee | canon | 0.00178 ± 0.00048 | 1.87 ± 0.18 | 75.8 ± 4.1 | 11.1 ± 3.4 | 1.09 ± 0.036 | 1.57 ± 0.13 | 0.27 ± 0.02 |
| case118_ieee | m1fullcanon | 0.00194 ± 0.00043 | 0.889 ± 0.08 | 33.8 ± 4.6 | 11.5 ± 2.7 | 0.626 ± 0.056 | 0.685 ± 0.062 | 0.141 ± 0.014 |

**P17 (held-out grids): PASS** (predicted no separation). case30_ieee VA m1 [0.7911, 7.07] vs canon [3.89, 6.169]; case30_ieee VM m1 [0.005591, 0.01188] vs canon [0.008638, 0.01324]; case57_ieee VA m1 [3.006, 8.083] vs canon [7.437, 8.971]; case57_ieee VM m1 [0.01588, 0.02022] vs canon [0.01489, 0.02155]

**P18 (training grids): FAIL** (predicted no separation). case14_ieee VA m1 [0.1317, 0.2125] vs canon [0.08881, 0.294]; case57_ieee VA m1 [0.1872, 0.2575] vs canon [0.227, 0.3351]; case118_ieee VA m1 [0.6861, 0.8782] vs canon [1.233, 1.99] SEPARATED; case14_ieee VA m1 [0.1419, 0.2455] vs canon [0.1081, 0.118] SEPARATED; case30_ieee VA m1 [0.1942, 0.2701] vs canon [0.3837, 0.4738] SEPARATED; case118_ieee VA m1 [0.7903, 0.9855] vs canon [1.655, 2.086] SEPARATED
