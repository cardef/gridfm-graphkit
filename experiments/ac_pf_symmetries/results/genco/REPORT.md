# GENCO released checkpoints: Piano A F0 audit (cluster run)

### Gates (G3 = strict load, asserted by the script)

| checkpoint | test graphs (ours / official) | PBE ours (MVA) | PBE released (MVA) | rel | G1+G2 | released baseMVA | refit on 5000 train ids vs released |
|---|---|---|---|---|---|---|---|
| case14_ieee_base_s0 | 21946 / 21946 | 0.002691 | 0.002682 | +0.31% | pass | 84.6 | +0.14% |
| case14_ieee_base_s1 | 21943 / 21943 | 0.003889 | 0.003886 | +0.08% | pass | 84.66 | +0.20% |
| case14_ieee_base_s42 | 21945 / 21945 | 0.003148 | 0.003134 | +0.46% | pass | 84.59 | -0.16% |
| case14_ieee_small_s0 | 21946 / 21946 | 0.003022 | 0.003029 | -0.24% | pass | 84.6 | +0.14% |
| case14_ieee_small_s1 | 21943 / 21943 | 0.003346 | 0.003346 | +0.02% | pass | 84.66 | +0.20% |
| case14_ieee_small_s42 | 21945 / 21945 | 0.002986 | 0.002989 | -0.10% | pass | 84.59 | -0.16% |
| case14_ieee_tiny_s0 | 21946 / 21946 | 0.005124 | 0.005129 | -0.09% | pass | 84.6 | +0.14% |
| case14_ieee_tiny_s1 | 21943 / 21943 | 0.004535 | 0.00453 | +0.12% | pass | 84.66 | +0.20% |
| case14_ieee_tiny_s42 | 21945 / 21945 | 0.003873 | 0.003878 | -0.13% | pass | 84.59 | -0.16% |
| case30_ieee_base_s0 | 21758 / 21758 | 0.005199 | 0.005193 | +0.12% | pass | 59.26 | -0.45% |
| case30_ieee_base_s1 | 21758 / 21758 | 0.003968 | 0.003951 | +0.44% | pass | 59.7 | -0.48% |
| case30_ieee_base_s42 | 21758 / 21758 | 0.01168 | 0.01167 | +0.05% | pass | 59.57 | -0.08% |
| case30_ieee_small_s0 | 21758 / 21758 | 0.003828 | 0.00383 | -0.05% | pass | 59.26 | -0.45% |
| case30_ieee_small_s1 | 21758 / 21758 | 0.0035 | 0.0035 | +0.00% | pass | 59.7 | -0.48% |
| case30_ieee_small_s42 | 21758 / 21758 | 0.004781 | 0.00478 | +0.03% | pass | 59.57 | -0.08% |
| case30_ieee_tiny_s0 | 21758 / 21758 | 0.01116 | 0.01116 | -0.01% | pass | 59.26 | -0.45% |
| case30_ieee_tiny_s1 | 21758 / 21758 | 0.009194 | 0.009195 | -0.01% | pass | 59.7 | -0.48% |
| case30_ieee_tiny_s42 | 21758 / 21758 | 0.006472 | 0.006474 | -0.02% | pass | 59.57 | -0.08% |
| case57_ieee_base_s0 | 21694 / 21694 | 0.01438 | 0.01438 | +0.03% | pass | 111.8 | -0.09% |
| case57_ieee_base_s1 | 21686 / 21686 | 0.01661 | 0.01661 | +0.02% | pass | 111.7 | -0.22% |
| case57_ieee_base_s42 | 21670 / 21670 | 0.02183 | 0.02182 | +0.04% | pass | 112.1 | +0.33% |
| case57_ieee_small_s0 | 21694 / 21694 | 0.019 | 0.019 | -0.01% | pass | 111.8 | -0.09% |
| case57_ieee_small_s1 | 21686 / 21686 | 0.0222 | 0.0222 | -0.01% | pass | 111.7 | -0.22% |
| case57_ieee_small_s42 | 21670 / 21670 | 0.01795 | 0.01795 | -0.02% | pass | 112.1 | +0.33% |
| case57_ieee_tiny_s0 | 21694 / 21694 | 0.02666 | 0.02667 | -0.01% | pass | 111.8 | -0.09% |
| case57_ieee_tiny_s1 | 21686 / 21686 | 0.03394 | 0.03394 | -0.00% | pass | 111.7 | -0.22% |
| case57_ieee_tiny_s42 | 21670 / 21670 | 0.02917 | 0.02917 | -0.00% | pass | 112.1 | +0.33% |
| case118_ieee_base_s0 | 21908 / 21908 | 0.1084 | 0.1084 | +0.00% | pass | 137.9 | +0.35% |
| case118_ieee_base_s1 | 21910 / 21910 | 0.1148 | 0.1148 | +0.00% | pass | 137.6 | -0.10% |
| case118_ieee_base_s42 | 21912 / 21912 | 0.2393 | 0.2393 | +0.00% | pass | 137.9 | +0.29% |
| case118_ieee_small_s0 | 21908 / 21908 | 0.1284 | 0.1284 | -0.00% | pass | 137.9 | +0.35% |
| case118_ieee_small_s1 | 21910 / 21910 | 0.1367 | 0.1367 | -0.00% | pass | 137.6 | -0.10% |
| case118_ieee_small_s42 | 21912 / 21912 | 0.1004 | 0.1004 | -0.00% | pass | 137.9 | +0.29% |
| case118_ieee_tiny_s0 | 21908 / 21908 | 0.1735 | 0.1735 | +0.00% | pass | 137.9 | +0.35% |
| case118_ieee_tiny_s1 | 21910 / 21910 | 0.1113 | 0.1113 | -0.00% | pass | 137.6 | -0.10% |
| case118_ieee_tiny_s42 | 21912 / 21912 | 0.1225 | 0.1225 | -0.00% | pass | 137.9 | +0.29% |

0 checkpoint(s) fail G1/G2.

### E0p: conventions in the public datasets (all scenarios)

| grid | scenarios | ref_va_deg | shift_nonzero_branches | tap_ne_1_branches | tap_gt_1_branches | scenarios_with_branch_outage | gens_out_of_service | mean_abs_Yff_cv_across_scenarios |
|---|---|---|---|---|---|---|---|---|
| case14_ieee | 219606 | [0.0, 0.0] | 0 | 630138 | 0 | 181182 | 38424 | 0.04734316908761198 |
| case30_ieee | 217778 | [0.0, 0.0] | 0 | 850897 | 0 | 192816 | 24962 | 0.03169483386428372 |
| case57_ieee | 216960 | [0.0, 0.0] | 0 | 3215478 | 428545 | 201507 | 15453 | 0.01856647025678265 |
| case118_ieee | 219103 | [0.0, 0.0] | 0 | 1963510 | 0 | 168342 | 50761 | 0.015057769145139807 |

### E1: EE_g / in-dist RMSE of the released checkpoints (mean ± std over seeds 0/1/42)

| checkpoint | phase 3.14159 (VA) | phase 0.1 (VA) | scale 0.1 (VM) | scale 10 (VM) | scale 0.01 (VM) | scale 100 (VM) | flip 0.5 (VA) | fliptrafo 1 (VM) |
|---|---|---|---|---|---|---|---|---|
| case14_ieee base | 6.68e+03 ± 4.3e+03 | 104 ± 80 | 2.34e+03 ± 1.6e+03 | 44.1 ± 21 | 2.55e+03 ± 1.3e+03 | 63.4 ± 16 | 0.000305 ± 0.00019 | 0.358 ± 0.19 |
| case14_ieee small | 5.46e+03 ± 4.4e+03 | 66.6 ± 39 | 1.47e+03 ± 60 | 42.5 ± 7.1 | 1.65e+03 ± 98 | 61.6 ± 9.2 | 0.000212 ± 0.00018 | 0.2 ± 0.057 |
| case14_ieee tiny | 5.3e+03 ± 2.9e+03 | 127 ± 1.1e+02 | 976 ± 5.4e+02 | 96.3 ± 43 | 1.06e+03 ± 6.4e+02 | 124 ± 63 | 0.000281 ± 0.0001 | 0.101 ± 0.029 |
| case30_ieee base | 4.54e+03 ± 7.2e+02 | 87.1 ± 14 | 560 ± 3.6e+02 | 41 ± 8.2 | 624 ± 4.2e+02 | 57.1 ± 5 | 0.000356 ± 8.8e-05 | 0.331 ± 0.15 |
| case30_ieee small | 6.17e+03 ± 4.2e+02 | 123 ± 14 | 1.33e+03 ± 2.7e+02 | 86.9 ± 27 | 1.4e+03 ± 3.5e+02 | 77.8 ± 5.9 | 0.000314 ± 3.3e-05 | 0.225 ± 0.061 |
| case30_ieee tiny | 4.05e+03 ± 8.8e+02 | 87.1 ± 17 | 494 ± 4.5e+02 | 38.9 ± 5.7 | 572 ± 5.5e+02 | 61.3 ± 18 | 0.00031 ± 3.4e-05 | 0.0541 ± 0.039 |
| case57_ieee base | 2.84e+03 ± 2.6e+02 | 21.3 ± 3.3 | 580 ± 79 | 448 ± 1.2e+02 | 570 ± 1e+02 | 151 ± 27 | 0.000276 ± 1.9e-05 | 0.256 ± 0.061 |
| case57_ieee small | 2.69e+03 ± 1.9e+02 | 18.2 ± 5.1 | 468 ± 45 | 311 ± 31 | 478 ± 47 | 90.1 ± 39 | 0.000266 ± 3e-05 | 0.368 ± 0.065 |
| case57_ieee tiny | 1.99e+03 ± 2.4e+02 | 12.7 ± 1.3 | 377 ± 42 | 115 ± 49 | 411 ± 54 | 68.5 ± 16 | 0.000272 ± 2.9e-05 | 0.219 ± 0.091 |
| case118_ieee base | 309 ± 52 | 3.85 ± 0.93 | 193 ± 1.5e+02 | 11.1 ± 0.95 | 220 ± 1.6e+02 | 21.3 ± 9.3 | 0.000109 ± 1.2e-05 | 0.129 ± 0.05 |
| case118_ieee small | 304 ± 58 | 4.32 ± 0.62 | 87 ± 90 | 29.7 ± 14 | 95 ± 1.1e+02 | 46.2 ± 27 | 8.1e-05 ± 3e-05 | 0.114 ± 0.037 |
| case118_ieee tiny | 342 ± 52 | 3.49 ± 0.36 | 127 ± 73 | 23.9 ± 16 | 144 ± 83 | 48.5 ± 26 | 0.000171 ± 3.3e-05 | 0.0706 ± 0.036 |

### Released checkpoints, in-dist RMSE on the first 2000 official test graphs

| checkpoint | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) |
|---|---|---|---|---|---|
| case14_ieee base | 0.000237 ± 7.2e-05 | 0.0629 ± 0.062 | 0.241 ± 0.26 | 0.813 ± 0.17 | 0.00325 ± 0.00066 |
| case14_ieee small | 0.000228 ± 5.5e-05 | 0.0644 ± 0.05 | 0.226 ± 0.2 | 0.771 ± 0.12 | 0.00305 ± 0.00035 |
| case14_ieee tiny | 0.000287 ± 6e-05 | 0.0485 ± 0.034 | 0.176 ± 0.12 | 0.988 ± 0.11 | 0.00459 ± 0.00065 |
| case30_ieee base | 0.000316 ± 4.9e-05 | 0.0394 ± 0.0069 | 0.202 ± 0.093 | 1.61 ± 0.22 | 0.00704 ± 0.0035 |
| case30_ieee small | 0.000242 ± 1.8e-05 | 0.0287 ± 0.0021 | 0.123 ± 0.018 | 1.25 ± 0.085 | 0.00407 ± 0.00065 |
| case30_ieee tiny | 0.000353 ± 6.3e-05 | 0.044 ± 0.0079 | 0.239 ± 0.045 | 1.73 ± 0.3 | 0.00898 ± 0.0018 |
| case57_ieee base | 0.000605 ± 7.2e-05 | 0.0619 ± 0.0054 | 1.02 ± 0.084 | 4.24 ± 0.38 | 0.0179 ± 0.0033 |
| case57_ieee small | 0.000705 ± 4e-06 | 0.0655 ± 0.0042 | 1.05 ± 0.076 | 4.63 ± 0.24 | 0.0199 ± 0.0019 |
| case57_ieee tiny | 0.00102 ± 5.8e-05 | 0.0887 ± 0.011 | 1.43 ± 0.21 | 5.73 ± 0.62 | 0.03 ± 0.0034 |
| case118_ieee base | 0.00143 ± 0.00028 | 0.54 ± 0.076 | 16.1 ± 2.1 | 9.54 ± 1.9 | 0.153 ± 0.06 |
| case118_ieee small | 0.00117 ± 6.3e-05 | 0.5 ± 0.086 | 14.7 ± 2.4 | 7.7 ± 0.46 | 0.123 ± 0.017 |
| case118_ieee tiny | 0.00107 ± 7.6e-05 | 0.461 ± 0.093 | 13.9 ± 2.8 | 7.01 ± 0.51 | 0.134 ± 0.024 |

### E2: post-hoc Canonicalize, in-dist change relative to raw, and its worst EE (rel)

| checkpoint | VM | VA | PG | QG | PBE | max EE rel |
|---|---|---|---|---|---|---|
| case14_ieee base | +11.03% ± 13.6% | +77.22% ± 100.3% | +98.88% ± 105.4% | +7.69% ± 4.3% | +45.24% ± 4.8% | 1.3e-05 |
| case14_ieee small | +16.59% ± 24.1% | +70.11% ± 89.6% | +112.20% ± 114.4% | +5.03% ± 5.5% | +42.94% ± 17.2% | 1.3e-05 |
| case14_ieee tiny | +13.94% ± 7.4% | +73.04% ± 14.9% | +105.25% ± 7.1% | +16.08% ± 8.2% | +26.48% ± 5.8% | 1.9e-05 |
| case30_ieee base | +4.59% ± 2.7% | +15.42% ± 3.0% | +46.81% ± 21.1% | +3.85% ± 2.1% | +31.60% ± 14.9% | 3.9e-05 |
| case30_ieee small | +4.52% ± 0.6% | +17.86% ± 6.9% | +32.67% ± 9.8% | +3.95% ± 0.7% | +13.94% ± 0.9% | 2.8e-05 |
| case30_ieee tiny | +3.06% ± 1.4% | +8.70% ± 7.1% | +14.26% ± 8.3% | +2.78% ± 1.1% | +6.56% ± 1.0% | 3.0e-05 |
| case57_ieee base | +1.39% ± 0.8% | +10.08% ± 1.7% | +28.22% ± 3.9% | +3.91% ± 1.6% | +23.81% ± 4.6% | 2.6e-05 |
| case57_ieee small | +1.57% ± 1.3% | +5.81% ± 1.3% | +13.97% ± 6.0% | +2.30% ± 1.0% | +12.26% ± 3.3% | 1.6e-05 |
| case57_ieee tiny | -0.07% ± 0.6% | +2.19% ± 0.7% | +5.33% ± 0.5% | +2.50% ± 0.1% | +4.30% ± 0.5% | 1.7e-05 |
| case118_ieee base | +0.69% ± 0.6% | +1.47% ± 0.6% | +2.04% ± 0.7% | +0.58% ± 0.6% | +1.84% ± 0.9% | 3.0e-05 |
| case118_ieee small | +1.59% ± 0.7% | +1.51% ± 0.6% | +2.08% ± 1.0% | +1.50% ± 0.7% | +1.58% ± 0.5% | 3.0e-05 |
| case118_ieee tiny | +0.67% ± 0.2% | +1.01% ± 0.6% | +0.96% ± 0.6% | +0.59% ± 0.2% | +1.00% ± 0.4% | 4.0e-05 |

### E5: zero-shot on the other grids (2000 test graphs, mean ± std over seeds)

| source | target | normalizer | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) |
|---|---|---|---|---|---|---|---|
| case14_ieee base | case30_ieee | source | 0.292 ± 0.19 | 4.77 ± 1.2 | 146 ± 44 | 519 ± 5.1e+02 | 59.7 ± 34 |
| case14_ieee base | case30_ieee | refit | 0.354 ± 0.19 | 5.55 ± 0.85 | 196 ± 70 | 603 ± 4.6e+02 | 53.5 ± 30 |
| case14_ieee base | case30_ieee | canon | 0.25 ± 0.19 | 4.32 ± 1.4 | 118 ± 47 | 486 ± 5.3e+02 | 59.5 ± 31 |
| case14_ieee base | case57_ieee | source | 0.312 ± 0.19 | 12.8 ± 0.75 | 584 ± 2e+02 | 1.12e+03 ± 1e+03 | 74.6 ± 54 |
| case14_ieee base | case57_ieee | refit | 0.255 ± 0.16 | 12.5 ± 0.62 | 569 ± 2.1e+02 | 956 ± 9.3e+02 | 59.8 ± 41 |
| case14_ieee base | case57_ieee | canon | 0.216 ± 0.14 | 12.2 ± 0.6 | 537 ± 1.8e+02 | 754 ± 7.6e+02 | 54.5 ± 33 |
| case14_ieee base | case118_ieee | source | 0.421 ± 0.26 | 12.4 ± 0.11 | 999 ± 78 | 3.6e+03 ± 1.7e+03 | 382 ± 67 |
| case14_ieee base | case118_ieee | refit | 0.349 ± 0.22 | 12.2 ± 0.32 | 896 ± 88 | 3.36e+03 ± 1.6e+03 | 327 ± 56 |
| case14_ieee base | case118_ieee | canon | 0.249 ± 0.17 | 11.7 ± 0.61 | 824 ± 1.3e+02 | 2.97e+03 ± 1.6e+03 | 244 ± 61 |
| case14_ieee small | case30_ieee | source | 0.187 ± 0.056 | 5.06 ± 0.13 | 148 ± 17 | 344 ± 1e+02 | 43.3 ± 5.5 |
| case14_ieee small | case30_ieee | refit | 0.24 ± 0.084 | 6.27 ± 0.55 | 170 ± 38 | 347 ± 15 | 40.8 ± 4.7 |
| case14_ieee small | case30_ieee | canon | 0.128 ± 0.04 | 4.13 ± 0.34 | 120 ± 18 | 268 ± 1.3e+02 | 40.5 ± 7.6 |
| case14_ieee small | case57_ieee | source | 0.21 ± 0.095 | 15 ± 0.89 | 528 ± 38 | 476 ± 1.7e+02 | 38.8 ± 8.4 |
| case14_ieee small | case57_ieee | refit | 0.164 ± 0.088 | 14.2 ± 1.1 | 478 ± 61 | 346 ± 2.1e+02 | 33 ± 11 |
| case14_ieee small | case57_ieee | canon | 0.137 ± 0.078 | 13.6 ± 1.2 | 455 ± 61 | 290 ± 1.8e+02 | 36.3 ± 11 |
| case14_ieee small | case118_ieee | source | 0.243 ± 0.065 | 14.4 ± 2.1 | 1.04e+03 ± 2.1e+02 | 2.33e+03 ± 5.2e+02 | 371 ± 56 |
| case14_ieee small | case118_ieee | refit | 0.199 ± 0.057 | 14.1 ± 2.1 | 995 ± 1.8e+02 | 2.22e+03 ± 5.2e+02 | 334 ± 58 |
| case14_ieee small | case118_ieee | canon | 0.147 ± 0.045 | 14 ± 2.1 | 935 ± 1.5e+02 | 1.98e+03 ± 5.1e+02 | 273 ± 57 |
| case14_ieee tiny | case30_ieee | source | 0.137 ± 0.086 | 4.96 ± 1.2 | 119 ± 86 | 268 ± 2e+02 | 46.1 ± 25 |
| case14_ieee tiny | case30_ieee | refit | 0.203 ± 0.078 | 5.58 ± 1.1 | 128 ± 91 | 326 ± 1.7e+02 | 51.4 ± 25 |
| case14_ieee tiny | case30_ieee | canon | 0.0937 ± 0.089 | 4.29 ± 1.4 | 97.8 ± 79 | 222 ± 2.2e+02 | 37 ± 37 |
| case14_ieee tiny | case57_ieee | source | 0.173 ± 0.068 | 12.6 ± 2 | 661 ± 2.2e+02 | 442 ± 2.6e+02 | 60.2 ± 32 |
| case14_ieee tiny | case57_ieee | refit | 0.126 ± 0.056 | 12.2 ± 1.6 | 595 ± 1.9e+02 | 371 ± 2.4e+02 | 55.5 ± 28 |
| case14_ieee tiny | case57_ieee | canon | 0.0993 ± 0.047 | 11.9 ± 1.4 | 507 ± 1.2e+02 | 304 ± 2.2e+02 | 48.9 ± 24 |
| case14_ieee tiny | case118_ieee | source | 0.187 ± 0.11 | 12.5 ± 2 | 885 ± 3.1e+02 | 1.56e+03 ± 1.1e+03 | 330 ± 2e+02 |
| case14_ieee tiny | case118_ieee | refit | 0.145 ± 0.093 | 11.5 ± 1.3 | 804 ± 2.5e+02 | 1.45e+03 ± 1e+03 | 271 ± 1.7e+02 |
| case14_ieee tiny | case118_ieee | canon | 0.097 ± 0.064 | 10.3 ± 0.68 | 701 ± 1.6e+02 | 1.26e+03 ± 9.1e+02 | 196 ± 1.2e+02 |
| case30_ieee base | case14_ieee | source | 0.0175 ± 0.0085 | 3.75 ± 0.56 | 35.7 ± 9.6 | 21.9 ± 11 | 4.95 ± 2.6 |
| case30_ieee base | case14_ieee | refit | 0.0107 ± 0.0063 | 3.03 ± 0.24 | 30.4 ± 6.2 | 19.7 ± 9.6 | 2.94 ± 1.4 |
| case30_ieee base | case14_ieee | canon | 0.0387 ± 0.024 | 4.86 ± 0.71 | 45.4 ± 13 | 44.5 ± 24 | 11.8 ± 6.8 |
| case30_ieee base | case57_ieee | source | 0.0924 ± 0.06 | 10.9 ± 2.1 | 322 ± 78 | 285 ± 2.6e+02 | 60.7 ± 49 |
| case30_ieee base | case57_ieee | refit | 0.0493 ± 0.021 | 10.4 ± 1.7 | 298 ± 71 | 136 ± 98 | 33.3 ± 21 |
| case30_ieee base | case57_ieee | canon | 0.072 ± 0.044 | 10.8 ± 2 | 310 ± 74 | 202 ± 1.8e+02 | 49.6 ± 38 |
| case30_ieee base | case118_ieee | source | 0.145 ± 0.1 | 11.1 ± 0.0053 | 490 ± 1.4e+02 | 1.31e+03 ± 9.2e+02 | 270 ± 1.7e+02 |
| case30_ieee base | case118_ieee | refit | 0.0922 ± 0.065 | 10.7 ± 0.73 | 482 ± 1.5e+02 | 1.15e+03 ± 8.2e+02 | 192 ± 1.2e+02 |
| case30_ieee base | case118_ieee | canon | 0.0928 ± 0.065 | 10.8 ± 0.72 | 482 ± 1.5e+02 | 1.16e+03 ± 8.3e+02 | 193 ± 1.2e+02 |
| case30_ieee small | case14_ieee | source | 0.0459 ± 0.013 | 5.6 ± 0.48 | 57.1 ± 1.3 | 64.8 ± 21 | 12 ± 1.2 |
| case30_ieee small | case14_ieee | refit | 0.0298 ± 0.011 | 4.9 ± 0.99 | 56.3 ± 11 | 34.4 ± 6.2 | 6.48 ± 1.7 |
| case30_ieee small | case14_ieee | canon | 0.0699 ± 0.021 | 5.92 ± 0.33 | 57.2 ± 4.5 | 92.1 ± 28 | 19.8 ± 4.2 |
| case30_ieee small | case57_ieee | source | 0.181 ± 0.046 | 11.2 ± 0.39 | 393 ± 74 | 206 ± 43 | 34.6 ± 6.1 |
| case30_ieee small | case57_ieee | refit | 0.108 ± 0.032 | 11.1 ± 0.25 | 333 ± 9 | 136 ± 28 | 50.3 ± 18 |
| case30_ieee small | case57_ieee | canon | 0.158 ± 0.041 | 11.1 ± 0.26 | 350 ± 22 | 180 ± 37 | 33.8 ± 9.4 |
| case30_ieee small | case118_ieee | source | 0.12 ± 0.024 | 10.6 ± 0.48 | 667 ± 71 | 1.07e+03 ± 4.8e+02 | 232 ± 47 |
| case30_ieee small | case118_ieee | refit | 0.0723 ± 0.01 | 10.2 ± 0.36 | 658 ± 15 | 876 ± 3.6e+02 | 154 ± 25 |
| case30_ieee small | case118_ieee | canon | 0.073 ± 0.01 | 10.2 ± 0.34 | 658 ± 15 | 879 ± 3.7e+02 | 156 ± 26 |
| case30_ieee tiny | case14_ieee | source | 0.0106 ± 0.0054 | 3.08 ± 0.98 | 25.9 ± 6.3 | 18.5 ± 9.4 | 3.22 ± 1.4 |
| case30_ieee tiny | case14_ieee | refit | 0.00801 ± 0.0034 | 2.67 ± 0.82 | 26.1 ± 7.4 | 15.2 ± 6.4 | 2.25 ± 0.61 |
| case30_ieee tiny | case14_ieee | canon | 0.018 ± 0.015 | 3.66 ± 1 | 31.3 ± 9.1 | 23.9 ± 9.9 | 6.37 ± 4.5 |
| case30_ieee tiny | case57_ieee | source | 0.0711 ± 0.063 | 10.8 ± 0.016 | 345 ± 76 | 175 ± 1.6e+02 | 33.3 ± 29 |
| case30_ieee tiny | case57_ieee | refit | 0.0479 ± 0.03 | 10.5 ± 0.4 | 303 ± 6.6 | 91.8 ± 46 | 19.1 ± 15 |
| case30_ieee tiny | case57_ieee | canon | 0.0625 ± 0.052 | 10.7 ± 0.21 | 307 ± 28 | 117 ± 86 | 27.2 ± 24 |
| case30_ieee tiny | case118_ieee | source | 0.131 ± 0.12 | 11 ± 0.44 | 557 ± 1.7e+02 | 1.25e+03 ± 9.9e+02 | 231 ± 1.6e+02 |
| case30_ieee tiny | case118_ieee | refit | 0.0846 ± 0.072 | 9.87 ± 0.21 | 486 ± 67 | 1.11e+03 ± 9e+02 | 163 ± 1.1e+02 |
| case30_ieee tiny | case118_ieee | canon | 0.085 ± 0.072 | 9.87 ± 0.21 | 486 ± 67 | 1.11e+03 ± 9e+02 | 163 ± 1.1e+02 |
| case57_ieee base | case14_ieee | source | 0.115 ± 0.073 | 5.38 ± 0.72 | 43 ± 6.8 | 108 ± 38 | 22.9 ± 11 |
| case57_ieee base | case14_ieee | refit | 0.161 ± 0.039 | 6.13 ± 1.4 | 49 ± 7 | 154 ± 38 | 41.6 ± 15 |
| case57_ieee base | case14_ieee | canon | 0.223 ± 0.027 | 8.1 ± 0.54 | 72 ± 5.4 | 241 ± 15 | 59.5 ± 8.4 |
| case57_ieee base | case30_ieee | source | 0.237 ± 0.098 | 5.95 ± 1.5 | 53.1 ± 19 | 220 ± 84 | 64.6 ± 17 |
| case57_ieee base | case30_ieee | refit | 0.365 ± 0.061 | 9.86 ± 0.32 | 133 ± 31 | 378 ± 45 | 43.1 ± 4.3 |
| case57_ieee base | case30_ieee | canon | 0.292 ± 0.075 | 7.44 ± 0.69 | 71.9 ± 21 | 300 ± 35 | 82.1 ± 30 |
| case57_ieee base | case118_ieee | source | 0.286 ± 0.035 | 18 ± 2 | 913 ± 1.9e+02 | 2.53e+03 ± 4.1e+02 | 434 ± 23 |
| case57_ieee base | case118_ieee | refit | 0.277 ± 0.033 | 17.4 ± 2 | 878 ± 1.8e+02 | 2.48e+03 ± 3.9e+02 | 425 ± 16 |
| case57_ieee base | case118_ieee | canon | 0.246 ± 0.031 | 15.9 ± 2 | 851 ± 1.4e+02 | 2.36e+03 ± 2.7e+02 | 405 ± 26 |
| case57_ieee small | case14_ieee | source | 0.0727 ± 0.03 | 4.21 ± 0.46 | 42.9 ± 9.1 | 66.3 ± 2.3 | 12.2 ± 4 |
| case57_ieee small | case14_ieee | refit | 0.0716 ± 0.028 | 4.47 ± 0.6 | 44.8 ± 6.5 | 77.4 ± 18 | 16.5 ± 4.6 |
| case57_ieee small | case14_ieee | canon | 0.118 ± 0.03 | 5.77 ± 0.096 | 63.5 ± 4.7 | 133 ± 26 | 32.7 ± 3.1 |
| case57_ieee small | case30_ieee | source | 0.255 ± 0.081 | 8.01 ± 2.5 | 89.2 ± 46 | 252 ± 90 | 51.5 ± 14 |
| case57_ieee small | case30_ieee | refit | 0.319 ± 0.066 | 10.2 ± 1.4 | 131 ± 20 | 348 ± 49 | 38.8 ± 4.9 |
| case57_ieee small | case30_ieee | canon | 0.265 ± 0.078 | 8.02 ± 2.7 | 93.7 ± 50 | 258 ± 79 | 52.1 ± 15 |
| case57_ieee small | case118_ieee | source | 0.197 ± 0.007 | 19.1 ± 3.1 | 1.07e+03 ± 2.6e+02 | 1.87e+03 ± 1.7e+02 | 367 ± 21 |
| case57_ieee small | case118_ieee | refit | 0.186 ± 0.0072 | 18.4 ± 3.3 | 1.01e+03 ± 2.7e+02 | 1.82e+03 ± 1.6e+02 | 356 ± 21 |
| case57_ieee small | case118_ieee | canon | 0.167 ± 0.012 | 17.3 ± 3.6 | 922 ± 2.7e+02 | 1.69e+03 ± 1.3e+02 | 335 ± 13 |
| case57_ieee tiny | case14_ieee | source | 0.0285 ± 0.019 | 4.1 ± 1.1 | 37.9 ± 14 | 56.8 ± 27 | 3.84 ± 2.4 |
| case57_ieee tiny | case14_ieee | refit | 0.0408 ± 0.038 | 3.91 ± 0.72 | 34.6 ± 10 | 55 ± 23 | 6.1 ± 4.2 |
| case57_ieee tiny | case14_ieee | canon | 0.045 ± 0.035 | 4.64 ± 0.68 | 43.1 ± 10 | 68.8 ± 28 | 10.6 ± 6.3 |
| case57_ieee tiny | case30_ieee | source | 0.0879 ± 0.078 | 5.62 ± 1.1 | 63.4 ± 17 | 76.7 ± 43 | 17.5 ± 15 |
| case57_ieee tiny | case30_ieee | refit | 0.232 ± 0.087 | 14.1 ± 5.1 | 208 ± 1.2e+02 | 464 ± 1.8e+02 | 71.5 ± 25 |
| case57_ieee tiny | case30_ieee | canon | 0.103 ± 0.06 | 6.6 ± 1 | 73.8 ± 22 | 175 ± 1.5e+02 | 47.2 ± 31 |
| case57_ieee tiny | case118_ieee | source | 0.268 ± 0.084 | 22.2 ± 6.4 | 849 ± 6.3e+02 | 2.65e+03 ± 9.6e+02 | 528 ± 2.1e+02 |
| case57_ieee tiny | case118_ieee | refit | 0.241 ± 0.072 | 21.9 ± 6.4 | 813 ± 5.7e+02 | 2.6e+03 ± 9.4e+02 | 499 ± 1.9e+02 |
| case57_ieee tiny | case118_ieee | canon | 0.198 ± 0.059 | 20.5 ± 6.2 | 778 ± 4.6e+02 | 2.44e+03 ± 8.9e+02 | 433 ± 1.7e+02 |
| case118_ieee base | case14_ieee | source | 0.0268 ± 0.014 | 3.91 ± 0.52 | 40.6 ± 13 | 48.2 ± 23 | 4.13 ± 0.66 |
| case118_ieee base | case14_ieee | refit | 0.0206 ± 0.011 | 6.39 ± 3.8 | 126 ± 1.2e+02 | 53.1 ± 16 | 7.2 ± 5.1 |
| case118_ieee base | case14_ieee | canon | 0.0231 ± 0.015 | 10.7 ± 9.4 | 208 ± 2.3e+02 | 53.2 ± 12 | 14.7 ± 16 |
| case118_ieee base | case30_ieee | source | 0.063 ± 0.031 | 12.8 ± 8 | 160 ± 1.3e+02 | 52.7 ± 21 | 29.5 ± 27 |
| case118_ieee base | case30_ieee | refit | 0.0629 ± 0.036 | 15.9 ± 7.7 | 270 ± 1.7e+02 | 74 ± 26 | 48.8 ± 43 |
| case118_ieee base | case30_ieee | canon | 0.0625 ± 0.035 | 15.9 ± 7.8 | 269 ± 1.7e+02 | 73.9 ± 23 | 49 ± 43 |
| case118_ieee base | case57_ieee | source | 0.0945 ± 0.055 | 17.3 ± 7.9 | 319 ± 1.4e+02 | 108 ± 2.1 | 73.8 ± 33 |
| case118_ieee base | case57_ieee | refit | 0.0919 ± 0.055 | 16.3 ± 8.2 | 337 ± 1.6e+02 | 103 ± 14 | 70.8 ± 33 |
| case118_ieee base | case57_ieee | canon | 0.0869 ± 0.051 | 16.1 ± 6.4 | 579 ± 1e+02 | 158 ± 62 | 76.8 ± 16 |
| case118_ieee small | case14_ieee | source | 0.0296 ± 0.013 | 3.47 ± 0.91 | 38.3 ± 9.2 | 22 ± 6.9 | 3.94 ± 0.8 |
| case118_ieee small | case14_ieee | refit | 0.0152 ± 0.0067 | 4.92 ± 2.3 | 53 ± 25 | 30.1 ± 12 | 3.37 ± 1.5 |
| case118_ieee small | case14_ieee | canon | 0.0205 ± 0.0086 | 8.52 ± 3.7 | 88.7 ± 40 | 59.9 ± 25 | 6.83 ± 3.3 |
| case118_ieee small | case30_ieee | source | 0.196 ± 0.016 | 16.7 ± 10 | 299 ± 2.7e+02 | 135 ± 56 | 88.6 ± 81 |
| case118_ieee small | case30_ieee | refit | 0.136 ± 0.054 | 11.6 ± 4.8 | 181 ± 70 | 116 ± 56 | 85.9 ± 65 |
| case118_ieee small | case30_ieee | canon | 0.136 ± 0.055 | 11.6 ± 4.9 | 181 ± 72 | 115 ± 55 | 85.9 ± 66 |
| case118_ieee small | case57_ieee | source | 0.105 ± 0.033 | 10.6 ± 6.7 | 209 ± 1.6e+02 | 129 ± 60 | 69.4 ± 58 |
| case118_ieee small | case57_ieee | refit | 0.104 ± 0.032 | 10.6 ± 6.9 | 206 ± 1.8e+02 | 102 ± 43 | 65.6 ± 53 |
| case118_ieee small | case57_ieee | canon | 0.108 ± 0.034 | 12.2 ± 4.5 | 239 ± 57 | 82 ± 23 | 66.2 ± 44 |
| case118_ieee tiny | case14_ieee | source | 0.019 ± 0.013 | 2.48 ± 0.51 | 22.6 ± 3.9 | 61.3 ± 42 | 2.81 ± 0.93 |
| case118_ieee tiny | case14_ieee | refit | 0.0158 ± 0.0039 | 3.77 ± 0.87 | 35.8 ± 11 | 57 ± 14 | 2.56 ± 0.52 |
| case118_ieee tiny | case14_ieee | canon | 0.0133 ± 0.0018 | 5.46 ± 2.5 | 56.8 ± 29 | 45.1 ± 13 | 4.26 ± 2.1 |
| case118_ieee tiny | case30_ieee | source | 0.0761 ± 0.042 | 11.5 ± 2.5 | 121 ± 40 | 48.1 ± 20 | 27.9 ± 13 |
| case118_ieee tiny | case30_ieee | refit | 0.039 ± 0.0044 | 10.7 ± 5 | 116 ± 58 | 40.7 ± 25 | 13.1 ± 0.17 |
| case118_ieee tiny | case30_ieee | canon | 0.039 ± 0.0037 | 10.4 ± 4.8 | 113 ± 56 | 40.8 ± 25 | 13 ± 0.64 |
| case118_ieee tiny | case57_ieee | source | 0.0488 ± 0.0074 | 9.37 ± 3.1 | 204 ± 1e+02 | 73.9 ± 25 | 32.3 ± 4.4 |
| case118_ieee tiny | case57_ieee | refit | 0.0467 ± 0.0082 | 9.18 ± 3.6 | 209 ± 1e+02 | 76.3 ± 26 | 24 ± 8.4 |
| case118_ieee tiny | case57_ieee | canon | 0.0529 ± 0.006 | 12.2 ± 6 | 309 ± 1.9e+02 | 96.8 ± 11 | 42.5 ± 8.6 |

E5a: 540 (pair, channel) checks. Probe at k = refit/source base vs the refit run, worst relative difference 1.6e-04; |ΔRMSE| > EE_S3(k): 0 violation(s); median |ΔRMSE| / EE = 0.41.

E5b: canon under source vs refit normalizer, worst relative difference 7.5e-05 over 108 pairs.

E5c (H2, exploratory): Spearman ρ(EE_S3,VM(k=10) on the target, zero-shot VM RMSE) over 108 pairs = 0.90; residualized on (source, target, size) = 0.84.

E5d (amendment 22:00, 81 held-out pairs, sources case30/57/118), ρ_partial with EE_S3,VM(k=10): raw error 0.76 (E5d-1: > 0.5), raw − canon 0.50 (E5d-2: > 0.5), canon error 0.69 (E5d-3: below raw).
