# R2: M0+Canon vs M1 on held-out grids

Runs: heldout_case118_ieee canon 3, heldout_case118_ieee canonresnorm 3, heldout_case118_ieee dccanon 3, heldout_case118_ieee dcinjcanon 3, heldout_case118_ieee dczcanon 3, heldout_case118_ieee m1fullcanon 3, heldout_case118_ieee m3canon 3, heldout_case30_ieee canon 3, heldout_case30_ieee canonvref 1, heldout_case30_ieee dccanon 3, heldout_case30_ieee dcinj1hop 3, heldout_case30_ieee dcinjcanon 3, heldout_case30_ieee dczcanon 3, heldout_case30_ieee m1fullcanon 3, heldout_case30_ieee m3canon 3, heldout_case57_ieee canon 3, heldout_case57_ieee dccanon 3, heldout_case57_ieee dcinj1hop 3, heldout_case57_ieee dcinjcanon 3, heldout_case57_ieee dczcanon 3, heldout_case57_ieee m1fullcanon 3, heldout_case57_ieee m3canon 3

### heldout_case118_ieee: zero-shot

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) | VM, PQ buses (p.u.) |
|---|---|---|---|---|---|---|---|---|---|---|
| case118_ieee | canon | 0.00632 ± 0.00064 | 11.5 ± 1 | 435 ± 2.1e+02 | 42.6 ± 22 | 16.7 ± 2.7 | 7.13 ± 1.6 | 8.89 ± 0.47 | 2.02 ± 0.19 | nan ± nan |
| case118_ieee | m1fullcanon | 0.00731 ± 0.0011 | 11.2 ± 1.1 | 317 ± 1.7e+02 | 78.2 ± 53 | 21.2 ± 3.7 | 6.85 ± 1.5 | 8.84 ± 0.22 | 2.07 ± 0.11 | nan ± nan |
| case118_ieee | m3canon | 0.00312 ± 0.00086 | 10.6 ± 1.7 | 350 ± 1.9e+02 | 18.8 ± 2 | 11.8 ± 1.6 | 6.74 ± 2.4 | 7.96 ± 0.26 | 1.87 ± 0.11 | nan ± nan |
| case118_ieee | canonresnorm | 0.00466 ± 0.0014 | 10.7 ± 1.5 | 457 ± 1.7e+02 | 27.8 ± 5.9 | 9.92 ± 0.92 | 7.67 ± 1.8 | 7.45 ± 0.33 | 1.8 ± 0.22 | nan ± nan |
| case118_ieee | dccanon | 0.00699 ± 0.001 | 1.95 ± 0.042 | 39 ± 27 | 34.8 ± 12 | 7.54 ± 0.078 | 1.52 ± 0.065 | 1.22 ± 0.11 | 0.426 ± 0.041 | nan ± nan |
| case118_ieee | dczcanon | 0.00531 ± 0.00047 | 1.99 ± 0.28 | 32.4 ± 11 | 18 ± 3.5 | 6.39 ± 1.9 | 1.65 ± 0.27 | 1.1 ± 0.097 | 0.401 ± 0.06 | nan ± nan |
| case118_ieee | dcinjcanon | 0.00566 ± 0.00032 | 2.77 ± 1.8 | 131 ± 1e+02 | 28.2 ± 2 | 7.41 ± 2.7 | 2.52 ± 1.6 | 1.16 ± 0.85 | 0.416 ± 0.22 | nan ± nan |

### heldout_case118_ieee: in distribution

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) | VM, PQ buses (p.u.) |
|---|---|---|---|---|---|---|---|---|---|---|
| case14_ieee | canon | 0.00345 ± 0.0012 | 0.333 ± 0.049 | 2.17 ± 0.43 | 13.7 ± 4.9 | 0.15 ± 0.021 | 0.328 ± 0.05 | 0.0535 ± 0.0096 | 0.0939 ± 0.019 | nan ± nan |
| case14_ieee | m1fullcanon | 0.00082 ± 0.00034 | 0.198 ± 0.087 | 2.44 ± 0.87 | 2.31 ± 1 | 0.161 ± 0.032 | 0.188 ± 0.083 | 0.0622 ± 0.025 | 0.0535 ± 0.022 | nan ± nan |
| case14_ieee | m3canon | 0.00148 ± 0.0011 | 0.145 ± 0.091 | 1.26 ± 0.5 | 5.89 ± 4.6 | 0.0906 ± 0.015 | 0.143 ± 0.092 | 0.0218 ± 0.004 | 0.0422 ± 0.027 | nan ± nan |
| case14_ieee | canonresnorm | 0.00238 ± 0.0017 | 0.238 ± 0.17 | 1.72 ± 0.92 | 9.57 ± 7 | 0.105 ± 0.038 | 0.235 ± 0.17 | 0.0329 ± 0.017 | 0.0663 ± 0.047 | nan ± nan |
| case14_ieee | dccanon | 0.00241 ± 0.00082 | 0.19 ± 0.06 | 0.666 ± 0.13 | 9.54 ± 3.4 | 0.0731 ± 0.0056 | 0.189 ± 0.06 | 0.0178 ± 0.0035 | 0.0567 ± 0.018 | nan ± nan |
| case14_ieee | dczcanon | 0.00249 ± 0.002 | 0.18 ± 0.14 | 0.469 ± 0.19 | 9.75 ± 7.8 | 0.0848 ± 0.036 | 0.179 ± 0.14 | 0.0167 ± 0.0097 | 0.0559 ± 0.043 | nan ± nan |
| case14_ieee | dcinjcanon | 0.00292 ± 0.0012 | 0.202 ± 0.062 | 0.792 ± 0.35 | 11.6 ± 4.6 | 0.13 ± 0.04 | 0.2 ± 0.061 | 0.0307 ± 0.017 | 0.067 ± 0.024 | nan ± nan |
| case30_ieee | canon | 0.0058 ± 0.00033 | 0.649 ± 0.009 | 4.06 ± 0.67 | 22.2 ± 2 | 0.183 ± 0.0086 | 0.641 ± 0.0073 | 0.0979 ± 0.013 | 0.112 ± 0.0014 | nan ± nan |
| case30_ieee | m1fullcanon | 0.00384 ± 9.3e-05 | 0.244 ± 0.046 | 2.69 ± 0.66 | 4.09 ± 0.21 | 0.182 ± 0.018 | 0.233 ± 0.044 | 0.0749 ± 0.013 | 0.0491 ± 0.0077 | nan ± nan |
| case30_ieee | m3canon | 0.00444 ± 0.00051 | 0.554 ± 0.091 | 3.76 ± 0.93 | 19.5 ± 2.8 | 0.151 ± 0.016 | 0.549 ± 0.09 | 0.0752 ± 0.012 | 0.096 ± 0.015 | nan ± nan |
| case30_ieee | canonresnorm | 0.00412 ± 0.00077 | 0.5 ± 0.12 | 3.62 ± 0.83 | 17.8 ± 4.6 | 0.146 ± 0.024 | 0.494 ± 0.12 | 0.0716 ± 0.012 | 0.0881 ± 0.02 | nan ± nan |
| case30_ieee | dccanon | 0.00471 ± 0.00042 | 0.139 ± 0.025 | 0.504 ± 0.11 | 6.52 ± 0.96 | 0.176 ± 0.015 | 0.136 ± 0.026 | 0.0277 ± 0.0018 | 0.0321 ± 0.003 | nan ± nan |
| case30_ieee | dczcanon | 0.00508 ± 0.00086 | 0.236 ± 0.15 | 0.526 ± 0.2 | 12.1 ± 8 | 0.179 ± 0.036 | 0.23 ± 0.16 | 0.0316 ± 0.003 | 0.0524 ± 0.026 | nan ± nan |
| case30_ieee | dcinjcanon | 0.00456 ± 0.00092 | 0.193 ± 0.043 | 2.08 ± 0.78 | 8.22 ± 4.3 | 0.185 ± 0.036 | 0.186 ± 0.038 | 0.0501 ± 0.022 | 0.0449 ± 0.016 | nan ± nan |
| case57_ieee | canon | 0.00413 ± 0.00032 | 0.249 ± 0.03 | 4.38 ± 0.92 | 16.5 ± 0.82 | 0.207 ± 0.015 | 0.203 ± 0.029 | 0.145 ± 0.013 | 0.0746 ± 0.0066 | nan ± nan |
| case57_ieee | m1fullcanon | 0.00397 ± 0.00031 | 0.216 ± 0.021 | 3.89 ± 0.43 | 12.4 ± 1.1 | 0.186 ± 0.016 | 0.156 ± 0.03 | 0.147 ± 0.0039 | 0.0747 ± 0.00056 | nan ± nan |
| case57_ieee | m3canon | 0.00302 ± 0.00015 | 0.207 ± 0.035 | 4.22 ± 0.65 | 15.3 ± 0.93 | 0.134 ± 0.0055 | 0.187 ± 0.032 | 0.0889 ± 0.014 | 0.0478 ± 0.0064 | nan ± nan |
| case57_ieee | canonresnorm | 0.00334 ± 0.0003 | 0.228 ± 0.03 | 4.31 ± 0.48 | 17.2 ± 3.7 | 0.15 ± 0.012 | 0.194 ± 0.029 | 0.12 ± 0.012 | 0.0618 ± 0.0068 | nan ± nan |
| case57_ieee | dccanon | 0.00422 ± 0.00045 | 0.118 ± 0.012 | 3.04 ± 1.1 | 15 ± 4 | 0.162 ± 0.026 | 0.105 ± 0.012 | 0.0526 ± 0.0059 | 0.0351 ± 0.0051 | nan ± nan |
| case57_ieee | dczcanon | 0.00459 ± 0.00026 | 0.183 ± 0.078 | 2.64 ± 0.47 | 24.3 ± 16 | 0.149 ± 0.023 | 0.174 ± 0.08 | 0.0551 ± 0.009 | 0.047 ± 0.018 | nan ± nan |
| case57_ieee | dcinjcanon | 0.00389 ± 0.00034 | 0.0665 ± 0.0092 | 1.92 ± 0.58 | 7.84 ± 1.8 | 0.16 ± 0.025 | 0.0548 ± 0.009 | 0.0373 ± 0.0065 | 0.0313 ± 0.0025 | nan ± nan |

### heldout_case30_ieee: zero-shot

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) | VM, PQ buses (p.u.) |
|---|---|---|---|---|---|---|---|---|---|---|
| case30_ieee | canon | 0.0109 ± 0.0019 | 5.34 ± 1 | 31.3 ± 13 | 9.71 ± 3.8 | 5.56 ± 0.19 | 4.67 ± 1.1 | 2.55 ± 0.2 | 1.36 ± 0.14 | nan ± nan |
| case30_ieee | m1fullcanon | 0.00817 ± 0.0027 | 3.45 ± 2.7 | 45.8 ± 39 | 8.04 ± 1.8 | 2.8 ± 0.63 | 3.21 ± 2.6 | 1.16 ± 0.58 | 0.727 ± 0.41 | nan ± nan |
| case30_ieee | m3canon | 0.0114 ± 0.0016 | 4.99 ± 1.8 | 54.5 ± 43 | 16.9 ± 8.5 | 4.09 ± 0.69 | 4.65 ± 1.9 | 1.72 ± 0.21 | 1.1 ± 0.21 | nan ± nan |
| case30_ieee | dccanon | 0.011 ± 0.0017 | 2.14 ± 0.16 | 35.3 ± 12 | 18 ± 12 | 1.87 ± 0.33 | 2.1 ± 0.16 | 0.394 ± 0.044 | 0.387 ± 0.034 | nan ± nan |
| case30_ieee | dczcanon | 0.0119 ± 0.0031 | 1.81 ± 0.2 | 18.7 ± 8.8 | 41 ± 24 | 1.84 ± 0.38 | 1.77 ± 0.2 | 0.351 ± 0.13 | 0.345 ± 0.011 | nan ± nan |
| case30_ieee | dcinjcanon | 0.00629 ± 0.00038 | 0.692 ± 0.54 | 8.23 ± 5.8 | 9.96 ± 2 | 1.71 ± 0.54 | 0.609 ± 0.52 | 0.312 ± 0.16 | 0.18 ± 0.095 | nan ± nan |
| case30_ieee | canonvref | 0.0134 | 7.69 | 97.9 | 11.4 | 3.89 | 7.32 | 2.36 | 1.44 | 0.0134 |
| case30_ieee | dcinj1hop | 0.0152 ± 0.0014 | 1.72 ± 0.64 | 17.2 ± 5.6 | 21.5 ± 8.2 | 19.5 ± 3.6 | 1.59 ± 0.63 | 0.629 ± 0.16 | 0.393 ± 0.099 | 0.0154 ± 0.0014 |

### heldout_case30_ieee: in distribution

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) | VM, PQ buses (p.u.) |
|---|---|---|---|---|---|---|---|---|---|---|
| case14_ieee | canon | 0.000689 ± 8.7e-05 | 0.158 ± 0.096 | 1.91 ± 1.2 | 2.33 ± 0.41 | 0.18 ± 0.054 | 0.151 ± 0.092 | 0.046 ± 0.026 | 0.043 ± 0.024 | nan ± nan |
| case14_ieee | m1fullcanon | 0.000808 ± 0.0002 | 0.167 ± 0.034 | 1.96 ± 0.37 | 2.77 ± 0.55 | 0.145 ± 0.032 | 0.16 ± 0.032 | 0.0483 ± 0.011 | 0.0446 ± 0.0091 | nan ± nan |
| case14_ieee | m3canon | 0.000511 ± 0.00017 | 0.0765 ± 0.0048 | 0.89 ± 0.071 | 1.81 ± 0.64 | 0.0869 ± 0.0086 | 0.0733 ± 0.0045 | 0.0219 ± 0.002 | 0.0214 ± 0.002 | nan ± nan |
| case14_ieee | dccanon | 0.00193 ± 0.00079 | 0.138 ± 0.058 | 0.453 ± 0.16 | 7.57 ± 3.1 | 0.0867 ± 0.015 | 0.138 ± 0.058 | 0.0144 ± 0.0036 | 0.0431 ± 0.018 | nan ± nan |
| case14_ieee | dczcanon | 0.00754 ± 0.0042 | 0.54 ± 0.31 | 1.13 ± 0.7 | 29.9 ± 17 | 0.0766 ± 0.014 | 0.54 ± 0.31 | 0.0237 ± 0.011 | 0.169 ± 0.097 | nan ± nan |
| case14_ieee | dcinjcanon | 0.00252 ± 0.00071 | 0.181 ± 0.041 | 0.909 ± 0.41 | 10 ± 2.8 | 0.15 ± 0.043 | 0.179 ± 0.041 | 0.0231 ± 0.013 | 0.0588 ± 0.015 | nan ± nan |
| case14_ieee | canonvref | 0.000349 | 0.123 | 1.95 | 0.354 | 0.189 | 0.117 | 0.0392 | 0.0351 | 0.000349 |
| case14_ieee | dcinj1hop | 0.00449 ± 0.00097 | 0.364 ± 0.14 | 5.29 ± 2.1 | 12.3 ± 2.4 | 1.84 ± 0.38 | 0.33 ± 0.18 | 0.113 ± 0.023 | 0.16 ± 0.01 | 0.00362 ± 0.00082 |
| case57_ieee | canon | 0.0042 ± 0.00033 | 0.289 ± 0.046 | 5.27 ± 1.3 | 17 ± 0.48 | 0.259 ± 0.029 | 0.242 ± 0.05 | 0.156 ± 0.0072 | 0.0817 ± 0.0034 | nan ± nan |
| case57_ieee | m1fullcanon | 0.00407 ± 0.00037 | 0.217 ± 0.03 | 3.74 ± 0.21 | 13.5 ± 0.83 | 0.2 ± 0.013 | 0.154 ± 0.033 | 0.152 ± 0.011 | 0.0753 ± 0.0053 | nan ± nan |
| case57_ieee | m3canon | 0.00303 ± 0.00036 | 0.2 ± 0.024 | 3.84 ± 0.47 | 14.7 ± 1.4 | 0.145 ± 0.0089 | 0.182 ± 0.024 | 0.0812 ± 0.0054 | 0.0452 ± 0.0033 | nan ± nan |
| case57_ieee | dccanon | 0.00403 ± 0.00037 | 0.0932 ± 0.011 | 2.73 ± 0.097 | 7.85 ± 2.3 | 0.155 ± 0.018 | 0.0758 ± 0.0084 | 0.0541 ± 0.0075 | 0.0304 ± 0.004 | nan ± nan |
| case57_ieee | dczcanon | 0.00438 ± 0.00016 | 0.158 ± 0.053 | 3.31 ± 0.19 | 20.9 ± 11 | 0.158 ± 0.019 | 0.116 ± 0.056 | 0.1 ± 0.032 | 0.0556 ± 0.019 | nan ± nan |
| case57_ieee | dcinjcanon | 0.00392 ± 0.00029 | 0.114 ± 0.048 | 2.3 ± 1.2 | 8.28 ± 0.55 | 0.196 ± 0.033 | 0.0988 ± 0.04 | 0.0566 ± 0.026 | 0.0366 ± 0.0068 | nan ± nan |
| case57_ieee | canonvref | 0.00413 | 0.234 | 4.51 | 1.07 | 0.288 | 0.177 | 0.154 | 0.0797 | 0.00413 |
| case57_ieee | dcinj1hop | 0.017 ± 0.0016 | 0.338 ± 0.071 | 8.12 ± 1.4 | 25.8 ± 13 | 2.99 ± 0.31 | 0.163 ± 0.03 | 0.296 ± 0.066 | 0.251 ± 0.056 | 0.0171 ± 0.0015 |
| case118_ieee | canon | 0.00156 ± 0.00027 | 1.65 ± 0.31 | 72.8 ± 13 | 9.26 ± 2 | 1.2 ± 0.18 | 1.4 ± 0.3 | 0.878 ± 0.12 | 0.251 ± 0.038 | nan ± nan |
| case118_ieee | m1fullcanon | 0.00192 ± 0.00021 | 0.771 ± 0.08 | 27.3 ± 1.4 | 10.9 ± 0.04 | 0.607 ± 0.092 | 0.55 ± 0.06 | 0.539 ± 0.061 | 0.128 ± 0.012 | nan ± nan |
| case118_ieee | m3canon | 0.00111 ± 5.9e-05 | 1.31 ± 0.057 | 57.5 ± 5.7 | 7.11 ± 0.3 | 0.885 ± 0.049 | 1.09 ± 0.079 | 0.719 ± 0.018 | 0.2 ± 0.0068 | nan ± nan |
| case118_ieee | dccanon | 0.00249 ± 0.00016 | 0.625 ± 0.019 | 19 ± 0.88 | 15.1 ± 0.98 | 0.295 ± 0.051 | 0.598 ± 0.018 | 0.18 ± 0.0082 | 0.0928 ± 0.0036 | nan ± nan |
| case118_ieee | dczcanon | 0.0028 ± 0.00037 | 0.64 ± 0.035 | 18.9 ± 1.8 | 17 ± 1.7 | 0.287 ± 0.067 | 0.603 ± 0.029 | 0.212 ± 0.035 | 0.0992 ± 0.0089 | nan ± nan |
| case118_ieee | dcinjcanon | 0.00151 ± 0.00038 | 0.195 ± 0.078 | 7.57 ± 5.4 | 8.17 ± 1.5 | 0.395 ± 0.12 | 0.17 ± 0.064 | 0.0944 ± 0.047 | 0.0399 ± 0.013 | nan ± nan |
| case118_ieee | canonvref | 0.000592 | 2.3 | 91.7 | 1.99 | 1.25 | 1.96 | 1.2 | 0.322 | 0.000592 |
| case118_ieee | dcinj1hop | 0.00756 ± 0.0011 | 0.45 ± 0.018 | 23.6 ± 3.7 | 19.5 ± 8.1 | 6.24 ± 1.3 | 0.327 ± 0.041 | 0.307 ± 0.018 | 0.14 ± 0.0057 | 0.0075 ± 0.0011 |

### heldout_case57_ieee: zero-shot

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) | VM, PQ buses (p.u.) |
|---|---|---|---|---|---|---|---|---|---|---|
| case57_ieee | canon | 0.018 ± 0.0027 | 8.26 ± 0.63 | 285 ± 14 | 66.2 ± 22 | 10 ± 1.9 | 7.35 ± 0.88 | 3.72 ± 0.31 | 2.06 ± 0.067 | nan ± nan |
| case57_ieee | m1fullcanon | 0.0177 ± 0.0018 | 5.55 ± 2.1 | 186 ± 73 | 70.1 ± 21 | 8.74 ± 1.5 | 4.84 ± 2.2 | 2.59 ± 0.4 | 1.35 ± 0.32 | nan ± nan |
| case57_ieee | m3canon | 0.0217 ± 0.0064 | 8.22 ± 1.8 | 248 ± 86 | 27.4 ± 14 | 6.35 ± 1.1 | 7.91 ± 1.8 | 2.22 ± 0.23 | 1.48 ± 0.3 | nan ± nan |
| case57_ieee | dccanon | 0.0203 ± 0.0032 | 1.62 ± 0.15 | 27.6 ± 9.5 | 89.9 ± 8.5 | 2.52 ± 0.41 | 1.53 ± 0.14 | 0.528 ± 0.038 | 0.38 ± 0.0089 | nan ± nan |
| case57_ieee | dczcanon | 0.017 ± 0.00098 | 1.5 ± 0.033 | 33.9 ± 4.2 | 75.9 ± 17 | 2.83 ± 0.22 | 1.43 ± 0.031 | 0.456 ± 0.014 | 0.351 ± 0.012 | nan ± nan |
| case57_ieee | dcinjcanon | 0.0171 ± 0.0017 | 0.587 ± 0.16 | 13.1 ± 3.5 | 38.7 ± 4.1 | 3.44 ± 0.37 | 0.468 ± 0.16 | 0.334 ± 0.11 | 0.203 ± 0.038 | nan ± nan |
| case57_ieee | dcinj1hop | 0.0339 ± 0.0022 | 2.8 ± 1.4 | 38.9 ± 22 | 88.7 ± 48 | 58.8 ± 1.1 | 2.13 ± 1.4 | 1.74 ± 0.7 | 0.68 ± 0.24 | 0.0342 ± 0.0022 |

### heldout_case57_ieee: in distribution

| grid | arm | VM (p.u.) | VA (deg) | PG (MW) | QG (Mvar) | PBE (MVA) | VA offset (deg) | VA rest (deg) | θ_f − θ_t (deg) | VM, PQ buses (p.u.) |
|---|---|---|---|---|---|---|---|---|---|---|
| case14_ieee | canon | 0.000474 ± 0.00013 | 0.112 ± 0.0046 | 1.46 ± 0.17 | 1.6 ± 0.51 | 0.127 ± 0.0065 | 0.106 ± 0.0046 | 0.0354 ± 0.0017 | 0.0311 ± 0.001 | nan ± nan |
| case14_ieee | m1fullcanon | 0.00067 ± 0.00031 | 0.209 ± 0.047 | 2.47 ± 0.51 | 2.23 ± 0.73 | 0.155 ± 0.035 | 0.199 ± 0.045 | 0.0624 ± 0.014 | 0.0548 ± 0.012 | nan ± nan |
| case14_ieee | m3canon | 0.001 ± 0.0005 | 0.118 ± 0.064 | 1.47 ± 0.28 | 3.95 ± 2.1 | 0.115 ± 0.039 | 0.115 ± 0.065 | 0.0261 ± 0.0067 | 0.0336 ± 0.016 | nan ± nan |
| case14_ieee | dccanon | 0.0013 ± 0.00033 | 0.098 ± 0.018 | 0.512 ± 0.13 | 5.11 ± 1.3 | 0.0603 ± 0.0038 | 0.0973 ± 0.018 | 0.012 ± 0.0026 | 0.0296 ± 0.006 | nan ± nan |
| case14_ieee | dczcanon | 0.00213 ± 0.00067 | 0.163 ± 0.051 | 0.466 ± 0.2 | 8.45 ± 2.7 | 0.0666 ± 0.01 | 0.162 ± 0.052 | 0.0137 ± 0.0034 | 0.0483 ± 0.016 | nan ± nan |
| case14_ieee | dcinjcanon | 0.00262 ± 0.0016 | 0.21 ± 0.13 | 1.01 ± 0.39 | 10.4 ± 6.7 | 0.163 ± 0.063 | 0.208 ± 0.13 | 0.0252 ± 0.014 | 0.064 ± 0.038 | nan ± nan |
| case14_ieee | dcinj1hop | 0.0052 ± 0.00051 | 0.517 ± 0.063 | 5.98 ± 0.84 | 17.8 ± 2 | 1.64 ± 0.36 | 0.511 ± 0.063 | 0.0765 ± 0.018 | 0.179 ± 0.02 | 0.00317 ± 0.0006 |
| case30_ieee | canon | 0.00406 ± 0.00061 | 0.422 ± 0.038 | 4.74 ± 0.37 | 5.91 ± 3.2 | 0.227 ± 0.0084 | 0.409 ± 0.04 | 0.103 ± 0.0044 | 0.0753 ± 0.0039 | nan ± nan |
| case30_ieee | m1fullcanon | 0.00443 ± 0.00038 | 0.241 ± 0.034 | 2.94 ± 0.48 | 3.49 ± 1.3 | 0.205 ± 0.023 | 0.229 ± 0.033 | 0.0748 ± 0.0092 | 0.0488 ± 0.0064 | nan ± nan |
| case30_ieee | m3canon | 0.00314 ± 0.0012 | 0.364 ± 0.2 | 3.26 ± 0.96 | 10.8 ± 7 | 0.156 ± 0.027 | 0.359 ± 0.2 | 0.0619 ± 0.023 | 0.0637 ± 0.032 | nan ± nan |
| case30_ieee | dccanon | 0.0042 ± 0.00052 | 0.0506 ± 0.013 | 0.451 ± 0.086 | 2.32 ± 0.32 | 0.165 ± 0.016 | 0.0443 ± 0.014 | 0.0237 ± 0.0024 | 0.0187 ± 0.0026 | nan ± nan |
| case30_ieee | dczcanon | 0.00404 ± 0.00067 | 0.07 ± 0.0045 | 0.283 ± 0.02 | 3.29 ± 0.23 | 0.158 ± 0.017 | 0.0645 ± 0.0042 | 0.0271 ± 0.0021 | 0.0215 ± 0.002 | nan ± nan |
| case30_ieee | dcinjcanon | 0.00477 ± 0.00048 | 0.196 ± 0.07 | 1.34 ± 0.11 | 10.5 ± 6.4 | 0.207 ± 0.022 | 0.191 ± 0.073 | 0.0403 ± 0.0023 | 0.0497 ± 0.015 | nan ± nan |
| case30_ieee | dcinj1hop | 0.00855 ± 0.0019 | 0.403 ± 0.064 | 4.55 ± 1.3 | 17.5 ± 2.2 | 1.92 ± 0.24 | 0.292 ± 0.065 | 0.274 ± 0.033 | 0.192 ± 0.017 | 0.00817 ± 0.0021 |
| case118_ieee | canon | 0.00178 ± 0.00048 | 1.87 ± 0.18 | 75.8 ± 4.1 | 11.1 ± 3.4 | 1.09 ± 0.036 | 1.57 ± 0.13 | 1 ± 0.14 | 0.27 ± 0.02 | nan ± nan |
| case118_ieee | m1fullcanon | 0.00194 ± 0.00043 | 0.889 ± 0.08 | 33.8 ± 4.6 | 11.5 ± 2.7 | 0.626 ± 0.056 | 0.685 ± 0.062 | 0.565 ± 0.051 | 0.141 ± 0.014 | nan ± nan |
| case118_ieee | m3canon | 0.000876 ± 0.00025 | 1.52 ± 0.58 | 59.9 ± 25 | 5.73 ± 1.8 | 0.856 ± 0.25 | 1.25 ± 0.55 | 0.854 ± 0.23 | 0.223 ± 0.073 | nan ± nan |
| case118_ieee | dccanon | 0.00242 ± 0.0001 | 0.615 ± 0.026 | 18.6 ± 1.6 | 15.4 ± 0.55 | 0.223 ± 0.017 | 0.59 ± 0.026 | 0.173 ± 0.0044 | 0.0917 ± 0.0038 | nan ± nan |
| case118_ieee | dczcanon | 0.00241 ± 0.00012 | 0.625 ± 0.021 | 17.9 ± 1.7 | 15.2 ± 0.43 | 0.232 ± 0.02 | 0.598 ± 0.018 | 0.181 ± 0.011 | 0.0927 ± 0.0037 | nan ± nan |
| case118_ieee | dcinjcanon | 0.00146 ± 0.00024 | 0.166 ± 0.01 | 5.01 ± 0.68 | 8.75 ± 1.7 | 0.344 ± 0.067 | 0.14 ± 0.014 | 0.0848 ± 0.025 | 0.0388 ± 0.0064 | nan ± nan |
| case118_ieee | dcinj1hop | 0.0075 ± 0.00044 | 0.56 ± 0.052 | 39.3 ± 1.2 | 21.9 ± 4.8 | 6.25 ± 0.71 | 0.414 ± 0.058 | 0.375 ± 0.029 | 0.15 ± 0.014 | 0.0073 ± 0.00044 |

**P17 (held-out grids, folds 1-2): PASS** (predicted no separation). case30_ieee VA m1 [0.7911, 7.07] vs canon [3.89, 6.169]; case30_ieee VM m1 [0.005591, 0.01188] vs canon [0.008638, 0.01324]; case57_ieee VA m1 [3.006, 8.083] vs canon [7.437, 8.971]; case57_ieee VM m1 [0.01588, 0.02022] vs canon [0.01489, 0.02155]

**P18 (training grids, folds 1-2): FAIL** (predicted no separation). case14_ieee VA m1 [0.1317, 0.2125] vs canon [0.08881, 0.294]; case57_ieee VA m1 [0.1872, 0.2575] vs canon [0.227, 0.3351]; case118_ieee VA m1 [0.6861, 0.8782] vs canon [1.233, 1.99] SEPARATED; case14_ieee VA m1 [0.1419, 0.2455] vs canon [0.1081, 0.118] SEPARATED; case30_ieee VA m1 [0.1942, 0.2701] vs canon [0.3837, 0.4738] SEPARATED; case118_ieee VA m1 [0.7903, 0.9855] vs canon [1.655, 2.086] SEPARATED

**P20 (case118 held out, VA): FAIL** (predicted m1 entirely below canon): m1 [10.37, 12.74] vs canon [10.06, 12.41] deg

**P21 (not scored: P20 failed): —** m1/canon ratio of the offset 0.96, of the rest 0.99 (predicted offset ratio below)

**P23 (M3, case118 held out, VA): FAIL** (predicted m3 entirely below canon): m3 [9.277, 12.96] vs canon [10.06, 12.41] deg

**P24 (M3 vs DC, case118): PASS** (predicted some seed above DC's 1.95 deg): m3 [9.277, 12.96] deg

P25 (exploratory), zero-shot case118, m3 vs canon: dVA [1.789, 2.024] vs [1.779, 2.251]; VM [0.002398, 0.004323] vs [0.005432, 0.006928]; PBE [9.509, 13.19] vs [13.65, 20.22]

**P26 (M3 below canon on zero-shot VM, folds 1-2): FAIL** case30_ieee m3 [0.009275, 0.01313] vs canon [0.008638, 0.01324]; case57_ieee m3 [0.01614, 0.03063] vs canon [0.01489, 0.02155]

**P27 (M3 below canon on zero-shot PBE, folds 1-2): PASS** case30_ieee m3 [3.294, 4.969] vs canon [5.386, 5.823] below; case57_ieee m3 [4.906, 7.614] vs canon [8.266, 12.63] below

**P28 (ablation, case118 VM): FAIL** (predicted (i)) unresolved: resnorm [0.003403, 0.006642] vs canon [0.005432, 0.006928], m3 [0.002398, 0.004323]

P28, reported alongside (PBE): (i) residual normalization: resnorm [8.626, 10.74] vs canon [13.65, 20.22], m3 [9.509, 13.19]

**P29 (DC prior below DC, zero-shot VA): FAIL** case118_ieee dc-prior [1.897, 1.999] vs DC 1.95

**P30 (DC prior below DC, zero-shot VA): FAIL** case30_ieee dc-prior [1.979, 2.353] vs DC 1.56; case57_ieee dc-prior [1.454, 1.813] vs DC 1.08

**P31 (DC prior below canon, zero-shot VA): PASS** case118_ieee dc-prior [1.897, 1.999] vs canon [10.06, 12.41]; case30_ieee dc-prior [1.979, 2.353] vs canon [3.89, 6.169]; case57_ieee dc-prior [1.454, 1.813] vs canon [7.437, 8.971]

**P29m (DC prior <= 0.75 x DC, zero-shot VA): FAIL** case118_ieee [1.897, 1.999] vs 1.46

**P30m (DC prior <= 0.75 x DC, zero-shot VA): FAIL** case30_ieee [1.979, 2.353] vs 1.17; case57_ieee [1.454, 1.813] vs 0.81

**P33 (DC prior offset below DC's on case118): FAIL** offset [1.458, 1.606] vs DC 1.6 deg

**P34 (dcinjcanon offset <= DC's, zero-shot): FAIL** case118_ieee offset [0.9683, 4.769] vs DC 1.6; case30_ieee offset [0.1997, 1.35] vs DC 1.51 ok; case57_ieee offset [0.2734, 0.6764] vs DC 0.393

**P35 (dcinjcanon VA <= 0.75 x DC, zero-shot): FAIL** case118_ieee VA [1.063, 5.317] vs 1.46; case30_ieee VA [0.2671, 1.453] vs 1.17; case57_ieee VA [0.3676, 0.7262] vs 0.81 ok

**P36 (dcinjcanon VA below dczcanon, zero-shot): FAIL** case118_ieee VA [1.063, 5.317] vs dcz [1.641, 2.316]; case30_ieee VA [0.2671, 1.453] vs dcz [1.564, 2.061] ok; case57_ieee VA [0.3676, 0.7262] vs dcz [1.458, 1.539] ok

**P37 (dcinjcanon VA below DC with losses, zero-shot): FAIL** case118_ieee VA [1.063, 5.317] vs 1.476; case30_ieee VA [0.2671, 1.453] vs 1.943 ok; case57_ieee VA [0.3676, 0.7262] vs 1.563 ok

**P34s (dcinjcanon offset <= 0.75 x DC's, zero-shot): FAIL** case118_ieee offset [0.9683, 4.769] vs 1.2; case30_ieee offset [0.1997, 1.35] vs 1.13; case57_ieee offset [0.2734, 0.6764] vs 0.295

**P37s (dcinjcanon VA below the P-theta iteration, zero-shot): FAIL** case118_ieee VA [1.063, 5.317] vs 2.997; case30_ieee VA [0.2671, 1.453] vs 1.003; case57_ieee VA [0.3676, 0.7262] vs 1.445 ok

**P45 (dcinj1hop rest and branch differences at or below DC's, zero-shot): incomplete** case30_ieee rest [0.4397, 0.834] vs 0.397, dVA [0.2782, 0.5191] vs 0.339; case57_ieee rest [1.103, 2.719] vs 1.01, dVA [0.4486, 1.013] vs 0.532

**P39 (REF voltage given vs hidden, zero-shot VA, predicted no separation): incomplete** case30_ieee vref [7.687, 7.687] vs canon [3.89, 6.169] SEPARATED

**P40 (REF voltage given, zero-shot VM on PQ buses below canon on case30 and case118): incomplete** case30_ieee vref [0.0134, 0.0134] vs canon [0.008602, 0.0135]
