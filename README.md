1. Using a numerical solver, data with different parameter values of \( c_1 \) and a grid size of 64x64 was generated for \( t = 0 \) to \( t = 200 \). A trained FFNO was given the same initial values as the numerical solver to predict data for \( t = 0 \) to \( t = 200 \). The computed values of \( \Gamma_n \pm \delta \Gamma_n \) are shown below:

| c1  | Numerical Solver \( \Gamma_n \pm \delta \Gamma_n \) | FFNO \( \Gamma_n \pm \delta \Gamma_n \) |
|-----|-----------------------------------------------------|-----------------------------------------|
| 0.01 | 1.52 ± 1.46                                        | 1.53 ± 1.40                             |
| 0.5  | 0.45 ± 0.33                                        | 0.44 ± 0.32                             |
| 1.0  | 0.22 ± 0.23                                        | 0.22 ± 0.22                             |
| 1.5  | 0.09 ± 0.14                                        | 0.09 ± 0.14                             |

2. For \( c_1 = 1.0 \), changing the grid resolution results in the following \( \Gamma_n \pm \delta \Gamma_n \):
| Grid size | Numerical solver \( \Gamma_n \pm \sigma_{\Gamma_n} \) | F-FNO \( \Gamma_n \pm \sigma_{\Gamma_n} \) |
|-----------|-------------------------------------------------------|-------------------------------------------|
| 64x64     | 0.22 ± 0.23                                          | 0.22 ± 0.22                               |
| 128x128   | 0.26 ± 0.22                                          | 0.25 ± 0.23                               |
