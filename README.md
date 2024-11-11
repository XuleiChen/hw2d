# Gamma_n Comparison for FFNO and Numerical Solver

1. Data for various values of parameter `c1` was generated using a numerical solver on a grid size of 64x64, covering the time range \( t = 0 \) to \( t = 200 \).
2. Using identical initial conditions as the numerical solver, a trained Fourier Neural Operator (FFNO) was tasked to infer the data for \( t = 0 \) to \( t = 200 \).
3. The resulting values of \( \Gamma_n \pm \delta \Gamma_n \) from the FFNO's inference are shown in comparison to the numerical solver's output:

| c1  | Numerical Solver \( \Gamma_n \pm \delta \Gamma_n \) | FFNO \( \Gamma_n \pm \delta \Gamma_n \) |
|-----|-----------------------------------------------------|-----------------------------------------|
| 0.01 | 1.52 ± 1.46                                        | 1.53 ± 1.40                             |
| 0.5  | 0.45 ± 0.33                                        | 0.44 ± 0.32                             |
| 1.0  | 0.22 ± 0.23                                        | 0.22 ± 0.22                             |
| 1.5  | 0.09 ± 0.14                                        | 0.09 ± 0.14                             |

