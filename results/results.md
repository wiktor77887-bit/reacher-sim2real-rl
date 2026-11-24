| Policy | Model | Mean reward | Mean final distance ± SE [cm] | Median [cm] | Success (< 2 cm) |
|---|---|---:|---:|---:|---:|
| Random | Nominal model | -42.33 ± 3.81 | 18.28 ± 0.84 | 16.53 | 0% |
| Random | "Real robot" | -42.65 ± 4.25 | 17.00 ± 0.83 | 16.49 | 2% |
| Random | "Real robot", weak motors | -43.32 ± 4.53 | 18.60 ± 0.82 | 17.58 | 0% |
| PPO | Nominal model | -4.18 ± 2.21 | 1.46 ± 0.15 | 1.06 | 80% |
| PPO | "Real robot" | -4.74 ± 2.47 | 1.34 ± 0.12 | 0.94 | 83% |
| PPO | "Real robot", weak motors | -5.94 ± 3.13 | 2.06 ± 0.33 | 1.21 | 75% |
| PPO + DR | Nominal model | -4.32 ± 2.09 | 1.62 ± 0.11 | 1.43 | 67% |
| PPO + DR | "Real robot" | -5.07 ± 2.59 | 1.75 ± 0.24 | 1.09 | 71% |
| PPO + DR | "Real robot", weak motors | -6.33 ± 3.22 | 3.00 ± 0.37 | 2.00 | 50% |
| SAC | Nominal model | -3.90 ± 1.66 | 1.93 ± 0.12 | 1.59 | 63% |
| SAC | "Real robot" | -4.37 ± 1.82 | 1.85 ± 0.12 | 1.56 | 65% |
| SAC | "Real robot", weak motors | -5.31 ± 2.27 | 1.87 ± 0.12 | 1.62 | 67% |
