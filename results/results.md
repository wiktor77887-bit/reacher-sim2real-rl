| Policy | Model | Mean reward | Mean final distance ± SE [cm] | Median [cm] | Success (< 2 cm) |
|---|---|---:|---:|---:|---:|
| Random | Nominal model | -42.33 ± 3.81 | 18.28 ± 0.84 | 16.53 | 0% |
| Random | "Real robot" | -42.65 ± 4.25 | 17.00 ± 0.83 | 16.49 | 2% |
| Random | "Real robot", weak motors | -43.32 ± 4.53 | 18.60 ± 0.82 | 17.58 | 0% |
| PPO | Nominal model | -3.93 ± 1.81 | 1.01 ± 0.06 | 0.85 | 92% |
| PPO | "Real robot" | -4.52 ± 2.10 | 0.86 ± 0.06 | 0.73 | 94% |
| PPO | "Real robot", weak motors | -5.76 ± 2.70 | 1.62 ± 0.12 | 1.43 | 74% |
| PPO + DR | Nominal model | -4.39 ± 2.10 | 1.59 ± 0.10 | 1.48 | 76% |
| PPO + DR | "Real robot" | -5.02 ± 2.50 | 1.63 ± 0.19 | 1.15 | 76% |
| PPO + DR | "Real robot", weak motors | -6.31 ± 3.17 | 3.10 ± 0.38 | 2.12 | 46% |
| SAC | Nominal model | -3.58 ± 1.49 | 0.84 ± 0.06 | 0.71 | 97% |
| SAC | "Real robot" | -4.12 ± 1.67 | 0.83 ± 0.05 | 0.73 | 97% |
| SAC | "Real robot", weak motors | -5.20 ± 2.18 | 0.93 ± 0.06 | 0.78 | 96% |
