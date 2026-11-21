# Results summary (auto-generated from results/*.json)

## FD solver (nu=0.05, a=1.1)

Convergence vs exact solution at T=0.5, dt = 0.4 dx^2/nu

| N | dt | rel L2 | max err | cell Peclet | observed order |
|---|---|---|---|---|---|
| 16 | 1.25e-01 | 5.840e-02 | 2.760e-02 | 1.71 | nan |
| 32 | 3.12e-02 | 1.353e-02 | 6.466e-03 | 0.85 | 2.11 |
| 64 | 7.81e-03 | 3.322e-03 | 1.591e-03 | 0.43 | 2.03 |
| 128 | 1.95e-03 | 8.270e-04 | 3.980e-04 | 0.21 | 2.01 |
| 256 | 4.88e-04 | 2.065e-04 | 9.954e-05 | 0.11 | 2.00 |

Stability test, N=64, T=1 (dt as multiple of the predicted limit)

| dt / dt_max | r = nu dt/dx^2 | predicted stable | blow-up step | final max abs u |
|---|---|---|---|---|
| 0.9 | 0.45 | True | None | 2.089e-01 |
| 1.1 | 0.55 | False | None | 2.250e+00 |
| 1.5 | 0.75 | False | 32 | 5.034e+121 |
| 3.0 | 1.51 | False | 17 | 6.780e+156 |

## PINN vs exact solution (iterations=6000, seed=0)

| case | model | window | MSE | rel L2 |
|---|---|---|---|---|
| easy (nu=0.05, a=1.1) | PINN | t in [0,1] (trained) | 1.050e-06 | 4.506e-03 |
| easy (nu=0.05, a=1.1) | PINN | t in (1,2] (extrapolation) | 3.307e-05 | 5.364e-02 |
| easy (nu=0.05, a=1.1) | FD (N=256) | t in [0,1] (trained) | 1.788e-09 | 1.860e-04 |
| easy (nu=0.05, a=1.1) | FD (N=256) | t in (1,2] (extrapolation) | 4.195e-10 | 1.911e-04 |
| hard (nu=0.01, a=1.002) | PINN | t in [0,1] (trained) | 2.811e-05 | 4.842e-02 |
| hard (nu=0.01, a=1.002) | PINN | t in (1,2] (extrapolation) | 2.832e-04 | 2.697e-01 |
| hard (nu=0.01, a=1.002) | FD (N=512) | t in [0,1] (trained) | 1.115e-08 | 9.646e-04 |
| hard (nu=0.01, a=1.002) | FD (N=512) | t in (1,2] (extrapolation) | 2.067e-10 | 2.304e-04 |

Wall-clock (single run, this machine)

| case | PINN training (s) | FD rollout to t=2 (s) |
|---|---|---|
| easy | 226.6 | 0.20 |
| hard | 295.3 | 0.16 |

## Learned time-stepper vs FD reference (N=64, nu=0.05, dt_nn=0.05, explicit FD limit=0.0098; 2000 training ICs, 40 epochs, 200 test ICs)

Rel. L2 is averaged over test ICs; training rollout horizon is T=1.

| scenario | one-step rel L2 | rel L2 @T=1 | rel L2 @T=3 | MSE (t<=1) | FD-from-noisy-IC rel L2 @T=1 |
|---|---|---|---|---|---|
| in-distribution | 1.730e-04 | 2.695e-03 | 9.972e-03 | 2.044e-07 | n/a |
| IC noise 0.01 | n/a | 5.023e-02 | 1.543e-01 | 6.313e-05 | 1.070e-02 |
| IC noise 0.05 | n/a | 2.508e-01 | 7.689e-01 | 1.576e-03 | 5.352e-02 |
| nu = 0.02 (unseen) | 4.361e-02 | 3.416e-01 | 4.341e-01 | 5.259e-03 | n/a |
| nu = 0.10 (unseen) | 2.211e-02 | 5.433e-01 | 9.642e-01 | 4.229e-03 | n/a |
| amplitude x1.5 | 3.536e-04 | 4.816e-03 | 1.310e-02 | 8.956e-06 | n/a |
| amplitude x2 | 1.052e-03 | 1.420e-02 | 2.262e-02 | 3.202e-04 | n/a |

Learned time-step sweep (in-distribution)

| dt_nn | dt / explicit limit | one-step rel L2 | rollout rel L2 @T=1 |
|---|---|---|---|
| 0.025 | 2.6 | 3.952e-05 | 1.252e-03 |
| 0.05 | 5.1 | 1.730e-04 | 2.695e-03 |
| 0.1 | 10.2 | 5.676e-04 | 3.922e-03 |

Rollout wall-clock for the in-distribution test set: FD 0.080s, NN 0.301s. Worst non-in-distribution scenario at T=1: nu = 0.10 (unseen).
