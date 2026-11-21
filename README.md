# 1D Burgers' equation: finite differences vs. PyTorch neural models

A small, self-contained project that solves the viscous Burgers' equation with a classical
finite-difference (FD) scheme (with stability and convergence checks) and compares it with two compact
PyTorch models: a **PINN** and a **learned time-stepper**. Only `numpy`, `torch` and `matplotlib` are used.

> **Status of the results (please read).** The FD solver, stability test and convergence study were
> executed and the numbers below come from that run. The PyTorch parts (`run_pinn.py`,
> `run_stepper.py`) were written and syntax-checked, and their non-torch plumbing was smoke-tested with
> stubs, but **they were not executed where this was authored (no PyTorch available, no network)**, so this
> README contains **no neural-network results**. Running `python run_all.py` produces them, together with
> `results/summary.md` and the figures, with every number read from the JSON files in `results/`.

## Mathematical formulation

Viscous Burgers' equation on the periodic domain x in [-1, 1):

    u_t + u u_x = nu u_xx,    u(x, 0) = u0(x)

**Exact reference.** With the Cole-Hopf transform u = -2 nu phi_x / phi, the heat-equation solution
phi = a + exp(-pi^2 nu t) cos(pi x), a > 1, gives the exact solution

    u(x,t) = 2 nu pi e^{-pi^2 nu t} sin(pi x) / (a + e^{-pi^2 nu t} cos(pi x)).

Smaller (a-1) and smaller nu give steeper gradients. Two cases are used for the PINN:
*easy* (nu=0.05, a=1.1) and *hard* (nu=0.01, a=1.002, max|u0| = 0.99, a steep gradient).

**FD scheme** (`burgers/fd.py`): forward Euler in time, central differences in space, conservative flux

    u_i^{n+1} = u_i^n - dt (F_{i+1} - F_{i-1}) / (2 dx) + nu dt (u_{i+1} - 2u_i + u_{i-1}) / dx^2,  F = u^2/2.

Order O(dt) + O(dx^2). **Stability check:** a von Neumann analysis of the frozen-coefficient
advection-diffusion problem (speed c = max|u|) gives the conditions

    nu dt / dx^2 <= 1/2     and     c^2 dt <= 2 nu    (the latter implies CFL c dt/dx <= 1).

Burgers' maximum principle (max|u| never increases) makes the initial max|u| a valid bound for c.
`advance()` raises `StabilityError` when a user-supplied `dt` violates `dt_max = min(dx^2/(2nu), 2nu/c^2)`;
by default it picks `dt <= 0.8 dt_max`. It also warns when the cell Peclet number c dx / nu exceeds 2
(spurious oscillations are then likely). The analysis is heuristic for the nonlinear equation, which is why
the blow-up test below checks it empirically.

**PINN** (`burgers/pinn.py`): u_theta(x,t) = MLP([sin(pi x), cos(pi x), t]) (4x40 tanh), which is exactly
periodic in x. The loss is the mean PDE residual^2 at 4096 random collocation points (autograd derivatives)
plus 10 x the initial-condition error. No solution data is used.

**Learned time-stepper** (`burgers/stepper.py`): a periodic 1D CNN with `u^{n+1} = u^n + CNN(u^n)`, trained on
(u^n, u^{n+1}) pairs from the FD solver on N=64 points with random smooth ICs (3 Fourier modes,
max|u0| in [0.5, 1], nu = 0.05). One learned step is dt = 0.05, about 5x the explicit stability limit
(0.0098); the FD solver sub-steps internally to make the targets. At test time it is rolled out autoregressively.

## Metrics

`MSE = mean((pred-ref)^2)` and relative L2 `||pred-ref||_2 / ||ref||_2` (`burgers/metrics.py`).
PINN errors are measured against the exact solution (FD is reported against it too); stepper errors are
measured against the FD reference on the same grid.

## Experiments

| Script | What it does |
|---|---|
| `run_fd.py` | convergence study vs exact solution; stability test with dt above the limit |
| `run_pinn.py` | PINN on easy/hard cases; profiles, error over time, extrapolation to t in (1,2] after training on [0,1] |
| `run_stepper.py` | stepper: in-distribution, IC noise (0.01, 0.05), unseen nu (0.02, 0.10), larger amplitudes (x1.5, x2), 3x longer rollout (T=3 vs trained T=1), learned step dt in {0.025, 0.05, 0.1} |
| `report.py` | writes `results/summary.md` from the JSON files |

## How to run

    pip install -r requirements.txt
    python run_all.py              # everything (PINN: 6000 iterations per case; CPU)
    python run_all.py --quick      # smoke test with tiny budgets (numbers not meaningful)
    python run_fd.py               # numpy only
    python run_pinn.py --iters 6000 --seed 0
    python run_stepper.py

Outputs go to `results/`: `fd_checks.png`, `pinn_easy.png`, `pinn_hard.png`, `pinn_compare.png`,
`stepper.png`, `*_metrics.json`, `summary.md`. Everything is seeded, but results can still vary across
PyTorch versions/hardware, so treat a single seed as one sample, not a distribution.

## Results actually obtained (FD solver, `python run_fd.py`)

Convergence vs the exact solution (nu=0.05, a=1.1, T=0.5, dt = 0.4 dx^2/nu):

| N | rel L2 | max err | observed order |
|---|---|---|---|
| 16 | 5.840e-02 | 2.760e-02 | n/a |
| 32 | 1.353e-02 | 6.466e-03 | 2.11 |
| 64 | 3.322e-03 | 1.591e-03 | 2.03 |
| 128 | 8.270e-04 | 3.980e-04 | 2.01 |
| 256 | 2.065e-04 | 9.954e-05 | 2.00 |

The observed order is about 2, as expected with dt proportional to dx^2.

Stability test (N=64, T=1, dt_max = predicted limit):

| dt / dt_max | predicted stable | outcome |
|---|---|---|
| 0.9 | yes | max abs u = 0.209 at T=1, no blow-up |
| 1.1 | no | no overflow within T=1, but max abs u has grown to 2.25 (from 0.69 initially) and is still rising: slow instability |
| 1.5 | no | overflow at step 32 (max abs u ~ 1e121 before stopping) |
| 3.0 | no | overflow at step 17 |

In the hard PINN case, a separate check showed FD with N=512 has relative L2 error of about 9.6e-4 vs the
exact solution over t in [0,1] (ad-hoc check, not part of the scripts; `run_pinn.py` records it).

## Key limitations

- **Scope.** Periodic domain, smooth ICs, viscous regime. No inviscid shocks, no boundary conditions other
  than periodic, no 2D. The FD scheme is only first-order in time and central differencing is not shock-capturing;
  at cell Peclet numbers above 2 it oscillates. Use upwind/WENO/finite-volume schemes for that regime.
- **Stability bound is heuristic** (frozen-coefficient analysis); it matched the blow-up test above, but
  that is evidence for one problem, not a proof.
- **PINN** is trained per problem instance (one nu, one IC); it is not a general solver. It is typically far
  more expensive than FD for this problem and has no built-in guarantee outside the trained time window or
  on steep gradients; `run_pinn.py` tests both (hard case, extrapolation to t in (1,2]).
- **Learned stepper** is tied to its grid, nu, amplitude range and step size; errors compound in rollouts.
  `run_stepper.py` quantifies this on unseen nu, amplitudes, noise and longer horizons. It is trained against the
  FD solver, so it can at best reproduce FD's own error (including the FD discretisation error).
- **Single seed / small models / modest budgets.** No hyperparameter tuning was done. Any conclusion
  should be drawn from `results/summary.md` generated by your own run, ideally over several seeds.

## Layout

    burgers/problems.py   exact solution, random ICs        burgers/pinn.py     PINN model, training
    burgers/fd.py         FD solver + stability check       burgers/stepper.py  CNN stepper, data, rollout
    burgers/metrics.py    MSE, relative L2                  burgers/plotting.py figures
    run_fd.py run_pinn.py run_stepper.py run_all.py report.py
