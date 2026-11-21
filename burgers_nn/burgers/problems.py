"""Problem definitions: domain, exact (Cole-Hopf) solution, random initial conditions.

Domain: x in [-1, 1) periodic (L = 2), grid x_i = -1 + i*dx, dx = L/N.
"""
import numpy as np

L = 2.0


def grid(n):
    dx = L / n
    return -1.0 + dx * np.arange(n), dx


def exact_solution(x, t, nu, a):
    """Exact periodic solution via the Cole-Hopf transform u = -2 nu phi_x / phi with
    phi(x,t) = a + exp(-pi^2 nu t) cos(pi x), a > 1 (so phi > 0 everywhere).

    phi solves the heat equation phi_t = nu phi_xx, hence u solves
    u_t + u u_x = nu u_xx. Smaller (a-1) / smaller nu => steeper gradients.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    e = np.exp(-np.pi ** 2 * nu * t)
    return 2.0 * nu * np.pi * e * np.sin(np.pi * x) / (a + e * np.cos(np.pi * x))


def random_ic(rng, n_samples, n_grid, amp_range=(0.5, 1.0), n_modes=3, mean_scale=0.2):
    """Random smooth periodic ICs: low-order Fourier modes + small mean, rescaled so that
    max|u0| ~ U(amp_range). Returns array (n_samples, n_grid)."""
    x, _ = grid(n_grid)
    k = np.arange(1, n_modes + 1)[None, :, None]
    amp = rng.standard_normal((n_samples, n_modes, 1)) / k
    phase = rng.uniform(0, 2 * np.pi, (n_samples, n_modes, 1))
    u = (amp * np.sin(k * np.pi * x[None, None, :] + phase)).sum(axis=1)
    u += rng.uniform(-mean_scale, mean_scale, (n_samples, 1))
    target = rng.uniform(*amp_range, (n_samples, 1))
    return u * target / np.max(np.abs(u), axis=1, keepdims=True)
