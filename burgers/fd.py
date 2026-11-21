"""Classical finite-difference solver for u_t + (u^2/2)_x = nu u_xx on a periodic grid.

Scheme: forward Euler in time, 2nd-order central differences in space for both the
conservative flux (u^2/2)_x and the diffusion term (FTCS). Accuracy: O(dt) + O(dx^2).

Stability (von Neumann analysis of the frozen-coefficient advection-diffusion equation with
speed c = max|u|): the explicit scheme is stable iff
    r = nu dt / dx^2 <= 1/2          (diffusion limit)
    c^2 dt <= 2 nu                   (advective limit, implies CFL c dt/dx <= 1)
For Burgers the maximum principle gives max|u(t)| <= max|u(0)|, so using the initial max is a
valid bound. Separately, cell Peclet number c dx / nu <= 2 is needed to avoid spurious
oscillations (not an instability, but a resolution requirement).
"""
import math
import warnings

import numpy as np


class StabilityError(RuntimeError):
    pass


def stability_report(umax, dx, dt, nu):
    dt_diff = dx ** 2 / (2.0 * nu)
    dt_adv = 2.0 * nu / umax ** 2 if umax > 0 else math.inf
    dt_max = min(dt_diff, dt_adv)
    return {
        "dt": dt,
        "dt_max": dt_max,
        "diffusion_number": nu * dt / dx ** 2,
        "cfl": umax * dt / dx,
        "cell_peclet": umax * dx / nu,
        "dt_stable": bool(dt <= dt_max * (1 + 1e-12)),
        "grid_resolved": bool(umax * dx / nu <= 2.0),
    }


def _rhs(u, dx, nu):
    up, um = np.roll(u, -1, axis=-1), np.roll(u, 1, axis=-1)
    flux_x = (up ** 2 - um ** 2) / (4.0 * dx)  # d/dx (u^2/2), central
    uxx = (up - 2.0 * u + um) / dx ** 2
    return -flux_x + nu * uxx


def advance(u0, T, nu, dx, dt=None, safety=0.8, check="raise", track_max=False):
    """Advance u0 (shape (..., N)) by time T. If dt is None, pick dt = T/n_steps with
    dt <= safety * dt_max. If dt is given it is used (rounded so n*dt = T) and checked.

    check: "raise" -> StabilityError if dt violates the stability limit; "warn"; "ignore".
    Returns (u, info). info has the stability report, n_steps, and optionally the max|u|
    history (useful to visualise blow-up when check="ignore").
    """
    u = np.array(u0, dtype=float)
    umax = float(np.max(np.abs(u)))
    dt_max = stability_report(umax, dx, 1.0, nu)["dt_max"]
    n = max(1, math.ceil(T / (safety * dt_max))) if dt is None else max(1, int(round(T / dt)))
    dt_eff = T / n
    rep = stability_report(umax, dx, dt_eff, nu)
    if not rep["dt_stable"]:
        msg = (f"dt={dt_eff:.3e} exceeds stable limit {rep['dt_max']:.3e} "
               f"(r={rep['diffusion_number']:.2f}, CFL={rep['cfl']:.2f})")
        if check == "raise":
            raise StabilityError(msg)
        if check == "warn":
            warnings.warn(msg)
    if not rep["grid_resolved"] and check != "ignore":
        warnings.warn(f"cell Peclet number {rep['cell_peclet']:.2f} > 2: expect oscillations")
    hist = [umax] if track_max else None
    blew_up_at = None
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(n):
            u = u + dt_eff * _rhs(u, dx, nu)
            if track_max:
                hist.append(float(np.max(np.abs(u))))
            if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e100:
                blew_up_at = k + 1
                break
    info = {"report": rep, "n_steps": n, "dt": dt_eff, "blew_up_at": blew_up_at}
    if track_max:
        info["max_history"] = np.array(hist)
    return u, info


def rollout(u0, dt_out, n_out, nu, dx, safety=0.8):
    """Snapshots every dt_out for n_out intervals (each interval sub-stepped stably).
    Returns array (n_out+1, *u0.shape)."""
    traj = [np.array(u0, dtype=float)]
    for _ in range(n_out):
        u, _ = advance(traj[-1], dt_out, nu, dx, safety=safety)
        traj.append(u)
    return np.stack(traj)
