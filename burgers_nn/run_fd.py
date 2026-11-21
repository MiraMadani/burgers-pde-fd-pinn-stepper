"""Reference FD solver: convergence + stability checks (numpy only)."""
import json
import os

import numpy as np

from burgers.fd import advance
from burgers.metrics import rel_l2
from burgers.plotting import plot_fd_checks
from burgers.problems import exact_solution, grid

OUT = "results"
NU, A, T = 0.05, 1.1, 0.5


def convergence():
    rows = []
    for n in (16, 32, 64, 128, 256):
        x, dx = grid(n)
        dt = 0.4 * dx ** 2 / NU
        u, info = advance(exact_solution(x, 0.0, NU, A), T, NU, dx, dt=dt, check="raise")
        ex = exact_solution(x, T, NU, A)
        rows.append({"N": n, "dx": dx, "dt": info["dt"], "n_steps": info["n_steps"],
                     "rel_l2": rel_l2(u, ex), "max_err": float(np.max(np.abs(u - ex))),
                     "cell_peclet": info["report"]["cell_peclet"]})
    for prev, cur in zip(rows[:-1], rows[1:]):
        cur["observed_order"] = float(np.log2(prev["rel_l2"] / cur["rel_l2"]))
    return rows


def stability(n=64, t_end=1.0, factors=(0.9, 1.1, 1.5, 3.0)):
    x, dx = grid(n)
    u0 = exact_solution(x, 0.0, NU, A)
    umax = float(np.max(np.abs(u0)))
    dt_max = min(dx ** 2 / (2 * NU), 2 * NU / umax ** 2)
    out = []
    for f in factors:
        u, info = advance(u0, t_end, NU, dx, dt=f * dt_max, check="ignore", track_max=True)
        out.append({"factor": f, "dt": info["dt"], "dt_max": dt_max, "n_steps": info["n_steps"],
                    "diffusion_number": info["report"]["diffusion_number"],
                    "declared_stable": info["report"]["dt_stable"],
                    "blew_up_at_step": info["blew_up_at"],
                    "final_max_abs_u": float(np.max(np.abs(u))) if np.all(np.isfinite(u)) else float("inf"),
                    "rel_l2_vs_exact": rel_l2(u, exact_solution(x, t_end, NU, A)) if np.all(np.isfinite(u)) else float("inf"),
                    "max_history": info["max_history"], "x": x, "u_final": u,
                    "exact": exact_solution(x, t_end, NU, A)})
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    with np.errstate(over="ignore", invalid="ignore"):
        conv, stab = convergence(), stability()
    plot_fd_checks(conv, stab, f"{OUT}/fd_checks.png")
    slim = [{k: v for k, v in s.items() if k not in ("max_history", "x", "u_final", "exact")} for s in stab]
    with open(f"{OUT}/fd_metrics.json", "w") as f:
        json.dump({"problem": {"nu": NU, "a": A, "T_convergence": T, "T_stability": 1.0},
                   "convergence": conv, "stability": slim}, f, indent=2)
    print("Convergence (rel L2 vs exact, dt = 0.4 dx^2/nu):")
    for r in conv:
        print(f"  N={r['N']:4d} dt={r['dt']:.2e} rel_l2={r['rel_l2']:.3e} max={r['max_err']:.3e} "
              f"Pe={r['cell_peclet']:.2f} order={r.get('observed_order', float('nan')):.2f}")
    print("Stability (N=64, T=1):")
    for s in slim:
        print(f"  dt={s['factor']:.1f} dt_max (r={s['diffusion_number']:.2f}) declared_stable={s['declared_stable']} "
              f"blow-up step={s['blew_up_at_step']} final max|u|={s['final_max_abs_u']:.3e}")


if __name__ == "__main__":
    main()
