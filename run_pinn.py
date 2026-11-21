"""Train PINNs on two exact-solution cases and compare with the FD reference.

Cases: "easy" (nu=0.05, a=1.1, smooth) and "hard" (nu=0.01, a=1.002, steep gradient).
Training window t in [0, 1]; evaluation also on t in (1, 2] (time extrapolation).
Errors are measured against the exact Cole-Hopf solution.
"""
import argparse
import json
import os
import time

import numpy as np

from burgers.fd import rollout
from burgers.metrics import mse, rel_l2
from burgers.pinn import predict, train_pinn
from burgers.plotting import plot_pinn_case, plot_pinn_compare
from burgers.problems import exact_solution, grid

OUT = "results"
CASES = {"easy": dict(nu=0.05, a=1.1, n_grid=256), "hard": dict(nu=0.01, a=1.002, n_grid=512)}
T_TRAIN, T_EVAL, DT_OUT = 1.0, 2.0, 0.01


def window_metrics(pred, ref, mask):
    return {"mse": mse(pred[mask], ref[mask]), "rel_l2": rel_l2(pred[mask], ref[mask])}


def main(iters, seed):
    os.makedirs(OUT, exist_ok=True)
    results, compare = {"iters": iters, "seed": seed}, {}
    n_out = int(round(T_EVAL / DT_OUT))
    times = np.arange(n_out + 1) * DT_OUT
    for name, cfg in CASES.items():
        nu, a = cfg["nu"], cfg["a"]
        x, dx = grid(cfg["n_grid"])
        exact = exact_solution(x[None, :], times[:, None], nu, a)
        t0 = time.time()
        fd = rollout(exact[0], DT_OUT, n_out, nu, dx)
        fd_time = time.time() - t0
        print(f"[{name}] training PINN (nu={nu}, a={a}, iters={iters})", flush=True)
        model, hist, train_time = train_pinn(nu, a, t_max=T_TRAIN, iters=iters, seed=seed)
        pinn = predict(model, x, times)
        tr, ex = times <= T_TRAIN + 1e-9, times > T_TRAIN + 1e-9
        e_p = np.linalg.norm(pinn - exact, axis=1) / np.linalg.norm(exact, axis=1)
        e_f = np.linalg.norm(fd - exact, axis=1) / np.linalg.norm(exact, axis=1)
        results[name] = {
            "nu": nu, "a": a, "n_grid_fd": cfg["n_grid"], "max_abs_u0": float(np.abs(exact[0]).max()),
            "pinn_vs_exact": {"train_window": window_metrics(pinn, exact, tr),
                              "extrapolation_window": window_metrics(pinn, exact, ex)},
            "fd_vs_exact": {"train_window": window_metrics(fd, exact, tr),
                            "extrapolation_window": window_metrics(fd, exact, ex)},
            "pinn_vs_fd": {"train_window": window_metrics(pinn, fd, tr),
                           "extrapolation_window": window_metrics(pinn, fd, ex)},
            "pinn_rel_l2_at_t": {str(t): float(e_p[int(round(t / DT_OUT))]) for t in (0.0, 0.5, 1.0, 1.5, 2.0)},
            "pinn_train_seconds": train_time, "fd_seconds": fd_time,
            "final_losses": {"pde": hist["pde"][-1], "ic": hist["ic"][-1]},
        }
        plot_pinn_case(name, x, times, exact, fd, pinn, hist, T_TRAIN, f"{OUT}/pinn_{name}.png")
        compare[name] = (times, e_p, e_f)
        print(f"[{name}] PINN rel L2 (train/extrap) = "
              f"{results[name]['pinn_vs_exact']['train_window']['rel_l2']:.3e} / "
              f"{results[name]['pinn_vs_exact']['extrapolation_window']['rel_l2']:.3e};  FD = "
              f"{results[name]['fd_vs_exact']['train_window']['rel_l2']:.3e}", flush=True)
    plot_pinn_compare(compare, T_TRAIN, f"{OUT}/pinn_compare.png")
    with open(f"{OUT}/pinn_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="smoke test (300 iterations)")
    args = p.parse_args()
    main(300 if args.quick else args.iters, args.seed)
