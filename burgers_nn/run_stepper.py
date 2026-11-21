"""Learned time-stepper vs FD reference: in-distribution, noise, unseen nu, unseen amplitude,
longer rollouts, and different learned time steps. Reference = FD solver on the same grid."""
import argparse
import json
import os
import time

import numpy as np

from burgers.fd import rollout
from burgers.metrics import mse, rel_l2_per_sample
from burgers.plotting import plot_stepper
from burgers.problems import grid, random_ic
from burgers.stepper import make_pairs, one_step, rollout_nn, train_stepper

OUT = "results"
N_GRID, NU = 64, 0.05
DT_NN = 0.05          # learned step; explicit FD stability limit here is ~0.0098
HORIZON = 1.0         # training rollout horizon (data covers t in [0, 1])
T_EVAL = 3.0          # evaluation rollout length (3x beyond training horizon)


def evaluate(model, u0, nu, dt, dx, noise=0.0, noise_seed=7):
    n = int(round(T_EVAL / dt)); k1 = int(round(HORIZON / dt))
    u_in = u0 + noise * np.random.default_rng(noise_seed).standard_normal(u0.shape) if noise > 0 else u0
    ref = rollout(u0, dt, n, nu, dx)
    t0 = time.time(); pred = rollout_nn(model, u_in, n); nn_s = time.time() - t0
    err_t = rel_l2_per_sample(pred, ref).mean(axis=1)
    res = {
        "rel_l2_T1": float(err_t[k1]), "rel_l2_T3": float(err_t[n]),
        "mse_window_T1": mse(pred[:k1 + 1], ref[:k1 + 1]), "mse_window_T3": mse(pred, ref),
        "max_abs_pred_T3": float(np.nanmax(np.abs(pred[n]))), "max_abs_ref_T3": float(np.abs(ref[n]).max()),
        "nn_rollout_seconds": nn_s,
    }
    if noise == 0.0:
        pairs_in, pairs_out = ref[:-1].reshape(-1, u0.shape[1]), ref[1:].reshape(-1, u0.shape[1])
        res["one_step_rel_l2"] = float(rel_l2_per_sample(one_step(model, pairs_in), pairs_out).mean())
    else:  # baseline: FD itself started from the same noisy IC
        fd_noisy = rollout(u_in, dt, n, nu, dx)
        e = rel_l2_per_sample(fd_noisy, ref).mean(axis=1)
        res["fd_noisy_ic_rel_l2_T1"], res["fd_noisy_ic_rel_l2_T3"] = float(e[k1]), float(e[n])
    return res, (np.arange(n + 1) * dt, err_t), (ref, pred)


def main(quick):
    os.makedirs(OUT, exist_ok=True)
    n_ics, epochs, n_test = (200, 3, 50) if quick else (2000, 40, 200)
    _, dx = grid(N_GRID)

    def train_for(dt):
        k = int(round(HORIZON / dt))
        xtr, ytr = make_pairs(1, n_ics, N_GRID, NU, dx, dt, k)
        xva, yva = make_pairs(2, max(n_ics // 10, 20), N_GRID, NU, dx, dt, k)
        print(f"  training stepper dt={dt}: {len(xtr)} pairs", flush=True)
        return train_stepper(xtr, ytr, xva, yva, epochs=epochs)

    print("[stepper] main model", flush=True)
    model, hist = train_for(DT_NN)
    test = lambda amp, seed=123: random_ic(np.random.default_rng(seed), n_test, N_GRID, amp_range=amp)
    scenarios = {
        "in-distribution": dict(u0=test((0.5, 1.0)), nu=NU),
        "IC noise 0.01": dict(u0=test((0.5, 1.0)), nu=NU, noise=0.01),
        "IC noise 0.05": dict(u0=test((0.5, 1.0)), nu=NU, noise=0.05),
        "nu = 0.02 (unseen)": dict(u0=test((0.5, 1.0)), nu=0.02),
        "nu = 0.10 (unseen)": dict(u0=test((0.5, 1.0)), nu=0.10),
        "amplitude x1.5": dict(u0=test((0.75, 1.5)), nu=NU),
        "amplitude x2": dict(u0=test((1.0, 2.0)), nu=NU),
    }
    metrics, curves, outputs = {}, {}, {}
    for name, kw in scenarios.items():
        metrics[name], curves[name], outputs[name] = evaluate(model, kw["u0"], kw["nu"], DT_NN, dx, kw.get("noise", 0.0))
        print(f"  {name:22s} relL2@T=1: {metrics[name]['rel_l2_T1']:.3e}  @T=3: {metrics[name]['rel_l2_T3']:.3e}", flush=True)
    t0 = time.time(); rollout(scenarios["in-distribution"]["u0"], DT_NN, int(round(T_EVAL / DT_NN)), NU, dx)
    fd_seconds = time.time() - t0

    sweep = []
    for dt in (0.025, 0.05, 0.1):
        m = model if dt == DT_NN else train_for(dt)[0]
        r, _, _ = evaluate(m, scenarios["in-distribution"]["u0"], NU, dt, dx)
        sweep.append({"dt_nn": dt, "dt_over_explicit_limit": dt / (dx ** 2 / (2 * NU)),
                      "one_step_rel_l2": r["one_step_rel_l2"], "rollout_rel_l2_T1": r["rel_l2_T1"],
                      "rollout_rel_l2_T3": r["rel_l2_T3"]})
        print(f"  dt sweep dt={dt}: one-step {r['one_step_rel_l2']:.3e}  T=1 {r['rel_l2_T1']:.3e}", flush=True)

    worst = max((k for k in metrics if k != "in-distribution"),
                key=lambda k: -np.inf if np.isnan(metrics[k]["rel_l2_T1"]) else metrics[k]["rel_l2_T1"])
    k1 = int(round(HORIZON / DT_NN)); x, _ = grid(N_GRID)
    profiles = {name: (x, outputs[name][0][k1, 0], outputs[name][1][k1, 0]) for name in ("in-distribution", worst)}
    plot_stepper(hist, curves, HORIZON, profiles, sweep, f"{OUT}/stepper.png")
    with open(f"{OUT}/stepper_metrics.json", "w") as f:
        json.dump({"config": dict(n_grid=N_GRID, nu=NU, dt_nn=DT_NN, explicit_dt_limit=dx ** 2 / (2 * NU),
                                  train_ics=n_ics, epochs=epochs, n_test=n_test, horizon=HORIZON, t_eval=T_EVAL,
                                  quick=quick),
                   "scenarios": metrics, "worst_scenario": worst, "dt_sweep": sweep,
                   "timing": {"fd_rollout_seconds": fd_seconds,
                              "nn_rollout_seconds": metrics["in-distribution"]["nn_rollout_seconds"]},
                   "final_train_mse": hist["train"][-1], "final_val_mse": hist["val"][-1]}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="smoke test (tiny data, 3 epochs)")
    main(p.parse_args().quick)
