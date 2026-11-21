"""Build results/summary.md from the JSON files written by the run_*.py scripts.
Every number in the summary is read from those files (nothing is typed by hand)."""
import json
import os

R = "results"


def load(name):
    p = f"{R}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    L = ["# Results summary (auto-generated from results/*.json)\n"]
    fd = load("fd_metrics.json")
    if fd:
        L += [f"## FD solver (nu={fd['problem']['nu']}, a={fd['problem']['a']})\n",
              "Convergence vs exact solution at T=0.5, dt = 0.4 dx^2/nu\n",
              "| N | dt | rel L2 | max err | cell Peclet | observed order |", "|---|---|---|---|---|---|"]
        for r in fd["convergence"]:
            L.append(f"| {r['N']} | {r['dt']:.2e} | {r['rel_l2']:.3e} | {r['max_err']:.3e} | "
                     f"{r['cell_peclet']:.2f} | {r.get('observed_order', float('nan')):.2f} |")
        L += ["\nStability test, N=64, T=1 (dt as multiple of the predicted limit)\n",
              "| dt / dt_max | r = nu dt/dx^2 | predicted stable | blow-up step | final max abs u |", "|---|---|---|---|---|"]
        for s in fd["stability"]:
            L.append(f"| {s['factor']:.1f} | {s['diffusion_number']:.2f} | {s['declared_stable']} | "
                     f"{s['blew_up_at_step']} | {s['final_max_abs_u']:.3e} |")
    pn = load("pinn_metrics.json")
    if pn:
        L += [f"\n## PINN vs exact solution (iterations={pn['iters']}, seed={pn['seed']})\n",
              "| case | model | window | MSE | rel L2 |", "|---|---|---|---|---|"]
        for case in ("easy", "hard"):
            c = pn[case]
            for key, label in (("pinn_vs_exact", "PINN"), ("fd_vs_exact", f"FD (N={c['n_grid_fd']})")):
                for w, wl in (("train_window", "t in [0,1] (trained)"), ("extrapolation_window", "t in (1,2] (extrapolation)")):
                    m = c[key][w]
                    L.append(f"| {case} (nu={c['nu']}, a={c['a']}) | {label} | {wl} | {m['mse']:.3e} | {m['rel_l2']:.3e} |")
        L += ["\nWall-clock (single run, this machine)\n", "| case | PINN training (s) | FD rollout to t=2 (s) |", "|---|---|---|"]
        for case in ("easy", "hard"):
            L.append(f"| {case} | {pn[case]['pinn_train_seconds']:.1f} | {pn[case]['fd_seconds']:.2f} |")
    st = load("stepper_metrics.json")
    if st:
        c = st["config"]
        L += [f"\n## Learned time-stepper vs FD reference (N={c['n_grid']}, nu={c['nu']}, dt_nn={c['dt_nn']}, "
              f"explicit FD limit={c['explicit_dt_limit']:.4f}; {c['train_ics']} training ICs, {c['epochs']} epochs, "
              f"{c['n_test']} test ICs{', QUICK MODE' if c['quick'] else ''})\n",
              "Rel. L2 is averaged over test ICs; training rollout horizon is T=1.\n",
              "| scenario | one-step rel L2 | rel L2 @T=1 | rel L2 @T=3 | MSE (t<=1) | FD-from-noisy-IC rel L2 @T=1 |",
              "|---|---|---|---|---|---|"]
        for k, m in st["scenarios"].items():
            one = f"{m['one_step_rel_l2']:.3e}" if "one_step_rel_l2" in m else "n/a"
            fdn = f"{m['fd_noisy_ic_rel_l2_T1']:.3e}" if "fd_noisy_ic_rel_l2_T1" in m else "n/a"
            L.append(f"| {k} | {one} | {m['rel_l2_T1']:.3e} | {m['rel_l2_T3']:.3e} | {m['mse_window_T1']:.3e} | {fdn} |")
        L += ["\nLearned time-step sweep (in-distribution)\n",
              "| dt_nn | dt / explicit limit | one-step rel L2 | rollout rel L2 @T=1 |", "|---|---|---|---|"]
        for d in st["dt_sweep"]:
            L.append(f"| {d['dt_nn']} | {d['dt_over_explicit_limit']:.1f} | {d['one_step_rel_l2']:.3e} | {d['rollout_rel_l2_T1']:.3e} |")
        t = st["timing"]
        L.append(f"\nRollout wall-clock for the in-distribution test set: FD {t['fd_rollout_seconds']:.3f}s, "
                 f"NN {t['nn_rollout_seconds']:.3f}s. Worst non-in-distribution scenario at T=1: {st['worst_scenario']}.")
    open(f"{R}/summary.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {R}/summary.md")


if __name__ == "__main__":
    main()
