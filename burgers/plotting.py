import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fd_checks(conv, stab, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    h = np.array([c["dx"] for c in conv])
    ax[0].loglog(h, [c["rel_l2"] for c in conv], "o-", label="relative L2 error")
    ax[0].loglog(h, [c["max_err"] for c in conv], "s--", label="max error")
    ax[0].loglog(h, conv[0]["rel_l2"] * (h / h[0]) ** 2, "k:", label="slope 2")
    ax[0].set_xlabel("dx"); ax[0].set_ylabel("error vs exact solution")
    ax[0].set_title("Convergence (dt = 0.4 dx^2/nu)"); ax[0].legend()
    for s in stab:
        steps = np.arange(len(s["max_history"]))
        ax[1].semilogy(steps, np.maximum(s["max_history"], 1e-12),
                       label=f"dt = {s['factor']:.1f} x dt_max")
    ax[1].set_xlabel("time step"); ax[1].set_ylabel("max |u|")
    ax[1].set_title("Stability check: max|u| history"); ax[1].legend()
    ax[1].set_ylim(1e-2, 1e12)
    ax[2].plot(stab[0]["x"], stab[0]["exact"], "k-", lw=2, label="exact")
    ax[2].plot(stab[0]["x"], stab[0]["u_final"], "--", label=f"FD, {stab[0]['factor']:.1f} x dt_max")
    ax[2].set_xlabel("x"); ax[2].set_ylabel("u(x,T)")
    ax[2].set_title("Profile at T (stable run)"); ax[2].legend()
    save(fig, path)


def plot_pinn_case(name, x, times, exact, fd, pinn, loss_hist, t_train, path, t_show=(0.25, 0.5, 1.0, 1.5)):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, ts in zip(ax[0, :], (t_show[0], t_show[2], t_show[3])):
        k = int(np.argmin(np.abs(times - ts)))
        a.plot(x, exact[k], "k-", lw=2, label="exact")
        a.plot(x, fd[k], "g--", label="FD")
        a.plot(x, pinn[k], "r:", lw=2, label="PINN")
        a.set_title(f"{name}: t = {times[k]:.2f}"); a.set_xlabel("x"); a.legend()
    e_p = np.linalg.norm(pinn - exact, axis=1) / np.linalg.norm(exact, axis=1)
    e_f = np.linalg.norm(fd - exact, axis=1) / np.linalg.norm(exact, axis=1)
    ax[1, 0].semilogy(times, e_p, "r-", label="PINN"); ax[1, 0].semilogy(times, e_f, "g-", label="FD")
    ax[1, 0].axvline(t_train, color="gray", ls="--", label="end of training window")
    ax[1, 0].set_xlabel("t"); ax[1, 0].set_ylabel("relative L2 error vs exact"); ax[1, 0].legend()
    ax[1, 0].set_title("Error over time")
    im = ax[1, 1].imshow(np.abs(pinn - exact), aspect="auto", origin="lower",
                         extent=[x[0], x[-1], times[0], times[-1]])
    ax[1, 1].set_title("|PINN - exact|"); ax[1, 1].set_xlabel("x"); ax[1, 1].set_ylabel("t")
    plt.colorbar(im, ax=ax[1, 1])
    ax[1, 2].semilogy(loss_hist["iter"], loss_hist["pde"], label="PDE residual")
    ax[1, 2].semilogy(loss_hist["iter"], loss_hist["ic"], label="initial condition")
    ax[1, 2].set_xlabel("iteration"); ax[1, 2].set_title("Training loss"); ax[1, 2].legend()
    save(fig, path)


def plot_pinn_compare(cases, t_train, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (times, e_p, e_f) in cases.items():
        ax.semilogy(times, e_p, label=f"PINN, {name}")
        ax.semilogy(times, e_f, "--", label=f"FD, {name}")
    ax.axvline(t_train, color="gray", ls="--")
    ax.set_xlabel("t"); ax.set_ylabel("relative L2 error vs exact"); ax.legend()
    ax.set_title("PINN vs FD: easy / hard case and time extrapolation")
    save(fig, path)


def plot_stepper(histories, scen_err, horizon, profiles, dt_sweep, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].semilogy(histories["epoch"], histories["train"], label="train")
    ax[0, 0].semilogy(histories["epoch"], histories["val"], label="validation")
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("one-step MSE"); ax[0, 0].legend()
    ax[0, 0].set_title("Learned stepper: training")
    for name, (t, e) in scen_err.items():
        ax[0, 1].semilogy(t, np.maximum(e, 1e-8), label=name)
    ax[0, 1].axvline(horizon, color="gray", ls="--", label="training rollout horizon")
    ax[0, 1].set_xlabel("t"); ax[0, 1].set_ylabel("rel. L2 error vs FD (mean over ICs)")
    ax[0, 1].set_title("Rollout error by scenario"); ax[0, 1].legend(fontsize=7)
    for name, (x, ref, pred) in profiles.items():
        l, = ax[1, 0].plot(x, ref, "-", lw=2, label=f"FD: {name}")
        ax[1, 0].plot(x, pred, "--", color=l.get_color(), label=f"NN: {name}")
    ax[1, 0].set_xlabel("x"); ax[1, 0].set_title("Profiles at T = 1 (first test IC)")
    ax[1, 0].legend(fontsize=7)
    dts = [d["dt_nn"] for d in dt_sweep]
    ax[1, 1].loglog(dts, [d["one_step_rel_l2"] for d in dt_sweep], "o-", label="one-step")
    ax[1, 1].loglog(dts, [d["rollout_rel_l2_T1"] for d in dt_sweep], "s-", label="rollout to T=1")
    ax[1, 1].set_xlabel("learned time step dt"); ax[1, 1].set_ylabel("rel. L2 error")
    ax[1, 1].set_title("Effect of the learned time step"); ax[1, 1].legend()
    save(fig, path)
