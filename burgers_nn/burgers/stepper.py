"""Learned time-stepper: a small periodic 1D CNN u^{n+1} = u^n + CNN(u^n) trained on
(u^n, u^{n+1}) pairs produced by the FD solver, where one learned step spans dt_nn, which can
exceed the explicit-scheme stability limit (the FD solver sub-steps internally to produce targets).
At test time the network is rolled out autoregressively.
"""
import numpy as np
import torch
import torch.nn as nn

from .fd import rollout
from .problems import random_ic


class StepperNet(nn.Module):
    def __init__(self, channels=32, depth=4, kernel=5):
        super().__init__()
        layers, c = [], 1
        for _ in range(depth - 1):
            layers += [nn.Conv1d(c, channels, kernel, padding=kernel // 2, padding_mode="circular"), nn.GELU()]
            c = channels
        layers.append(nn.Conv1d(c, 1, kernel, padding=kernel // 2, padding_mode="circular"))
        self.net = nn.Sequential(*layers)

    def forward(self, u):  # u: (B, 1, N)
        return u + self.net(u)


def make_pairs(seed, n_ics, n_grid, nu, dx, dt_nn, n_steps, amp_range=(0.5, 1.0)):
    """FD-generated training pairs. Returns (inputs, targets), each (n_ics*n_steps, n_grid)."""
    rng = np.random.default_rng(seed)
    u0 = random_ic(rng, n_ics, n_grid, amp_range=amp_range)
    traj = rollout(u0, dt_nn, n_steps, nu, dx)  # (n_steps+1, n_ics, N)
    return traj[:-1].reshape(-1, n_grid), traj[1:].reshape(-1, n_grid)


def train_stepper(x_tr, y_tr, x_va, y_va, epochs=40, batch=256, lr=2e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = StepperNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    f = lambda a: torch.tensor(a, dtype=torch.float32)[:, None, :]
    xt, yt, xv, yv = f(x_tr), f(y_tr), f(x_va), f(y_va)
    hist = {"epoch": [], "train": [], "val": []}
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(xt.shape[0])
        tot, cnt = 0.0, 0
        for i in range(0, xt.shape[0], batch):
            idx = perm[i:i + batch]
            loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); cnt += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = ((model(xv) - yv) ** 2).mean().item()
        hist["epoch"].append(ep); hist["train"].append(tot / cnt); hist["val"].append(val)
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"    epoch {ep:3d}  train={tot / cnt:.3e}  val={val:.3e}", flush=True)
    return model, hist


@torch.no_grad()
def rollout_nn(model, u0, n_steps):
    """Autoregressive rollout; returns (n_steps+1, B, N) float64 array."""
    model.eval()
    u = torch.tensor(u0, dtype=torch.float32)[:, None, :]
    out = [np.array(u0, dtype=np.float64)]
    for _ in range(n_steps):
        u = model(u)
        out.append(u[:, 0, :].numpy().astype(np.float64))
    return np.stack(out)


@torch.no_grad()
def one_step(model, u):
    """Teacher-forced single step on a batch of states (B, N) -> (B, N)."""
    model.eval()
    return model(torch.tensor(u, dtype=torch.float32)[:, None, :])[:, 0, :].numpy().astype(np.float64)
