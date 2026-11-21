"""Physics-informed neural network (PINN) for 1D Burgers with periodic x in [-1, 1].

u_theta(x, t) = MLP([sin(pi x), cos(pi x), t]); the sin/cos features make the solution exactly
2-periodic in x. Loss = mean PDE residual^2 at random collocation points
         + lam_ic * mean (u(x,0) - u0(x))^2.
No solution data is used: only the PDE and the initial condition.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, width=40, depth=4):
        super().__init__()
        layers, c = [], 3
        for _ in range(depth):
            layers += [nn.Linear(c, width), nn.Tanh()]
            c = width
        layers.append(nn.Linear(c, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([torch.sin(math.pi * x), torch.cos(math.pi * x), t], dim=1))


def residual(model, x, t, nu):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def ic_torch(x, nu, a):
    return 2.0 * nu * math.pi * torch.sin(math.pi * x) / (a + torch.cos(math.pi * x))


def train_pinn(nu, a, t_max=1.0, iters=6000, n_f=4096, n_ic=256, width=40, depth=4,
               lr=1e-3, lam_ic=10.0, seed=0, log_every=250, verbose=True):
    torch.manual_seed(seed)
    model = PINN(width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.01 ** (1.0 / iters))  # lr -> lr/100
    hist = {"iter": [], "pde": [], "ic": []}
    t0 = time.time()
    for it in range(1, iters + 1):
        x = torch.rand(n_f, 1) * 2 - 1
        t = torch.rand(n_f, 1) * t_max
        l_pde = (residual(model, x, t, nu) ** 2).mean()
        x0 = torch.rand(n_ic, 1) * 2 - 1
        l_ic = ((model(x0, torch.zeros_like(x0)) - ic_torch(x0, nu, a)) ** 2).mean()
        loss = l_pde + lam_ic * l_ic
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == 1:
            hist["iter"].append(it); hist["pde"].append(l_pde.item()); hist["ic"].append(l_ic.item())
            if verbose:
                print(f"    iter {it:5d}  pde={l_pde.item():.3e}  ic={l_ic.item():.3e}", flush=True)
    return model, hist, time.time() - t0


@torch.no_grad()
def predict(model, x, times):
    """Evaluate on tensor-product grid; returns array (len(times), len(x))."""
    model.eval()
    X, Tt = np.meshgrid(x, times)  # (Nt, Nx)
    xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
    tt = torch.tensor(Tt.reshape(-1, 1), dtype=torch.float32)
    return model(xt, tt).numpy().reshape(X.shape).astype(np.float64)
