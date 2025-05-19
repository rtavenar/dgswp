#!/usr/bin/env python
# coding: utf-8


import torch
import math
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Any, Tuple
from torch import Tensor

from scipy.stats import gaussian_kde

from lib_hyp import sampleWrappedNormal, lorentz_to_poincare



params = {"text.usetex": True,
      "font.family": "serif",               # Match LaTeX default serif font (Computer Modern)
      "font.serif": ["Computer Modern Roman"],
      "text.latex.preamble": r"\usepackage{amsmath}",
      'legend.fontsize': 13,
      'axes.labelsize': 13,
      'axes.titlesize': 13,
      'mathtext.fontset': 'cm',
      'mathtext.rm': 'serif', 
      "ytick.left" : False,
      'lines.linewidth': 2.
      }
matplotlib.rcParams.update(params)


eps = 1e-7
max_clamp_norm = 40
max_norm = 85
ln_2: torch.Tensor = math.log(2)

radius = torch.Tensor([1.0])
device = "cpu"


def expand_proj_dims(x: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(x.shape[:-1] + torch.Size([1])).to(x.device).to(x.dtype)
    return torch.cat((zeros, x), dim=-1)

# We will use this clamping technique to ensure numerical stability of the Exp and Log maps
class LeakyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None

def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)

def cosh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.cosh(x)

def sinh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.sinh(x)

# Exp map for the origin has a special form which doesn't need the Lorentz norm
def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    assert x[..., 0].allclose(torch.zeros_like(x[..., 0]))
    x = x[..., 1:]
    x_norm = torch.norm(x, p=2, keepdim=True, dim=-1) / radius
    x_normed = F.normalize(x, p=2, dim=-1) * radius
    ret = torch.cat((cosh(x_norm) * radius, sinh(x_norm) * x_normed), dim=-1)
    assert torch.isfinite(ret).all()
    return ret

# Helper function to do the plotting
def plot_density(xy_poincare, probs, ax=None):
    axis_lim = 1.01

    x = xy_poincare[:, 0].view(-1, 100).detach().cpu()
    y = xy_poincare[:, 1].view(-1, 100).detach().cpu()
    z = probs.view(-1, 100).detach().cpu()

    ax.contourf(x, y, z, 100, antialiased=False, cmap='Oranges')
    ax.axis('off')

    # draw some fancy circle
    circle = plt.Circle((0, 0), 1, color='k', linewidth=2, fill=False)
    ax.add_patch(circle)
    # Makes the circle look like a circle
    ax.axis('equal')
    ax.set_xlim(-axis_lim, axis_lim)

def plot_distrib(X, ax):
    kernel = gaussian_kde(X.detach().cpu().numpy().T)

    # Map x, y coordinates on tangent space at origin to manifold (Lorentz model).
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    x, y = np.meshgrid(x, y)
    x = torch.Tensor(x).view(-1, 1)
    y = torch.Tensor(y).view(-1, 1)
    twodim = torch.cat([x, y], dim=1)
    threedim = expand_proj_dims(twodim)
    clamped_threedim = clamp(threedim, min=-max_clamp_norm,
          max=max_clamp_norm).to(X.device)

    on_mani = exp_map_mu0(clamped_threedim, radius)
    probs = torch.tensor(kernel.pdf(on_mani.T.detach().cpu().numpy()))
    # Calculate the poincare coordinates
    xy_poincare = lorentz_to_poincare(on_mani.squeeze(), radius)
    plot_density(xy_poincare, probs, ax=ax)

def plot_fig(X_target, L_hhsw, L_swp, L_dgswp, ax0, ax1, type="center", legend=True):
    #compares only with Poincare space  
    L_hhsw10 = np.log10(L_hhsw)
    L_swp10 = np.log10(L_swp)
    L_dgswp10 = np.log10(L_dgswp)

    median_hhsw = np.median(L_hhsw10, axis=0)
    q1_hhsw = np.quantile(L_hhsw10, 0.25, axis=0)
    q3_hhsw = np.quantile(L_hhsw10, 0.75, axis=0)

    median_swp = np.median(L_swp10, axis=0)
    q1_swp = np.quantile(L_swp10, 0.25, axis=0)
    q3_swp = np.quantile(L_swp10, 0.75, axis=0)

    median_dgswp = np.median(L_dgswp10, axis=0)
    q1_dgswp = np.quantile(L_dgswp10, 0.25, axis=0)
    q3_dgswp = np.quantile(L_dgswp10, 0.75, axis=0)

    n_epochs = len(median_hhsw)
    iterations = range(n_epochs)


    plot_distrib(X_target, ax0)

    ax1.plot(iterations, median_swp, label="SWD", c="C0")
    ax1.fill_between(iterations, q1_swp, q3_swp, alpha=0.5, color="C0")
    ax1.plot(iterations, median_hhsw, label="HHSW", c="C1")
    ax1.fill_between(iterations, q1_hhsw, q3_hhsw, alpha=0.5, color="C1")
    ax1.plot(iterations, median_dgswp, label="DGSWP", c="#1b9e77")
    ax1.fill_between(iterations, q1_dgswp, q3_dgswp, alpha=0.5, color="#1b9e77")

    ax1.set_xlabel("Iterations")
    ax1.set_xlim([0, n_epochs])
    ax1.set_ylabel(r"$\log_{10}(W_2^2)$")
    ax1.grid(True)
    if legend:
        ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), borderaxespad=0.)

    plt.savefig(f"fig/fig_gf_hyp_{type}.pdf", bbox_inches="tight")

mu = {
    "center": torch.tensor([1.5, np.sqrt(1.5**2-1), 0], dtype=torch.float64, device=device),
    "border": torch.tensor([8, np.sqrt(63), 0], dtype=torch.float64, device=device)}
Sigma = 0.1 * torch.tensor([[1,0],[0,1]], dtype=torch.float, device=device)

for loc in ["center", "border"]:
    X_target_wnd_c = sampleWrappedNormal(mu[loc], Sigma, 10000)

    L_hhsw_wnd_c = np.loadtxt(f"results_hyp/hhsw_loss_wnd_{loc}")
    L_swp_wnd_c = np.loadtxt(f"results_hyp/swp_loss_wnd_{loc}")
    L_swgg_wnd_c = np.loadtxt(f"results_hyp/dgswp_loss_wnd_{loc}")

    fig, ax = plt.subplots(1, 2, figsize=(7,2), gridspec_kw={"width_ratios":[1,3]})
    plt.subplots_adjust(wspace=.3)
    plot_fig(X_target_wnd_c, L_hhsw_wnd_c, L_swp_wnd_c, L_swgg_wnd_c, 
             ax[0], ax[1], loc, legend=True if loc == "border" else False)
