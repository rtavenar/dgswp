#!/usr/bin/env python
# coding: utf-8


import torch
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import gaussian_kde


radius = torch.Tensor([1.0])
device = "cpu"

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


# Helper function to do the plotting
def plot_density(xy_poincare, probs, radius, mu=None, ax=None):
    axis_lim = 1.01
    
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)

    x = xy_poincare[:, 0].view(-1, 100).detach().cpu()
    y = xy_poincare[:, 1].view(-1, 100).detach().cpu()
    z = probs.view(-1, 100).detach().cpu()
    # Define points within circle
    if mu is not None:
        mu = mu.cpu().numpy()
        plt.plot(mu[:, 0], mu[:, 1], 'b+')

    ax.contourf(x, y, z, 100, antialiased=False, cmap='Oranges')
    
    ax.axis('off')

    # draw some fancy circle
    circle = plt.Circle((0, 0), 1, color='k', linewidth=2, fill=False)
    ax.add_patch(circle)
    # Makes the circle look like a circle
    ax.axis('equal')
    ax.set_xlim(-axis_lim, axis_lim)

def plot_distrib(X, ax, h=None):
    kernel = gaussian_kde(X.detach().cpu().numpy().T)
    
    if h is not None:
        kernel.set_bandwidth(h)

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
    plot_density(xy_poincare, probs, radius,ax=ax)

def plot_fig(X_target, L_hhsw, L_swp, L_swgg_hyp, ax0, ax1, type="center", h=None, legend=True):
    #compares only with Poincare space  
    L_hhsw10 = np.log10(L_hhsw)
    L_swp10 = np.log10(L_swp)
    L_swgg_hyp10 = np.log10(L_swgg_hyp)

    mean_hhsw = np.mean(L_hhsw10, axis=0)
    std_hhsw = np.std(L_hhsw10, axis=0)

    mean_swp = np.mean(L_swp10, axis=0)
    std_swp = np.std(L_swp10, axis=0)

    mean_swgg_hyp = np.mean(L_swgg_hyp10, axis=0)
    std_swgg_hyp = np.std(L_swgg_hyp10, axis=0)

    n_epochs = len(mean_hhsw)
    iterations = range(n_epochs)


    plot_distrib(X_target, ax0, h)

    ax1.plot(iterations, mean_swp, label="SWD", c="C0")
    ax1.fill_between(iterations, mean_swp-std_swp, mean_swp+std_swp, alpha=0.5, color="C0")
    ax1.plot(iterations, mean_hhsw, label="HHSW", c="C1")
    ax1.fill_between(iterations, mean_hhsw-std_hhsw, mean_hhsw+std_hhsw, alpha=0.5, color="C1")
    ax1.plot(iterations, mean_swgg_hyp, label="DGSWP", c="#1b9e77")
    ax1.fill_between(iterations, mean_swgg_hyp-std_swgg_hyp, mean_swgg_hyp+std_swgg_hyp, alpha=0.5, color="#1b9e77")

    ax1.set_xlabel("Iterations")
    ax1.set_xlim([0, n_epochs])
    ax1.set_ylabel(r"$\log_{10}(W_2^2)$")
    ax1.grid(True)
    if legend:
        ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.)

    plt.savefig("fig/fig_gf_hyp_"+type+".pdf", format="pdf", bbox_inches="tight")

mu = torch.tensor([1.5, np.sqrt(1.5**2-1), 0], dtype=torch.float64, device=device)
Sigma = 0.1 * torch.tensor([[1,0],[0,1]], dtype=torch.float, device=device)
X_target_wnd_c = sampleWrappedNormal(mu, Sigma, 10000)

L_hhsw_wnd_c = np.loadtxt("results_hyp/hhsw_loss_wnd_center")
L_swp_wnd_c = np.loadtxt("results_hyp/swp_loss_wnd_center")
L_swgg_wnd_c = np.loadtxt("results_hyp/swgg_hyp_loss_wnd_center")

fig, ax = plt.subplots(1, 2, figsize=(7,2), gridspec_kw={"width_ratios":[1,3]})
plt.subplots_adjust(wspace=.3)
plot_fig(X_target_wnd_c, L_hhsw_wnd_c, L_swp_wnd_c, L_swgg_wnd_c, ax[0], ax[1], "center", legend=False)


mu = torch.tensor([8, np.sqrt(63), 0], dtype=torch.float64, device=device)
Sigma = 0.1 * torch.tensor([[1,0],[0,1]], dtype=torch.float, device=device)
X_target_wnd_b = sampleWrappedNormal(mu, Sigma, 10000)


L_hhsw_wnd_b = np.loadtxt("results_hyp/hhsw_loss_wnd_border")
L_swp_wnd_b = np.loadtxt("results_hyp/swp_loss_wnd_border")
L_swgg_wnd_b = np.loadtxt("results_hyp/swgg_hyp_loss_wnd_border")


fig, ax = plt.subplots(1, 2, figsize=(7,2), gridspec_kw={"width_ratios":[1,3]})
plt.subplots_adjust(wspace=0.3)
plot_fig(X_target_wnd_b, L_hhsw_wnd_b, L_swp_wnd_b, L_swgg_wnd_b, ax[0], ax[1], "border", legend=True)

