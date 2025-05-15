import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def alpha2theta(alpha):
    return np.array([np.cos(alpha), np.sin(alpha)])

def h(x, y, alpha):
    theta = alpha2theta(alpha)
    ind_x_1d = np.argsort(x.dot(theta))
    ind_y_1d = np.argsort(y.dot(theta))
    return np.sum((x[ind_x_1d] - y[ind_y_1d]) ** 2)

def g(x, y, alpha):
    theta = alpha2theta(alpha)
    x_1d = np.sort(x.dot(theta))
    y_1d = np.sort(y.dot(theta))
    return np.sum((x_1d - y_1d) ** 2)

def h_eps_N(x, y, alpha, epsilon, n_samples):
    return np.mean([h(x, y, alpha + epsilon * z) 
                    for z in np.random.randn(n_samples)])

def grad_h_eps_N_Stein(x, y, alpha, epsilon, n_samples):
    h0 = h(x, y, alpha)
    return np.mean([(h(x, y, alpha + epsilon * z) - h0) * z / epsilon 
                    for z in np.random.randn(n_samples)])


n = 3
n_samples = 10_000
epsilon = 1e-1
d = 2

np.random.seed(10)

x = np.random.randn(n, d) #+0.5
y = np.random.randn(n, d)

alphas = np.linspace(0, 2 * np.pi, num=200)
hs = [h(x, y, alpha) for alpha in alphas]
gs = [g(x, y, alpha) for alpha in alphas]
hs_eps_N = [h_eps_N(x, y, alpha, epsilon, n_samples) 
            for alpha in alphas]

params = {'legend.fontsize': 18,
      'xtick.labelsize' :18,
      'mathtext.fontset': 'cm',
      'mathtext.rm': 'serif', 
      "ytick.left" : False,
      'lines.linewidth': 2.5
      }
matplotlib.rcParams.update(params)

f, axes = plt.subplots(ncols=2, figsize=(8, 3), subplot_kw={'projection': 'polar'}, gridspec_kw={'wspace': 0.2})

axes[1].plot(alphas, hs, label="$h$")
axes[1].plot(alphas, hs_eps_N, label="$\\hat{h}_{\\varepsilon,N}$",
             color="tab:purple")
axes[0].plot(alphas, gs, label="$g$", color="C2")

axes[0].set_yticks([])
axes[1].set_yticks([])
for ax in axes:
    ax.set_xticks(ticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2], 
                    labels=["$0$", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3 \\pi}{2}$"])

handles0, labels0 = axes[0].get_legend_handles_labels()
handles1, labels1 = axes[1].get_legend_handles_labels()
f.legend(handles0 + handles1, labels0 + labels1, 
         loc="upper center", bbox_to_anchor=(.52, 1.1), ncols=1, frameon=False)

plt.savefig("fig_h_g.pdf", bbox_inches='tight')
