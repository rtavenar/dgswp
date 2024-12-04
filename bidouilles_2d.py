import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

def angles2w(alpha_beta):
    alpha, beta = alpha_beta
    # SO: https://stackoverflow.com/questions/30011741/3d-vector-defined-by-2-angles
    return np.array([np.cos(alpha) * np.cos(beta), 
                     np.sin(beta),
                     np.sin(alpha) * np.cos(beta)])

def G(x, y, alpha_beta):
    w = angles2w(alpha_beta)
    x_1d = np.sort(x.dot(w))
    y_1d = np.sort(y.dot(w))
    return np.sum((x_1d - y_1d) ** 2)

def G_eps(x, y, alpha_beta, epsilon, n_samples):
    return np.mean([G(x, y, alpha_beta + epsilon * z) for z in np.random.randn(n_samples, 2)])

def grad_G_eps_Stein(x, y, alpha_beta, epsilon, n_samples):
    G0 = G(x, y, alpha_beta)
    return np.mean([(G(x, y, alpha_beta + epsilon * z) - G0) * z / epsilon for z in np.random.randn(n_samples, 2)],
                   axis=0)


n = 3
n_samples = 100
epsilon = .1
d = 3

np.random.seed(0)

x = np.random.randn(n, d)
y = np.random.randn(n, d)

alphas, betas = np.mgrid[-np.pi:np.pi:.1, -np.pi:np.pi:.1]
alpha_betas = np.vstack((alphas.flatten(), betas.flatten())).T

# print(alphas.shape, alpha_betas.shape)

Gs = np.array([G(x, y, alpha_beta) for alpha_beta in alpha_betas])
Gs_eps =  np.array([G_eps(x, y, alpha_beta, epsilon, n_samples) for alpha_beta in alpha_betas])
# grad_Gs_eps_Stein = {}
grad_Gs_eps_Stein =  np.array([grad_G_eps_Stein(x, y, alpha_beta, epsilon, n_samples) for alpha_beta in alpha_betas])

# samples_zero_grad = (np.abs(gaussian_filter(grad_Gs_eps_Stein.reshape(alphas.shape), sigma=2.)) < .05)
# samples_zero_grad = (np.abs(grad_Gs_eps_Stein.reshape(alphas.shape)) < .05)
# print(samples_zero_grad.sum())

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 4, figure=fig)
# ax0 = fig.add_subplot(gs[0, 0], projection="3d")
# ax0.scatter(x[:, 0], x[:, 1], x[:, 2])
# ax0.scatter(y[:, 0], y[:, 1], y[:, 2])
ax0 = fig.add_subplot(gs[0, 0], projection="3d")
ax0.plot_surface(alphas, betas, Gs.reshape(alphas.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax0.set_title("$G$")
ax1 = fig.add_subplot(gs[0, 1], projection="3d")
ax1.plot_surface(alphas, betas, Gs_eps.reshape(alphas.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_title("$G_\\epsilon$")
ax2 = fig.add_subplot(gs[1, 0], projection="3d")
ax2.plot_surface(alphas, betas, np.linalg.norm(grad_Gs_eps_Stein, axis=-1).reshape(alphas.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_title("$\\|\\nabla_{(\\alpha,\\beta)}G_\\epsilon\\|$")
ax3 = fig.add_subplot(gs[1, 1], projection="3d")
ax3.plot_surface(alphas, betas, 
                 np.linalg.norm(
                     gaussian_filter(grad_Gs_eps_Stein.reshape(alphas.shape + (2, )), sigma=2., axes=(0, 1)), 
                     axis=-1), 
                 cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax3.set_title("Norm of smoothed $\\nabla_{(\\alpha,\\beta)}G_\\epsilon$")
ax4 = fig.add_subplot(gs[:, 2:])
ax4.imshow(Gs.reshape(alphas.shape), cmap=cm.coolwarm, extent=[-np.pi, np.pi, -np.pi, np.pi])
ax4.contour(np.linalg.norm(grad_Gs_eps_Stein, axis=-1).reshape(alphas.shape), 
            levels=[0.5], colors="white", extent=[-np.pi, np.pi, -np.pi, np.pi])

learning_rate = .1
for ab_0 in np.random.randn(5, 2):
    ab = [ab_0]
    G_vals = []
    for _ in range(15):
        G_vals.append(G(x, y, ab[-1]))
        g = grad_G_eps_Stein(x, y, ab[-1], epsilon, n_samples)
        ab.append(ab[-1] - learning_rate * g)
    ax4.plot([a for a, b in ab], [b for a, b in ab], marker="o", color="orange")
    ax4.plot([a for a, b in ab[-1:]], [b for a, b in ab[-1:]], marker="o", color="black")

ax4.set_title(".5-level of $\\|\\nabla_{(\\alpha,\\beta)}G_\\epsilon\\|$ overlayed on $G$")

plt.tight_layout()
plt.show()