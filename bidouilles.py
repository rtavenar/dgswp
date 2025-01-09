import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def theta2w(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def G(x, y, theta):
    w = theta2w(theta)
    x_1d = np.sort(x.dot(w))
    y_1d = np.sort(y.dot(w))
    return np.sum((x_1d - y_1d) ** 2)

def G_eps(x, y, theta, epsilon, n_samples):
    return np.mean([G(x, y, theta + epsilon * z) for z in np.random.randn(n_samples)])

def grad_G_eps_Stein(x, y, theta, epsilon, n_samples):
    G0 = G(x, y, theta)
    return np.mean([(G(x, y, theta + epsilon * z) - G0) * z / epsilon for z in np.random.randn(n_samples)])

# def dCw_dw(x, y, w):
#   x_y = x.T[:, :, None] - y.T[:, None, :]        # (d, n, n)
#   xw_yw = x.dot(w)[:, None] - y.dot(w)[None, :]  # (n, n)
#   return 2 * x_y * xw_yw[None, :, :]             # (d, n, n)

# def pi_star(x, y, theta):
#   w = theta2w(theta)
#   x_1d = x.dot(w)
#   y_1d = y.dot(w)

#   pi = np.zeros((x.shape[0], y.shape[0]))
#   pi[np.argsort(x_1d), np.argsort(y_1d)] = 1.

#   return pi

# def pi_star_epsilon(x, y, theta, epsilon, n_samples):
#   return np.mean(
#       [pi_star(x, y, theta + epsilon * z) for z in np.random.randn(n_samples)],
#       axis=0
#   )

# def grad_G_theta_Berthet(x, y, theta, epsilon, n_samples):
#   grad_G_C = pi_star_epsilon(x, y, theta, epsilon, n_samples)  # (n, n)
#   dC_dw = dCw_dw(x, y, theta2w(theta))                         # (d, n, n)
#   dG_dw = - np.sum(dC_dw * grad_G_C[None, :, :], axis=(1, 2))  # (d, )
#   return dG_dw.dot(dw_dtheta(theta))

# def grad_G(x, y, theta):
#   grad_G_C = pi_star(x, y, theta)                              # (n, n)
#   dC_dw = dCw_dw(x, y, theta2w(theta))                         # (d, n, n)
#   dG_dw = - np.sum(dC_dw * grad_G_C[None, :, :], axis=(1, 2))  # (d, )
#   return dG_dw.dot(dw_dtheta(theta))


n = 3
n_samples = 100
epsilon = .1
d = 2

np.random.seed(0)

x = np.random.randn(n, d)+0.5
y = np.random.randn(n, d)

thetas = np.linspace(-np.pi, np.pi, num=200)
Gs = [G(x, y, theta) for theta in thetas]
Gs_eps = [G_eps(x, y, theta, epsilon, n_samples) for theta in thetas]
grad_Gs_eps_Stein = {}
for n_samples in [10, 100, 1000]:
  grad_Gs_eps_Stein[n_samples] = [grad_G_eps_Stein(x, y, theta, epsilon, n_samples) for theta in thetas]
# grad_Gs_eps_Berthet = [grad_G_theta_Berthet(x, y, theta, epsilon, n_samples) for theta in thetas]
# grad_Gs = [grad_G(x, y, theta) for theta in thetas]

fig = plt.figure(figsize=(6, 8))
gs = GridSpec(4, 3, figure=fig)
# ax0 = fig.add_subplot(gs[0, 0])
# ax0.scatter(x[:, 0], x[:, 1])
# ax0.scatter(y[:, 0], y[:, 1])
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(thetas, Gs, label="$G(\\theta)$")
ax1.plot(thetas, Gs_eps, label="$G_\\varepsilon(\\theta)$")
ax1.set_xlabel("$\\theta$")
ax1.legend(loc="upper right")

theta_init = 0
lr = 1e-2
for _ in range(20):
    g = grad_G_eps_Stein(x, y, theta_init, epsilon, n_samples)
    theta_init = theta_init - lr * g
    ax1.scatter(theta_init, G(x, y, theta_init), c = "C2")

for i, n_samples in enumerate([10, 100, 1000]):
  ax2 = fig.add_subplot(gs[i + 1, :])
  ax2.axhline(y=0, color="k", linestyle="dashed")
  ax2.plot(thetas, grad_Gs_eps_Stein[n_samples], label="$\\nabla_{\\theta} G_\\varepsilon$ (Stein with variance red., " + str(n_samples) + " samples)")
  ax2.plot(thetas, gaussian_filter1d(grad_Gs_eps_Stein[n_samples], sigma=2.), 
          label="$\\nabla_{\\theta} G_\\varepsilon$ (Stein with variance red. and filtering, " + str(n_samples) + " samples)")
# ax2.plot(thetas, grad_Gs_eps_Berthet, label="$\\nabla_{\\theta} G_\\varepsilon$ (Berthet)")
# ax2.plot(thetas, grad_Gs, label="$\\nabla_{\\theta} G$")
  ax2.set_xlabel("$\\theta$")
  ax2.legend(loc=(0., 1.1))

plt.tight_layout()
plt.show()