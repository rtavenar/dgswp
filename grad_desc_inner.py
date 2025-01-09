import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def G(x, y, w):
    x_1d = np.sort(x.dot(w))
    y_1d = np.sort(y.dot(w))
    return np.sum((x_1d - y_1d) ** 2)

def G_eps(x, y, w, epsilon, n_samples):
    return np.mean([G(x, y, w + epsilon * z) for z in np.random.randn(n_samples)])

def grad_G_eps_Stein(x, y, w, epsilon, n_samples):
    G0 = G(x, y, w)
    return np.mean([(G(x, y, w + epsilon * z) - G0) * z / epsilon for z in np.random.randn(n_samples)])


n = 30
n_iter_grad = 100
learning_rate = 1e-3
epsilon = .1
d = 10

np.random.seed(0)

x = np.random.randn(n, d)
y = np.random.randn(n, d) + np.ones((1, d)) / d

w0 = np.random.randn(d, )
w0 /= np.linalg.norm(w0)

plt.figure()
for n_samples in [10, 100, 1000]:
    w = [w0]
    G_vals = []
    for _ in range(n_iter_grad):
        G_vals.append(G(x, y, w[-1]))
        g = grad_G_eps_Stein(x, y, w[-1], epsilon, n_samples)
        new_w = w[-1] - learning_rate * g
        w.append(new_w / np.linalg.norm(new_w))
    # print(G_vals)
    plt.plot(G_vals, label=f"n_samples={n_samples}")
    print("Dernier w:", w[-1])
plt.legend()
plt.title(f"Descente de gradient en dimension {d}")
plt.xlabel("Iterations")
plt.ylim([0, 10])
plt.tight_layout()
plt.show()