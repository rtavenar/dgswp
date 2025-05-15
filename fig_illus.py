import torch
import torch.nn as nn
from torchdyn.datasets import generate_moons
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import ot
import ot.plot

from losses import dgswp


def wass(x, y):
    dists = ot.utils.dist(x, y)
    a = torch.ones(x.shape[0])/x.shape[0]
    b = torch.ones(y.shape[0])/y.shape[0]
    
    return ot.emd2(a, b, dists)

def min_swgg_random(x, y, n_directions):
    thetas = np.random.randn(n_directions, 2)
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True)
    best_theta_idx = -1
    min_cost = np.inf
    for i, th in enumerate(thetas):
        indices_x = torch.argsort(x @ th)
        indices_y = torch.argsort(y @ th)
        sorted_x = x[indices_x].detach().numpy()
        sorted_y = y[indices_y].detach().numpy()
        cost = np.mean(np.sum(np.abs(sorted_x - sorted_y) ** 2, axis = -1))
        if cost < min_cost:
            min_cost = cost
            best_theta_idx = i
    return thetas[best_theta_idx], cost

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), np.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1.5, 0),
        (-1.75, 0.5),
        (0, 1.5),
        (0, -1),
        (.75 / np.sqrt(2), 1. / np.sqrt(2)),
        (1.3 / np.sqrt(2), -1. / np.sqrt(2)),
        (-1.95 / np.sqrt(2), 2 / np.sqrt(2)),
        (-1 / np.sqrt(2), -0.25 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.zeros(n//8, dtype=torch.int8)
    for i in range(1,8):
        multi = torch.cat((multi, i*torch.ones(n//8, dtype=torch.int8)))
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()

def sample_moons(n):
    x0, y = generate_moons(n, noise=0.2)
    x0[torch.where(y==0)[0],1] += .3
    return x0 * 3 - 1

# Define 1-hidden-layer MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Set parameters
n_samples = 1024
dim = 2

n_iter = 10_000
epsilon_Stein = 5e-1
n_samples_Stein = 20

# Load data
torch.manual_seed(2025)
np.random.seed(0)
x = sample_8gaussians(n_samples)
y = sample_moons(n_samples)
y[:,0] -= 2
y[:,1] *= 1.1
y[:,1] += 1

# OT
C = ot.dist(x, y)
cost = ot.emd2([], [], C.numpy())
plan = ot.emd([], [], C.numpy())

# min-SWGG
theta, cost_swgg = min_swgg_random(x, y, n_directions=100)
indices_x = torch.argsort(x @ theta)
indices_y = torch.argsort(y @ theta)
x0_plot_swgg = x[indices_x].detach().numpy()
x1_plot_swgg = y[indices_y].detach().numpy()

# min-DGSWP
torch.manual_seed(1)
model = MLP(input_dim=dim)
optimizer = torch.optim.SGD(model.parameters(), lr=.2)

dgswp(
    x, y, model, optimizer, n_iter=n_iter,
    roll_back=True, variance_reduction=True,
    n_samples_Stein=n_samples_Stein, epsilon_Stein=epsilon_Stein
)
indices_x = torch.argsort(model(x).flatten())
indices_y = torch.argsort(model(y).flatten())
x0_plot_nn = x[indices_x].detach().numpy()
x1_plot_nn = y[indices_y].detach().numpy()
cost_nn = np.mean(np.sum(np.abs(x0_plot_nn - x1_plot_nn)**2, axis = -1))



# Set plotting parameters
params = {"text.usetex": True,
      "font.family": "serif",               # Match LaTeX default serif font (Computer Modern)
      "font.serif": ["Computer Modern Roman"],
      "text.latex.preamble": r"\usepackage{amsmath}",
    'legend.fontsize': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    "ytick.left": False,
    'lines.linewidth': 1.
}
matplotlib.rcParams.update(params)

# Plot

plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.scatter(x[:,0].detach().numpy(), x[:,1].detach().numpy(), c= "C0", label='source', marker="+")
plt.scatter(y[:,0].detach().numpy(), y[:,1].detach().numpy(), c = 'C1', label='target')
ot.plot.plot2D_samples_mat(x.detach().numpy(), y.detach().numpy(), plan, linestyle="solid", alpha=.2,c="grey")
plt.legend()

plt.xlim(-11, 10)
plt.ylim(-7, 9)
plt.xticks([])
plt.yticks([])

plt.title("OT (exact), $\\langle C_{\\mu\\nu}, \\pi^\\star_\\text{OT}\\rangle$=" + f"{cost:.2f}")

plt.subplot(132)
col1 = np.arange(1,x0_plot_swgg.shape[0]+1)/x0_plot_swgg.shape[0]
plt.scatter(x0_plot_swgg[:, 0], x0_plot_swgg[:, 1], c=col1, cmap="winter", marker="+")
plt.scatter(x1_plot_swgg[:, 0], x1_plot_swgg[:, 1], c=col1, cmap="winter")
plt.xticks([])
plt.yticks([])
plt.xlim(-11, 10)
plt.ylim(-7, 9)

plt.axline(torch.tensor([0,0]), theta, color='k')

for x0i, x1i in zip(x0_plot_swgg, x1_plot_swgg):
    plt.plot([x0i[0], x1i[0]],
             [x0i[1], x1i[1]],
            linestyle="solid", alpha=.2,c="grey")
plt.title("min-SWGG, $\\langle C_{\\mu\\nu}, \\pi^\\theta\\rangle$=" + f"{cost_swgg:.2f}")

plt.subplot(133)
col1 = np.arange(1, x0_plot_nn.shape[0] + 1) / x0_plot_nn.shape[0]

plt.scatter(x0_plot_nn[:, 0], x0_plot_nn[:, 1], c=col1, cmap="winter", marker="+")
plt.scatter(x1_plot_nn[:, 0], x1_plot_nn[:, 1], c=col1, cmap="winter")
plt.xlim(-11, 10)
plt.ylim(-7, 9)
plt.xticks([])
plt.yticks([])
for x0i, x1i, col in zip(x0_plot_nn, x1_plot_nn, col1):
    plt.plot([x0i[0], x1i[0]],
             [x0i[1], x1i[1]],
            linestyle="solid", alpha=.2,c="grey")
plt.title("DGSWP (NN), $\\langle C_{\\mu\\nu}, \\pi^\\theta\\rangle$=" + f"{cost_nn:.2f}")
plt.tight_layout()
plt.savefig("fig_illus.pdf")