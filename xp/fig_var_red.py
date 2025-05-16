import torch
import torch.nn as nn
from torchdyn.datasets import generate_moons
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
import ot

from dgswp import dgswp


def wass(x, y):
    dists = ot.utils.dist(x, y)
    a = torch.ones(x.shape[0])/x.shape[0]
    b = torch.ones(y.shape[0])/y.shape[0]
    
    return ot.emd2(a, b, dists)

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
    #multi = torch.multinomial(torch.ones(8), n, replacement=True)
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
    def __init__(self, input_dim, output_dim=1):
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
n_iter = 1_000

epsilon_Stein = 5e-2
n_samples_Stein = 20

# Load data
torch.manual_seed(2025)
np.random.seed(0)
x = sample_8gaussians(n_samples)
y = sample_moons(n_samples)
y[:,0] -= 2
y[:,1] *= 1.1
y[:,1] += 1

losses_all = {
    "no_vr": {},  # variance_reduction = False
    "vr": {}      # variance_reduction = True
}

models_all = {
    "no_vr": {},  # variance_reduction = False
    "vr": {}      # variance_reduction = True
}

for seed in tqdm(range(10)):
    for key, use_vr in zip(["no_vr", "vr"], [False, True]):
        torch.manual_seed(seed)
        # Re-initialize model and optimizer for each run
        model = MLP(input_dim=dim)
        if use_vr:
            optimizer = torch.optim.SGD(model.parameters(), lr=.2)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=.0002)  # Does not support a too large LR

        losses_all[key][seed] = dgswp(
            x, y, model, optimizer, n_iter=n_iter,
            roll_back=True, variance_reduction=use_vr,
            n_samples_Stein=n_samples_Stein, epsilon_Stein=epsilon_Stein
        )
        models_all[key][seed] = model
        

# Prepare data for plotting
results = {}
min_length = min(1_000, n_iter)
for key in ["no_vr", "vr"]:
    loss_matrix = np.array([losses[:min_length] for losses in losses_all[key].values()])
    mean_loss = np.mean(loss_matrix, axis=0)
    std_loss = np.std(loss_matrix, axis=0)
    results[key] = {"mean": mean_loss, "std": std_loss, "iters": np.arange(min_length)}



# Set plotting parameters
params = {"text.usetex": True,
      "font.family": "serif",               # Match LaTeX default serif font (Computer Modern)
      "font.serif": ["Computer Modern Roman"],
      "text.latex.preamble": r"\usepackage{amsmath}",
    'legend.fontsize': 24,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    "ytick.left": False,
    'lines.linewidth': 4.
}
matplotlib.rcParams.update(params)

# Plot
plt.figure(figsize=(8, 5))

# Plot no variance reduction
plt.plot(results["no_vr"]["iters"], results["no_vr"]["mean"], label="Without Variance Reduction", color="blue")
plt.fill_between(
    results["no_vr"]["iters"],
    results["no_vr"]["mean"] - results["no_vr"]["std"],
    results["no_vr"]["mean"] + results["no_vr"]["std"],
    color="blue", alpha=0.3
)

# Plot with variance reduction
plt.plot(results["vr"]["iters"], results["vr"]["mean"], label="With Variance Reduction", color="green")
plt.fill_between(
    results["vr"]["iters"],
    results["vr"]["mean"] - results["vr"]["std"],
    results["vr"]["mean"] + results["vr"]["std"],
    color="green", alpha=0.3
)

plt.xlabel("Iteration")
plt.ylabel("$\\langle C_{\\mu\\nu}, \\pi^\\theta\\rangle$")
plt.legend()
plt.xlim([0, min_length])
plt.grid(True)
plt.tight_layout()
plt.savefig("fig/fig_var_red.pdf")
