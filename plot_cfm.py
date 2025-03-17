import os
import json
import torch
from torchdyn.core import NeuralODE
from cfm_utils import sample_8gaussians, sample_moons, torch_wrapper
import ot
import numpy as np
import matplotlib.pyplot as plt


def wass(x, y):
    dists = ot.utils.dist(x, y)
    a = torch.ones(x.shape[0])/x.shape[0]
    b = torch.ones(y.shape[0])/y.shape[0]
    return float(ot.emd2(a, b, dists))


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
x0_val = sample_8gaussians(1024)
x1_val = sample_moons(1024)

n_steps = 1

savedir = "models_cfm/8gaussian-moons"
timings = {}
wass_dists = {}
for fname in os.listdir(savedir):
    if fname.endswith(".json"):
        info = json.load(open(os.path.join(savedir, fname), "r"))
        if info["name"] not in timings.keys():
            timings[info["name"]] = {}
            wass_dists[info["name"]] = {}
        k = info.get("k", -1)

        model = torch.load(os.path.join(savedir, info["model"]), weights_only=False)
        node = NeuralODE(torch_wrapper(model), solver="euler")
        with torch.no_grad():
            traj = node.trajectory(
                x0_val,
                t_span=torch.linspace(0, 1, n_steps + 1),
            )
        timings[info["name"]][k] = timings[info["name"]].get(k, []) + [info["time"]]
        wass_dists[info["name"]][k] = wass_dists[info["name"]].get(k, []) + [wass(traj[-1], x1_val)]

for name in timings.keys():
    sorted_keys = sorted(timings[name].keys())

    x_means = [np.mean(timings[name][k]) for k in sorted_keys]
    y_means = [np.mean(wass_dists[name][k]) for k in sorted_keys]
    x_errors = [np.std(timings[name][k]) for k in sorted_keys]
    y_errors = [np.std(wass_dists[name][k]) for k in sorted_keys]

    plt.errorbar(
        x_means, y_means,
        xerr=x_errors, yerr=y_errors,
        label=name, marker="x", capsize=5
    )
    # for idx_k, k in enumerate(sorted_keys):
    #     if k != -1:
    #         plt.text(x_means[idx_k] + 2, y_means[idx_k] + .02, s=f"$k={k}$")

plt.legend();
plt.grid("on")
plt.xlabel("Timings (s)")
plt.ylabel("$W(m(\\mu), \\nu)$")
plt.title(f"{n_steps}-step Euler integration")
plt.tight_layout()
plt.savefig("cfm.pdf")
