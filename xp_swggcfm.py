import os
import time
import json
import sys
import torch
import numpy as np

from cfm import SWGGConditionalFlowMatcher
from cfm_utils import sample_8gaussians, sample_moons


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

savedir = "models_cfm/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)

try:
    ratio = int(sys.argv[1])
    seed = int(sys.argv[2])
except:
    sys.stderr.write(f"{sys.argv[0]} ratio seed\n")
    sys.exit(-1)

torch.manual_seed(seed)
np.random.seed(seed)

sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = SWGGConditionalFlowMatcher(sigma=sigma)

epochs = 20000

start = time.time()
for k in range(epochs // ratio):
    x0 = sample_8gaussians(batch_size * ratio)
    x1 = sample_moons(batch_size * ratio)
    FM.precompute_map(x0, x1)

    for i_gradient_steps in range(ratio):
        optimizer.zero_grad()

        t, xt, ut = FM.sample_location_and_conditional_flow_from_indices(
            slice(i_gradient_steps * batch_size, (i_gradient_steps + 1) * batch_size)
        )

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

    if (k + 1) % (5000 // ratio) == 0:
        end = time.time()
        print(f"{k+1} (equivalent to {(k+1)*ratio}): loss {loss.item():0.3f} time {(end - start):0.2f}")
        torch.save(model, f"{savedir}/swggcfm_k{ratio}_e{(k+1)*ratio}_seed{seed}.pt")

end = time.time()
info = {
    "name": "SWGG-CFM",
    "epochs": epochs,
    "k": ratio,
    "seed": seed,
    "model": f"swggcfm_k{ratio}_e{epochs}_seed{seed}.pt",
    "time": (end - start)
}
json.dump(info, open(f"{savedir}/swggcfm_k{ratio}_e{epochs}_seed{seed}.json", "w"))
