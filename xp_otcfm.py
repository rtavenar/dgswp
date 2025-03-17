import os
import time
import json
import sys
import torch
import numpy as np

from cfm import ExactOptimalTransportConditionalFlowMatcher
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
    seed = int(sys.argv[1])
except:
    sys.stderr.write(f"{sys.argv[0]} seed\n")
    sys.exit(-1)

torch.manual_seed(seed)
np.random.seed(seed)

sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

epochs = 20000

start = time.time()
for k in range(epochs):
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size)
    x1 = sample_moons(batch_size)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 5000 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        torch.save(model, f"{savedir}/otcfm_e{k+1}_seed{seed}.pt")

end = time.time()
info = {
    "name": "OT-CFM",
    "epochs": epochs,
    "seed": seed,
    "model": f"otcfm_e{epochs}_seed{seed}.pt",
    "time": (end - start)
}
json.dump(info, open(f"{savedir}/otcfm_e{epochs}_seed{seed}.json", "w"))
