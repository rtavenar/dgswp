import os
import time

import torch
from torchdyn.core import NeuralODE

from cfm import (ExactOptimalTransportConditionalFlowMatcher, 
                 ConditionalFlowMatcher, 
                 SWGGConditionalFlowMatcher)
from cfm_utils import sample_8gaussians, sample_moons, plot_trajectories, torch_wrapper

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

sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = SWGGConditionalFlowMatcher(sigma=sigma)

ratio = 100

start = time.time()
for k in range(20000 // ratio):
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
        start = end
    #     node = NeuralODE(
    #         torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    #     )
    #     with torch.no_grad():
    #         traj = node.trajectory(
    #             sample_8gaussians(1024),
    #             t_span=torch.linspace(0, 1, 100),
    #         )
    #         plot_trajectories(traj.cpu().numpy())
torch.save(model, f"{savedir}/swggcfm_v1.pt")