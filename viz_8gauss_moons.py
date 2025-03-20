import numpy as np
import torch
import matplotlib.pyplot as plt

from cfm_utils import sample_8gaussians, sample_moons
from losses import swgg_opt

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

x0 = sample_8gaussians(1024)
x1 = sample_moons(1024)
x0_val = sample_8gaussians(256)
x1_val = sample_moons(256)

n_iter=2000
epsilon_Stein=5e-2
n_samples_Stein=10
roll_back=True

model = torch.nn.Sequential(  # TODO
            torch.nn.Linear(in_features=2, out_features=256),
            torch.nn.SELU(),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.SELU(),
            # torch.nn.Linear(in_features=256, out_features=256),
            # torch.nn.SELU(),
            torch.nn.Linear(in_features=128, out_features=1)
        )
torch.nn.init.kaiming_normal_(model[0].weight, mode='fan_in', nonlinearity='selu')
torch.nn.init.kaiming_normal_(model[2].weight, mode='fan_in', nonlinearity='selu')
# torch.nn.init.kaiming_normal_(model[4].weight, mode='fan_in', nonlinearity='selu')
torch.nn.init.kaiming_normal_(model[-1].weight, mode='fan_in', nonlinearity='linear')
# torch.nn.init.zeros_(model[0].bias)
# torch.nn.init.zeros_(model[2].bias)
# torch.nn.init.zeros_(model[-1].bias)
opt_model = torch.optim.Adam(model.parameters(), lr=.05)

swgg_opt(x0, x1, model, opt_model, 
         n_iter=n_iter, epsilon_Stein=epsilon_Stein, n_samples_Stein=n_samples_Stein,
         roll_back=roll_back, log=True)

indices0 = torch.argsort(model(x0_val).flatten())
indices1 = torch.argsort(model(x1_val).flatten())

x0_plot = x0_val[indices0]
x1_plot = x1_val[indices1]
plt.scatter(x0_plot[:, 0], x0_plot[:, 1])
plt.scatter(x1_plot[:, 0], x1_plot[:, 1])
for x0i, x1i in zip(x0_plot, x1_plot):
    plt.plot([x0i[0], x1i[0]],
             [x0i[1], x1i[1]],
             color='k', linestyle="solid", alpha=.5)
plt.show()