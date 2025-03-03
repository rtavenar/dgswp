import torch
from torch import nn
import numpy as np
import sys
import re
import matplotlib.pyplot as plt

from datasets import data_gen_torch
from gradient_flows import (GeneralizedSWGGGradientFlow, 
                            SlicedWassersteinGradientFlow, 
                            UnOptimizedSWGGGradientFlow,
                            SWGGGradientFlow)

class SingleHiddenLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleHiddenLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # He init.
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SingleLayerInjectiveNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleHiddenLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # He init.
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = torch.cat((x, h), dim=-1)
        return h

n = 50
d = 2

n_iter = 2000
n_directions = 20

n_repeat = 10
if len(sys.argv) > 1 and sys.argv[1] in ["swiss_roll", "circle", "gaussian", "two_moons"]:
    dataset_name = sys.argv[1]
elif re.match(r'gaussian_\d+d', sys.argv[1]):
    dataset_name = "gaussian"
    d = int(re.search(r'\d+', sys.argv[1]).group())
else:
    dataset_name = "swiss_roll"
models = {
    "SW": SlicedWassersteinGradientFlow(learning_rate_flow=.01,
                                          n_iter_flow=n_iter,
                                          n_directions=n_directions),
    "SWGG (no optim.)": UnOptimizedSWGGGradientFlow(learning_rate_flow=.1,
                                          n_iter_flow=n_iter,
                                          n_directions=n_directions),
    "SWGG (optim.)": SWGGGradientFlow(learning_rate_flow=.01,
                                          n_iter_flow=n_iter,
                                          n_iter_inner=n_directions,
                                          learning_rate_inner=.01),
    "Linear Generalized-SWGG": GeneralizedSWGGGradientFlow(learning_rate_flow=.01,
                                          n_iter_flow=n_iter,
                                          model=nn.Linear(in_features=d, out_features=1),
                                          n_iter_inner=n_directions),
    "1-hidden-layer Generalized-SWGG": GeneralizedSWGGGradientFlow(learning_rate_flow=.01,
                                          n_iter_flow=n_iter,
                                          model=SingleHiddenLayerNet(input_size=d, hidden_size=256, output_size=1),
                                          n_iter_inner=n_directions),
}
losses_wasserstein = {
    name: np.zeros((n_repeat, n_iter)) for name in models.keys()
}

for i_repeat in range(n_repeat):
    source, target = data_gen_torch(n_samples_per_distrib=n, d=d,
                                    name=dataset_name, random_state=i_repeat)
    torch.manual_seed(0)
    for name, model in models.items():
        losses_wasserstein[name][i_repeat] = model.fit(source=source, target=target)[-1]

plt.figure()
for name in models.keys():
    p = plt.plot(np.arange(1, n_iter + 1), 
                 np.log10(np.mean(losses_wasserstein[name], axis=0)), 
                 label=name)
    color = p[0].get_color()
    plt.fill_between(np.arange(1, n_iter + 1),
                     np.log10(np.mean(losses_wasserstein[name], axis=0) - np.std(losses_wasserstein[name], axis=0)),
                     np.log10(np.mean(losses_wasserstein[name], axis=0) + np.std(losses_wasserstein[name], axis=0)),
                     alpha=.1
                     )
plt.legend()
plt.ylabel("$\log_{10}(W_2)$")
plt.xlabel("Iterations")
plt.xlim(1, n_iter)
plt.title(sys.argv[1].replace("_", " ").capitalize())
plt.savefig(f"gf_{sys.argv[1]}.pdf")