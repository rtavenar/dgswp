import time
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

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

n_iter = 100
learning_rate_flow = .01
n_directions = 10
n = 50

d_values = [2, 5, 10, 50, 100]
n_repeat = 5

timings = {}
for name in {"SW", "SWGG (no optim.)", 
             "SWGG (optim.)", "Linear Generalized-SWGG", 
             "1-hidden-layer Generalized-SWGG"}:
    timings[name] = []

torch.manual_seed(0)
for d in d_values:
    models = {
        "SW": SlicedWassersteinGradientFlow(learning_rate_flow=learning_rate_flow,
                                            n_iter_flow=n_iter,
                                            n_directions=n_directions),
        "SWGG (no optim.)": UnOptimizedSWGGGradientFlow(learning_rate_flow=learning_rate_flow,
                                            n_iter_flow=n_iter,
                                            n_directions=n_directions),
        "SWGG (optim.)": SWGGGradientFlow(learning_rate_flow=learning_rate_flow,
                                            n_iter_flow=n_iter,
                                            n_iter_inner=n_directions,
                                            learning_rate_inner=.01),
        "Linear Generalized-SWGG": GeneralizedSWGGGradientFlow(learning_rate_flow=learning_rate_flow,
                                            n_iter_flow=n_iter,
                                            model=nn.Linear(in_features=d, out_features=1),
                                            n_iter_inner=n_directions),
        "1-hidden-layer Generalized-SWGG": GeneralizedSWGGGradientFlow(learning_rate_flow=learning_rate_flow,
                                            n_iter_flow=n_iter,
                                            model=SingleHiddenLayerNet(input_size=d, hidden_size=256, output_size=1),
                                            n_iter_inner=n_directions),
    }
    model_timings = {key: [] for key in models.keys()}
    for _ in range(n_repeat):
        source = torch.rand(n, d)
        target = torch.rand(n, d)
        for name, model in models.items():
            start_time = time.time()
            model.fit(source, target)
            end_time = time.time()

            model_timings[name].append(end_time - start_time)

    for name in models.keys():
        timings[name].append(np.mean(model_timings[name]))

plt.figure(figsize=(10, 6))
for name in models.keys():
    plt.loglog(d_values, timings[name], label=name)
plt.xlabel('$d$')
plt.ylabel('Running time (s)')
plt.legend()
plt.grid()
plt.savefig("timings_gf_d.pdf")
