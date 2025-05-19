import torch
from torch import nn
import numpy as np
import sys
import re
import csv
from tqdm import tqdm

from dgswp import data_gen_torch
from dgswp import (DifferentiableGeneralizedWassersteinPlanGradientFlow, 
                   SlicedWassersteinGradientFlow, 
                   RandomSearchSWGGGradientFlow,
                   SWGGGradientFlow,
                   AugmentedSlicedWassersteinGradientFlow,
                   MaxSlicedWassersteinGradientFlow)

class SingleHiddenLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleHiddenLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.init()

    def init(self):
        # He init.
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SingleLayerInjectiveNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleLayerInjectiveNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init()

    def init(self):
        self.fc1 = nn.Sequential(nn.Linear(self.input_size, self.hidden_size, dtype=torch.float32))

    def forward(self, x):
        h = self.fc1(x)
        h = torch.cat((x, h), dim=-1)
        return h

n = 50
d = 2

n_iter = 2000
learning_rate_flow = .01
n_directions = 20

n_repeat = 10
if len(sys.argv) > 1 and sys.argv[1] in ["swiss_roll", "gaussian", "two_moons"]:
    dataset_name = sys.argv[1]
elif len(sys.argv) > 1 and re.match(r'gaussian_\d+d', sys.argv[1]):
    dataset_name = "gaussian"
    d = int(re.search(r'\d+', sys.argv[1]).group())
else:
    dataset_name = "swiss_roll"
models = {
    "SWD": SlicedWassersteinGradientFlow(learning_rate_flow=learning_rate_flow,
                                          n_iter_flow=n_iter,
                                          n_directions=n_directions),
    "Max-SW": MaxSlicedWassersteinGradientFlow(learning_rate_flow=learning_rate_flow,
                                          n_iter_flow=n_iter,
                                          d=d,
                                          n_iter_inner=n_directions,
                                          learning_rate_inner=.01),
    "min-SWGG (random search)": RandomSearchSWGGGradientFlow(learning_rate_flow=learning_rate_flow,
                                          n_iter_flow=n_iter,
                                          n_directions=n_directions),
    "min-SWGG (optim)": SWGGGradientFlow(learning_rate_flow=learning_rate_flow,
                                          n_iter_flow=n_iter,
                                          d=d,
                                          epsilon=0.1,
                                          n_iter_inner=n_directions,
                                          learning_rate_inner=.01),
    "ASWD": AugmentedSlicedWassersteinGradientFlow(learning_rate_flow=learning_rate_flow,
                                          n_iter_flow=n_iter,
                                          model=SingleLayerInjectiveNet(input_size=d, hidden_size=d),
                                          n_iter_inner=10,
                                          n_directions=n_directions,
                                          lambda_=.1,
                                          learning_rate_inner=.01),
    "DGSWP (linear)": DifferentiableGeneralizedWassersteinPlanGradientFlow(learning_rate_flow=learning_rate_flow,
                                          n_iter_flow=n_iter,
                                          model=nn.Linear(in_features=d, out_features=1),
                                          n_iter_inner=n_directions,
                                          learning_rate_inner=.01),
    "DGSWP (NN)": DifferentiableGeneralizedWassersteinPlanGradientFlow(learning_rate_flow=learning_rate_flow,
                                          n_iter_flow=n_iter,
                                          model=SingleHiddenLayerNet(input_size=d, hidden_size=256, output_size=1),
                                          n_iter_inner=n_directions,
                                          learning_rate_inner=.01),
}
losses_wasserstein = {
    name: np.zeros((n_repeat, n_iter)) for name in models.keys()
}

for i_repeat in tqdm(range(n_repeat)):
    source, target = data_gen_torch(n_samples_per_distrib=n, d=d,
                                    name=dataset_name, random_state=i_repeat)
    torch.manual_seed(0)
    for name, model in models.items():
        model.init()
        losses_wasserstein[name][i_repeat] = model.fit(source=source, target=target)[-1]

dataset_name_with_dim = dataset_name
if dataset_name == "gaussian":
    dataset_name_with_dim += f"_{d}d"
with open(f'results_gf/gf_{dataset_name_with_dim}.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['method', 'iteration', 'run_id', 'log10_W2'])

    for method, loss_matrix in losses_wasserstein.items():
        # loss_matrix shape is (n_repeat, n_iter)
        for run_id in range(n_repeat):
            for iter in range(n_iter):
                value = loss_matrix[run_id, iter]
                log_value = np.log10(value)
                writer.writerow([method, iter, run_id + 1, log_value])
