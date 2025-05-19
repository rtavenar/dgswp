# Differentiable Generalized Sliced Wasserstein Plans

This repository accompanies our paper: **"Differentiable Generalized Sliced Wasserstein Plans"** (anonymous submission).

It contains the code necessary to reproduce the figures presented in the paper. Each figure is generated using a script following the `fig_*.py` naming convention.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The CFM experiment imply additional requirements: see `torchcfm`'s GitHub repository for that.

## Example code

The main function in this code is the `dgswp` one from the `dgswp` package.
It runs our algorithm by taking as input two datasets `x` and `y`, a projection model to be fitted and its optimizer (hyper-parameters can be passed to `dgswp`, but default values work pretty well).
At the end of execution, the cost $\langle C, \pi(\theta)\rangle$ can be computed using the `H_module` function from the `dgswp` package.

```python
import torch.nn as nn
from torch.optim import SGD

from dgswp import dgswp, H_module, data_gen_torch

x, y = data_gen_torch(n_samples_per_distrib=50, d=2, name="two_moons")
model = nn.Linear(in_features=2, out_features=1)  # Use any `nn.Sequential` model here
opt = SGD(model.parameters())

dgswp(x, y, model, opt)
print(H_module(x, y, model))  # Outputs $\langle C, \pi(\theta)\rangle$ 
                              # where $\theta$ is the set of 
                              # all model parameters
```

## Reproducing the paper's results

To reproduce the figures from the paper, run the corresponding Python scripts. Each script generates one figure:

| Script        | Description                  |
|---------------|------------------------------|
| `xp/fig_illus.py`    | Figure 1 |
| `xp/fig_h_g.py`    | Figure 2 |
| `xp/fig_var_red.py`    | Figure 3 |
| `xp/xp_gradient_flows.py` | Generates result files in `results_gf/` (see below for details) |
| `xp/fig_gf.py` | Figure 4 from results stored in `results_gf/` |
| `xp/xp_gradient_flows_hyp.py` | Generates result files in `results_hyp/` (see below for details) |
| `xp/fig_gf_hyp.py` | Figure 5 from results stored in `results_hyp/` |
| `xp/train_cifar10_cfm.py` | Trains baseline models for the CIFAR10 CFM experiment |
| `xp/train_cifar10_dgswpcfm.py` | Trains our model for the CIFAR10 CFM experiment |
| `xp/compute_fid.py` | Computes FID score and NFE for a trained model |
| `xp/fig_cfm.py` | Figure 6 from results stored in `results_cfm/fid_results_with_runs.csv` |

Make sure that the base folder is in your PYTHONPATH.
One way to do so is to preprend all your python commands with `PYTHONPATH=.` as in:

```bash
PYTHONPATH=. python xp/fig_illus.py
```

To run the gradient flow experiment on all 4 datasets for Figure 4, you will need:

```bash
PYTHONPATH=. python xp/xp_gradient_flows.py swiss_roll
PYTHONPATH=. python xp/xp_gradient_flows.py two_moons
PYTHONPATH=. python xp/xp_gradient_flows.py gaussian
PYTHONPATH=. python xp/xp_gradient_flows.py gaussian_500d
```

To run the hyperbolic gradient flow experiment on both settings for Figure 5, you will need:

```bash
PYTHONPATH=. python xp/xp_gradient_flow_hyp.py --target center
PYTHONPATH=. python xp/xp_gradient_flow_hyp.py --target border
```

## External software

The `torchcfm` subpackage included here is a slightly modified version of code available at `https://github.com/atong01/conditional-flow-matching` that includes DGSWP as a way to prepare minibatches.

Similarly, the `lib_hyp` subpackage is an adaptation of the `lib` folder from `https://github.com/clbonet/Hyperbolic_Sliced-Wasserstein_via_Geodesic_and_Horospherical_Projections`.
