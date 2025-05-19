# Differentiable Generalized Sliced Wasserstein Plans

This repository accompanies our paper: **"Differentiable Generalized Sliced Wasserstein Plans"** (anonymous submission).

It contains the code necessary to reproduce the figures presented in the paper. Each figure is generated using a script following the `fig_*.py` naming convention.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Additional requirements are necessary for the CFM experiment: see `torchcfm`'s GitHub repository for that.

## Figure Generation

To reproduce the figures from the paper, run the corresponding Python scripts. Each script generates one figure:

| Script        | Description                  |
|---------------|------------------------------|
| `xp/fig_illus.py`    | Figure 1 |
| `xp/fig_h_g.py`    | Figure 2 |
| `xp/fig_var_red.py`    | Figure 3 |
| `xp/xp_gradient_flows.py` | Generates result files in `results_gf/` (see below for details) |
| `xp/fig_gf.py` | Figure 4 from results stored in `results_gf/` |
| `xp/xp_gradient_flows_hyp.py` | Generates result files in `results_hyp/` **TODO** |
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
