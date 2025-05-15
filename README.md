# Differentiable Generalized Sliced Wasserstein Plans

This repository accompanies our paper: **"Differentiable Generalized Sliced Wasserstein Plans"** (anonymous submission).

It contains the code necessary to reproduce the figures presented in the paper. Each figure is generated using a script following the `fig_*.py` naming convention.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Figure Generation

To reproduce the figures from the paper, run the corresponding Python scripts. Each script generates one figure:

| Script        | Description                  |
|---------------|------------------------------|
| `fig_illus.py`    | Figure 1 |
| `fig_h_g.py`    | Figure 2 |
| `fig_var_red.py`    | Figure 3 |
| `xp_gradient_flows.py` | Generates result files in `results_gf/` |
| `fig_gf.py` | Figure 4 from results stored in `results_gf/` |
| `xp_gradient_flows_hyp.py` | Generates result files in `results_hyp/` **TODO** |
| `fig_gf_hyp.py` | Figure 5 from results stored in `results_hyp/` |
| `xp_cfm.py` | Generates result files in `results_cfm/` **TODO: will  be a 2-step process** |
| `fig_cfm.py` | Figure 6 from results stored in `results_cfm/` |


To run the gradient flow experiment on all 4 datasets for Figure 4, you will need:

```bash
python xp_gradient_flows.py swiss_roll
python xp_gradient_flows.py two_moons
python xp_gradient_flows.py gaussian
python xp_gradient_flows.py gaussian_500d
```
