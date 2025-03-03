from sklearn.datasets import make_circles, make_spd_matrix, make_swiss_roll, make_moons
from sklearn.utils import check_random_state
import numpy as np
import torch


def data_gen(n_samples_per_distrib, d, name, random_state=0):
    rng = check_random_state(seed=random_state)
    if name == "circle":
        assert d == 2
        noisy_circles, y = make_circles(n_samples=n_samples_per_distrib*2, 
                                        factor=.5, noise=0.075, 
                                        random_state=rng)
        target = noisy_circles[y == 0]
        target = (target - target.mean(axis=0, keepdims=True)) / target.std(axis=0, keepdims=True)
        source = 2.5 * (rng.rand(n_samples_per_distrib, d) - .5) + 1.
    elif name == "swiss_roll":
        assert d == 2
        target = make_swiss_roll(n_samples=n_samples_per_distrib, random_state=rng)[0][:, (0, 2)]
        target = (target - target.mean(axis=0, keepdims=True)) / target.std(axis=0, keepdims=True)
        source = 2.5 * (rng.rand(n_samples_per_distrib, d) - .5) + 1.
    elif name == "two_moons":
        assert d == 2
        target = make_moons(n_samples=n_samples_per_distrib, random_state=rng)[0]
        target = (target - target.mean(axis=0, keepdims=True)) / target.std(axis=0, keepdims=True)
        source = 2.5 * (rng.rand(n_samples_per_distrib, d) - .5) + 1.
    elif name == "gaussian":
        mu_s = rng.randn(d)
        cov_s = make_spd_matrix(d, random_state=rng)
        target = rng.multivariate_normal(mu_s, cov_s, n_samples_per_distrib)
        source = 2.5 * (rng.rand(n_samples_per_distrib, d) - .5) + 1.
    return source, target


def data_gen_torch(n_samples_per_distrib, d, name, random_state=0):
    source_npy, target_npy = data_gen(n_samples_per_distrib, d, name, random_state)

    source = torch.clone(torch.tensor(source_npy, dtype=torch.float32))
    source.requires_grad_()
    target = torch.tensor(target_npy, dtype=torch.float32, requires_grad=False)
    return source, target