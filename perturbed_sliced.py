import numpy as np

import torch
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal

from partial import partial_ot_1d

_GUMBEL = 'gumbel'
_NORMAL = 'normal'
SUPPORTED_NOISES = (_GUMBEL, _NORMAL)

def sample_noise_with_gradients(noise, shape):
    """Samples a noise tensor according to a distribution with its gradient.

    Args:
    noise: (str) a type of supported noise distribution.
    shape: torch.tensor<int>, the shape of the tensor to sample.

    Returns:
    A tuple Tensor<float>[shape], Tensor<float>[shape] that corresponds to the
    sampled noise and the gradient of log the underlying probability
    distribution function. For instance, for a gaussian noise (normal), the
    gradient is equal to the noise itself.

    Raises:
    ValueError in case the requested noise distribution is not supported.
    See perturbations.SUPPORTED_NOISES for the list of supported distributions.
    """
    if noise not in SUPPORTED_NOISES:
        raise ValueError('{} noise is not supported. Use one of [{}]'.format(
            noise, SUPPORTED_NOISES))

    if noise == _GUMBEL:
        sampler = Gumbel(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = 1 - torch.exp(-samples)
    elif noise == _NORMAL:
        sampler = Normal(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = samples

    return samples, gradients

def project_in_1d(x, y, w):
    # w /= torch.sqrt(torch.sum(w ** 2, axis=-1, keepdims=True))
    proj_x = (w @ x.T).T
    proj_y = (w @ y.T).T
    return proj_x, proj_y

def compute_cost(x, y, w, max_iter_partial):
    proj_x, proj_y = project_in_1d(x, y, w)
    ind_x, ind_y, marginal_costs = partial_ot_1d(proj_x, proj_y, max_iter=max_iter_partial, p=2)

    sorted_ind_x = np.argsort(proj_x[ind_x])
    sorted_ind_y = np.argsort(proj_y[ind_y])
    subset_x = x[ind_x][sorted_ind_x]
    subset_y = y[ind_y][sorted_ind_y]
    cost = np.sum(np.abs(subset_x - subset_y))
    return cost / ind_x.shape[0]


def swgg_score_theta(thetas, x, y, max_iter_partial):
    perturbed_costs = [
        compute_cost(x, y, theta, max_iter_partial) 
        for theta in thetas
    ]
    return torch.tensor(perturbed_costs)

class SWGGStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, x, y, max_iter_partial, n_samples, noise_type, epsilon, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        input_shape = theta.shape  # [D, ]
        perturbed_input_shape = [n_samples] + list(input_shape)
        additive_noise, noise_gradient = sample_noise_with_gradients(noise_type, perturbed_input_shape)
        additive_noise = additive_noise.to(device)
        noise_gradient = noise_gradient.to(device)
        perturbed_input = theta + epsilon * additive_noise  # [N, D]

        # [...]
        # perturbed_output is [n_samples, ]
        perturbed_output = swgg_score_theta(perturbed_input, x, y, max_iter_partial)

        ctx.save_for_backward(perturbed_output, noise_gradient, torch.tensor(epsilon))
        return torch.mean(perturbed_output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        perturbed_output, noise_gradient, epsilon = ctx.saved_tensors
        grad_theta = grad_output * perturbed_output.unsqueeze(1) * noise_gradient / epsilon
        return torch.mean(grad_theta, dim=0), None, None, None, None, None, None, None

swgg_step = SWGGStep.apply


class PerturbedPartialSWGG:
    def __init__(self, max_iter_gradient, max_iter_partial=None, opt_lambda_fun=None, 
                 perturbation_n_samples=1000, perturbation_sigma=1., perturbation_noise="normal", device=None) -> None:
        self.max_iter_gradient = max_iter_gradient
        self.max_iter_partial = max_iter_partial
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.opt_lambda_fun = opt_lambda_fun
        self.perturbation_n_samples = perturbation_n_samples
        self.perturbation_sigma = perturbation_sigma
        self.perturbation_noise = perturbation_noise

    def draw_direction(self, d):
        w = np.random.randn(d)
        w /= np.sqrt(np.sum(w ** 2, axis=-1, keepdims=True))
        return w

    def project_in_1d(self, x, y, w):
        w /= np.linalg.norm(w)
        proj_x = np.dot(x, w)
        proj_y = np.dot(y, w)
        return proj_x, proj_y

    def fit(self, x, y):
        assert x.shape[:2] == y.shape[:2]
        n, d = x.shape[:2]

        min_cost = np.inf
        bool_indices_x, bool_indices_y = None, None
        list_w = []
        list_w_grad = []
        w = torch.tensor(self.draw_direction(d), requires_grad=True)
        if self.opt_lambda_fun is not None:
            opt = self.opt_lambda_fun([w])
        else:
            opt = torch.optim.SGD([w], lr=.01, momentum=0.9)
        list_costs = []
        for iter in range(self.max_iter_gradient):
            list_w.append(w.detach().numpy().copy())
            opt.zero_grad()
            output = swgg_step(w, x, y, 
                               self.max_iter_partial, 
                               self.perturbation_n_samples, 
                               self.perturbation_noise, 
                               self.perturbation_sigma, 
                               self.device)
            output.backward()
            list_w_grad.append(w.grad.detach().numpy().copy())
            opt.step()
            with torch.no_grad():
                w /= torch.sqrt(torch.sum(w ** 2, axis=-1, keepdims=True))

            cost = float(output)
            print(f"Iter. {iter}, OT cost={cost} ", end="")
            list_costs.append(cost)

            if cost < min_cost:
                best_w = w.detach().numpy().copy()
                min_cost = cost
                print("(best iter so far)")
            else:
                print()
        
        proj_x, proj_y = self.project_in_1d(x, y, torch.Tensor(best_w))
        ind_x, ind_y, _ = partial_ot_1d(proj_x, proj_y, max_iter=self.max_iter_partial, p=2)
        
        bool_indices_x = np.array([(i in ind_x) for i in range(n)], dtype=bool)
        bool_indices_y = np.array([(i in ind_y) for i in range(n)], dtype=bool)

        self.list_w_ = list_w
        self.list_w_grad_ = list_w_grad
        
        return min_cost, bool_indices_x, bool_indices_y, list_costs, best_w



if __name__ == "__main__":
    import ot
    import matplotlib.pyplot as plt
    n = 1000
    d = 20
    n_outliers = 1
    np.random.seed(0)
    torch.manual_seed(0)

    x = np.random.randn(n, d) * 3
    y = np.random.randn(n, d)

    x[:, 1] /= 10
    y[:, 1] /= 10

    M = ot.dist(x, y, metric="minkowski", p=1)
    cost = ot.partial.partial_wasserstein2([], [], M, m=(n - n_outliers) / n)
    print(f"Exact cost: {cost}")
    
    sliced = PerturbedPartialSWGG(max_iter_gradient=100, 
                                           max_iter_partial=n-n_outliers, 
                                        #    opt_lambda_fun=lambda param: torch.optim.SGD(param, lr=1e-1, momentum=.8),
                                           opt_lambda_fun=lambda param: torch.optim.AdamW(param, lr=1e-1),
                                           perturbation_n_samples=100,
                                           perturbation_sigma=.1,
                                           perturbation_noise="normal")
    _, bool_ind_x, bool_ind_y, list_costs, w = sliced.fit(x, y)

    plt.plot(list_costs)
    plt.title(f"Partial OT cost is: {cost:.3f}")
    plt.show()