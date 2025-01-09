import numpy as np

import torch
from torch.distributions.normal import Normal

def sample_noise_with_gradients(n_samples, shape):
    """Samples a noise tensor from N(0,1) with its gradient.

    Args:
    n_samples: int, the number of samples to be drawn.
    shape: torch.tensor<int>, the tensor shape of each sample.

    Returns:
    A tuple Tensor<float>[[n_samples, *shape]], Tensor<float>[[n_samples, *shape]] 
    that corresponds to the
    sampled noise and the gradient of log the underlying probability
    distribution function. In practice, for the N(0,1) distribution, the
    gradient is equal to the noise itself.
    """
    actual_shape = [n_samples] + list(shape)
    sampler = Normal(0.0, 1.0)
    samples = sampler.sample(actual_shape)
    first_sample = torch.zeros(shape)
    all_samples = torch.cat((first_sample.unsqueeze(0), samples), dim=0)
    gradients = all_samples

    return all_samples, gradients

def F(x, y, theta):
    pos_x_1d = np.argsort(theta @ x.T)
    pos_y_1d = np.argsort(theta @ y.T)
    return torch.mean(torch.sum((x[pos_x_1d] - y[pos_y_1d]) ** 2, dim=-1), dim=0)


def F_batch(thetas, x, y):
    perturbed_costs = [F(x, y, theta) for theta in thetas]
    return torch.tensor(perturbed_costs)

class F_epsilon(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, x, y, n_samples, epsilon, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        input_shape = theta.shape  # [D, ]
        additive_noise, noise_gradient = sample_noise_with_gradients(n_samples, input_shape)
        additive_noise = additive_noise.to(device)
        noise_gradient = noise_gradient.to(device)  # [N+1, D]
        perturbed_input = theta + epsilon * additive_noise  # [N+1, D]

        # [...]
        # perturbed_output is [N+1, ]
        perturbed_output = F_batch(perturbed_input, x, y)

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
        F0 = perturbed_output[0]
        grad_theta = grad_output * (perturbed_output[1:] - F0).unsqueeze(1) * noise_gradient[1:] / epsilon
        return torch.mean(grad_theta, dim=0), None, None, None, None, None



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_sparse_spd_matrix
    torch.manual_seed(0)
    np.random.seed(0)

    def draw_samples(n, d):
        cov_x = make_sparse_spd_matrix(dim=d, random_state=42,alpha=0.99, smallest_coef=0, largest_coef=0.22)
        diag_x = np.eye(d)
        for i in range(d//2):
            diag_x[i,i] = 20
        cov_x = cov_x + diag_x
        #cov_x[1,1] = 20
        mu_x = np.random.randn(d)
        x = np.random.multivariate_normal(mu_x, cov_x, size=n)
        cov_y = np.eye(d)
        cov_y = make_sparse_spd_matrix(dim=d, random_state=43)
        mu_y =  np.random.randn(d) 
        mu_y[0:d//2] =  mu_y[0:d//2] + 20
        #mu_y[0] =  mu_y[0] + 10 #1dim is the most important
        y = np.random.multivariate_normal(mu_y, cov_y, size=n)  
        return torch.tensor(x).float(), torch.tensor(y).float()

    n = 100
    epsilon = .1
    n_samples = 10
    d = 3
    x, y = draw_samples(n, d)
    learning_rate = 0.005
    n_iter = 100
    n_reps = 3

    losses = {}
    for id_rep in range(n_reps):
        losses[id_rep] = []
        thetas = [torch.randn(d, requires_grad=True)]
        with torch.no_grad():
            thetas[0] /= torch.norm(thetas[0])
        for i in range(n_iter):
            loss = F_epsilon.apply(thetas[-1], x, y, n_samples, epsilon, "cpu")
            loss.backward()
            with torch.no_grad():
                losses[id_rep].append(loss.item())
                theta_t = thetas[-1] - learning_rate * thetas[-1].grad
                theta_t /= torch.norm(theta_t)
                theta_t.requires_grad_()
                thetas.append(theta_t)
    
    for l in losses.values():
        plt.plot(l, color='b')
    plt.legend(["$F_{\\varepsilon}$"])
    plt.title("Et tout ça en torch")
    plt.show()