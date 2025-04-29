import numpy as np
import copy
import torch
from torch.distributions.normal import Normal

def sample_noise_with_gradients(n_samples, shape, normalize=False):
    """Samples a noise tensor from N(0,1) with its gradient.

    Args:
    n_samples: int, the number of samples to be drawn.
    shape: torch.tensor<int>, the tensor shape of each sample.

    Returns:
    A tuple Tensor<float>[[n_samples+1, *shape]], Tensor<float>[[n_samples+1, *shape]] 
    that corresponds to the
    sampled noise and the gradient of log the underlying probability
    distribution function. In practice, for the N(0,1) distribution, the
    gradient is equal to the noise itself.
    The first element in this tensor is always the zero element that can later be used
    for variance reduction.
    """
    actual_shape = [n_samples] + list(shape)
    sampler = Normal(0.0, 1.0)
    samples = sampler.sample(actual_shape)
    first_sample = torch.zeros(shape)
    if normalize:
        samples[:,:-1] /= torch.norm(samples[:,:-1], dim=-1, keepdim=True)**2

    all_samples = torch.cat((first_sample.unsqueeze(0), samples), dim=0)
    gradients = all_samples

    return all_samples, gradients

def F(x, y, theta, metric='sqeuclidean', p=2,): #implements only sqeuclidean for now
    pos_x_1d = torch.argsort(theta @ x.T)
    pos_y_1d = torch.argsort(theta @ y.T)
    return torch.mean(torch.sum(torch.abs(x[pos_x_1d] - y[pos_y_1d]) ** p, dim=-1), dim=0)

def F_module(x, y, model, metric='sqeuclidean', p=2): #implements only sqeuclidean for now
    pos_x_1d = torch.argsort(model(x).flatten())
    pos_y_1d = torch.argsort(model(y).flatten())
    return torch.mean(torch.sum(torch.abs(x[pos_x_1d] - y[pos_y_1d]) ** p, dim=-1), dim=0)


def F_batch(thetas, x, y, fun=None, metric='sqeuclidean', p=2):
    if fun is None:
        fun = lambda x, y, theta, metric, p: F(x, y, theta, metric, p)
    return torch.tensor([fun(x, y, theta) for theta in thetas], requires_grad=True)

class F_epsilon(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, x, y, fun, n_samples, epsilon, device):
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
        perturbed_output = F_batch(perturbed_input, x, y, fun)  # [N+1, ]

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
        F_minus_F0 = grad_output * (perturbed_output[1:] - F0)
        F_minus_F0_reshaped = F_minus_F0
        for _ in range(len(noise_gradient.shape) - 1):
            F_minus_F0_reshaped = F_minus_F0_reshaped.unsqueeze(1)

        grad_theta = F_minus_F0_reshaped * noise_gradient[1:] / epsilon
        
        return torch.mean(grad_theta, dim=0), None, None, None, None, None, None

class F_epsilon_module(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, x, y, fun, n_samples, epsilon, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        normalize = False
        if str(module) == "BusemannMap()":  
            normalize = True

        perturbed_modules = [copy.deepcopy(module) for _ in range(n_samples + 1)]  # list of N+1 models
        noise_gradients = []  # list of len(module.parameters()) noise gradients
        module_parameters = list(module.parameters())
        for i_param in range(len(module_parameters)):
            input_shape = module_parameters[i_param].shape  # [D, ] or multidim, whatever
            additive_noise, noise_gradient = sample_noise_with_gradients(n_samples, input_shape, normalize)
            additive_noise = additive_noise.to(device)
            noise_gradient = noise_gradient.to(device)  # [N+1, D]
            for idx_m, m in enumerate(perturbed_modules):
                list(m.parameters())[i_param] += epsilon * additive_noise[idx_m]
            noise_gradients.append(noise_gradient)
        perturbed_output = F_batch(perturbed_modules, x, y, fun)  # [N+1, ]

        ctx.save_for_backward(perturbed_output, *noise_gradients, torch.tensor(epsilon))
        ctx.module = module
        return torch.mean(perturbed_output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        tensors = ctx.saved_tensors
        perturbed_output = tensors[0]
        noise_gradients = tensors[1:-1]
        epsilon = tensors[-1]
        module = ctx.module

        F0 = perturbed_output[0]
        F_minus_F0 = grad_output * (perturbed_output[1:] - F0)
        module_parameters = list(module.parameters())
        for i_param in range(len(module_parameters)):
            noise_gradient = noise_gradients[i_param]
            F_minus_F0_reshaped = F_minus_F0
            for _ in range(len(noise_gradient.shape) - 1):
                F_minus_F0_reshaped = F_minus_F0_reshaped.unsqueeze(1)
            grad_theta = F_minus_F0_reshaped * noise_gradient[1:] / epsilon
            if module_parameters[i_param].grad is None:
                module_parameters[i_param].grad = torch.mean(grad_theta, dim=0)
            else:
                module_parameters[i_param].grad += torch.mean(grad_theta, dim=0)
        
        return None, None, None, None, None, None, None
        

def F_eps(theta, x, y, fun, n_samples, epsilon, device="cpu"):
    """
    This is the main function to be used for $F_\varepsilon$ computation.
    
    Arguments
    ---------

    theta: torch.tensor with requires_grad=True or torch.nn.Module
        Parameter(s) to be optimized

    x, y: torch.tensor s
        Input distributions in full dimension
    
    fun: function that takes x, y, theta as input
        Our $F$ function (operating on a single theta, be it a tensor of parameters or a module)

    n_samples: int
        Number of samples to be drawn for the Stein lemma's approximation of the gradient
    
    epsilon: float
        Standard deviation for the normal law to be used in the Stein lemma's approximation of the gradient
    
    device: str, default "cpu"
        Device to run torch operations on
    """
    if isinstance(theta, torch.nn.Module):
        # Ugly hack, not sure why we need that :(
        requires_grad = x.requires_grad
        x.requires_grad_()
        ret = F_epsilon_module.apply(theta, x, y, fun, n_samples, epsilon, device)
        x.requires_grad_(requires_grad)
        return ret
    else:
        return F_epsilon.apply(theta, x, y, fun, n_samples, epsilon, device)


def swgg_opt(x, y, model, opt, 
             n_iter=1000, epsilon_Stein=5e-2, n_samples_Stein=10, log=False):
    for i in range(n_iter):
        opt.zero_grad()
        loss = F_eps(model, x, y, fun=F_module, 
                     n_samples=n_samples_Stein, epsilon=epsilon_Stein)
        loss.backward()
        if log and i % 100 == 0:
            print(f"swgg log loss={loss.item():.3f}")
        opt.step()

