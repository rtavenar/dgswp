import numpy as np
import copy
import torch
from torch.distributions.normal import Normal

def sample_noise_with_gradients(n_samples, shape, normalize=False):
    """Samples noise from a standard normal distribution along with its gradient.

    Args:
        n_samples (int): Number of noise samples to draw.
        shape (torch.Size or list): Shape of each noise sample.
        normalize (bool, optional): If True, normalizes the noise for the case of Busemann maps.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - all_samples (Tensor): Noise tensor of shape (n_samples + 1, *shape), with the first element being zero.
            - gradients (Tensor): Gradients, equal to the noise tensor in the case of N(0, 1).
    """
    actual_shape = [n_samples] + list(shape)
    sampler = Normal(0.0, 1.0)
    samples = sampler.sample(actual_shape)
    first_sample = torch.zeros(shape)
    if normalize:
        samples[:, :-1] /= torch.norm(samples[:,:-1], dim=-1, keepdim=True) ** 2

    all_samples = torch.cat((first_sample.unsqueeze(0), samples), dim=0)
    gradients = all_samples

    return all_samples, gradients

def H_linear(x, y, theta, p=2):
    """Computes a linear version of our $h$ function (cf. paper) along a projection theta.

    Args:
        x (torch.Tensor): First distribution sample.
        y (torch.Tensor): Second distribution sample.
        theta (torch.Tensor): Projection direction.
        p (int, optional): Power used in the distance calculation. Defaults to 2.

    Returns:
        torch.Tensor: Computed distance.
    """
    pos_x_1d = torch.argsort(theta @ x.T)
    pos_y_1d = torch.argsort(theta @ y.T)
    return torch.mean(torch.sum(torch.abs(x[pos_x_1d] - y[pos_y_1d]) ** p, dim=-1), dim=0)

def H_module(x, y, model, p=2):
    """Computes the model-based variant our $h$ function (cf. paper).

    Args:
        x (torch.Tensor): First distribution sample.
        y (torch.Tensor): Second distribution sample.
        model (nn.Module): A model that defines the projection.
        p (int, optional): Power used in the distance calculation. Defaults to 2.

    Returns:
        torch.Tensor: Computed distance.
    """
    pos_x_1d = torch.argsort(model(x).flatten())
    pos_y_1d = torch.argsort(model(y).flatten())
    return torch.mean(torch.sum(torch.abs(x[pos_x_1d] - y[pos_y_1d]) ** p, dim=-1), dim=0)

def H_batch(thetas, x, y, fun=None):
    """Computes distances for a batch of projection directions or models.

    Args:
        thetas (list or torch.Tensor): A list of projection directions or models.
        x (torch.Tensor): First distribution sample.
        y (torch.Tensor): Second distribution sample.
        fun (callable, optional): Distance function $h$ to use. Defaults to `H_linear`.

    Returns:
        torch.Tensor: Batch of distances.
    """
    if fun is None:
        fun = H_linear
    return torch.tensor([fun(x, y, theta) for theta in thetas], requires_grad=True)

class H_epsilon_linear(torch.autograd.Function):
    """Custom autograd function for computing $h_\\varepsilon$ (cf. paper) 
       for a projection direction $\\theta$ (linear case).
    """
    
    @staticmethod
    def forward(ctx, theta, x, y, fun, n_samples, epsilon, variance_reduction, device):
        """Forward pass for linear $h_\\varepsilon$ computation.

        Args:
            ctx: Context to save information for the backward pass.
            theta (torch.Tensor): Projection direction.
            x (torch.Tensor): First distribution sample.
            y (torch.Tensor): Second distribution sample.
            fun (callable): Distance function $h$.
            n_samples (int): Number of samples for gradient estimation (parameter $N$).
            epsilon (float): Noise scale (parameter $\\varepsilon$).
            variance_reduction (bool): Whether to apply variance reduction.
            device (str): Torch device.

        Returns:
            torch.Tensor: Mean perturbed output.
        """
        input_shape = theta.shape
        additive_noise, noise_gradient = sample_noise_with_gradients(n_samples, input_shape)
        additive_noise = additive_noise.to(device)
        noise_gradient = noise_gradient.to(device)
        perturbed_input = theta + epsilon * additive_noise
        perturbed_output = H_batch(perturbed_input, x, y, fun).to(device)

        ctx.save_for_backward(perturbed_output, noise_gradient, torch.tensor(epsilon))
        ctx.variance_reduction = variance_reduction
        return torch.mean(perturbed_output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computing gradients with respect to theta.

        Args:
            grad_output (torch.Tensor): Gradient of loss w.r.t. output.

        Returns:
            Tuple: Gradient w.r.t. theta and None for the other inputs.
        """
        perturbed_output, noise_gradient, epsilon = ctx.saved_tensors
        H0 = perturbed_output[0]
        if ctx.variance_reduction:
            H_minus_H0 = grad_output * (perturbed_output[1:] - H0)
        else:
            H_minus_H0 = grad_output * perturbed_output[1:]

        H_minus_H0_reshaped = H_minus_H0
        for _ in range(len(noise_gradient.shape) - 1):
            H_minus_H0_reshaped = H_minus_H0_reshaped.unsqueeze(1)

        grad_theta = H_minus_H0_reshaped * noise_gradient[1:] / epsilon
        return torch.mean(grad_theta, dim=0), None, None, None, None, None, None, None

class H_epsilon_module(torch.autograd.Function):
    """Custom autograd function for computing $h_\\varepsilon$ (cf. paper) 
       in the case of a torch.nn.Module.
    """

    @staticmethod
    def forward(ctx, module, x, y, fun, n_samples, epsilon, variance_reduction, device):
        """Forward pass for module-based $h_\\varepsilon$ computation.

        Args:
            ctx: Context for storing tensors.
            module (nn.Module): Module parameterized function.
            x (torch.Tensor): First input distribution.
            y (torch.Tensor): Second input distribution.
            fun (callable): Distance function $h$.
            n_samples (int): Number of samples for gradient estimation (parameter $N$).
            epsilon (float): Noise scale (parameter $\\varepsilon$).
            variance_reduction (bool): Whether to apply variance reduction.
            device (str): Torch device.

        Returns:
            torch.Tensor: Mean perturbed output.
        """
        normalize = True if str(module) == "BusemannMap()" else False

        perturbed_modules = [copy.deepcopy(module) for _ in range(n_samples + 1)]
        noise_gradients = []
        module_parameters = list(module.parameters())
        for i_param in range(len(module_parameters)):
            input_shape = module_parameters[i_param].shape
            additive_noise, noise_gradient = sample_noise_with_gradients(n_samples, input_shape, normalize)
            additive_noise = additive_noise.to(device)
            noise_gradient = noise_gradient.to(device)
            for idx_m, m in enumerate(perturbed_modules):
                list(m.parameters())[i_param] += epsilon * additive_noise[idx_m]
            noise_gradients.append(noise_gradient)
        perturbed_output = H_batch(perturbed_modules, x, y, fun).to(device)

        ctx.save_for_backward(perturbed_output, *noise_gradients, torch.tensor(epsilon))
        ctx.module = module
        ctx.variance_reduction = variance_reduction
        return torch.mean(perturbed_output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computing gradients w.r.t. module parameters.

        Args:
            grad_output (torch.Tensor): Gradient of the output.

        Returns:
            Tuple: Gradients for module parameters and None for other inputs.
        """
        tensors = ctx.saved_tensors
        perturbed_output = tensors[0]
        noise_gradients = tensors[1:-1]
        epsilon = tensors[-1]
        module = ctx.module

        H0 = perturbed_output[0]
        if ctx.variance_reduction:
            H_minus_H0 = grad_output * (perturbed_output[1:] - H0)
        else:
            H_minus_H0 = grad_output * perturbed_output[1:]

        module_parameters = list(module.parameters())
        for i_param in range(len(module_parameters)):
            noise_gradient = noise_gradients[i_param]
            H_minus_H0_reshaped = H_minus_H0
            for _ in range(len(noise_gradient.shape) - 1):
                H_minus_H0_reshaped = H_minus_H0_reshaped.unsqueeze(1)
            grad_theta = H_minus_H0_reshaped * noise_gradient[1:] / epsilon
            if module_parameters[i_param].grad is None:
                module_parameters[i_param].grad = torch.mean(grad_theta, dim=0)
            else:
                module_parameters[i_param].grad += torch.mean(grad_theta, dim=0)

        return None, None, None, None, None, None, None, None

def H_eps(theta, x, y, fun, n_samples, epsilon, variance_reduction=True, device="cpu"):
    """Main entry point for computing $h_\\varepsilon$ either with a vector or a module.

    Args:
        theta (Union[torch.Tensor, nn.Module]): Parameters (vector or module).
        x (torch.Tensor): First input distribution.
        y (torch.Tensor): Second input distribution.
        fun (callable): Function $h$.
        n_samples (int): Number of samples for gradient estimation (parameter $N$).
        epsilon (float): Noise scale (parameter $\\varepsilon$).
        variance_reduction (bool, optional): Whether to apply variance reduction. Defaults to True.
        device (str, optional): Device to run computations on. Defaults to "cpu".

    Returns:
        torch.Tensor: Estimated $h_\\varepsilon$ value.
    """
    if isinstance(theta, torch.nn.Module):
        requires_grad = x.requires_grad
        x.requires_grad_()
        ret = H_epsilon_module.apply(theta, x, y, fun, n_samples, epsilon, variance_reduction, device)
        x.requires_grad_(requires_grad)
        return ret
    else:
        return H_epsilon_linear.apply(theta, x, y, fun, n_samples, epsilon, variance_reduction, device)

def dgswp(x, y, model, opt, 
          n_iter=1000, epsilon_Stein=5e-2, n_samples_Stein=10, 
          roll_back=False, variance_reduction=True, 
          log_wass_loss=True, device="cpu"):
    """Performs training using gradient flow on the sliced Wasserstein potential.

    Args:
        x (torch.Tensor): First input distribution.
        y (torch.Tensor): Second input distribution.
        model (torch.nn.Module): Model to optimize.
        opt (torch.optim.Optimizer): Optimizer.
        n_iter (int, optional): Number of iterations. Defaults to 1000.
        epsilon_Stein (float, optional): Noise scale for Stein gradient. Defaults to 5e-2.
        n_samples_Stein (int, optional): Number of noise samples. Defaults to 10.
        roll_back (bool, optional): If True, reverts to the best model parameters. Defaults to False.
        variance_reduction (bool, optional): Enables variance reduction in gradient estimates. Defaults to True.
        log_wass_loss (bool, optional): If True, logs Wasserstein loss instead of surrogate. Defaults to True.
        device (str, optional): Device for computation. Defaults to "cpu".

    Returns:
        List[float]: Log of loss values across iterations.
    """
    mem_loss_inner = []
    mem_params = []
    log_losses = []
    for i in range(n_iter):
        opt.zero_grad()
        loss = H_eps(model, x, y, fun=H_module, 
                     n_samples=n_samples_Stein, epsilon=epsilon_Stein, 
                     variance_reduction=variance_reduction, device=device)
        if roll_back:
            mem_loss_inner.append(loss.item())
            mem_params.append(model.state_dict())
        if log_wass_loss:
            with torch.no_grad():
                log_losses.append(float(H_module(x, y, model)))
        else:
            log_losses.append(loss.item())
        loss.backward()
        opt.step()
    if n_iter > 0 and roll_back:
        model.load_state_dict(mem_params[np.argmin(mem_loss_inner)])
    return log_losses
