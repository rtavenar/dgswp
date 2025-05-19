from torch.optim import SGD, Adam
import numpy as np
import torch
import ot
from abc import ABC, abstractmethod

from . import H_eps, H_module

def wass(x, y):
    """
    Computes the squared Wasserstein distance between two point clouds x and y
    using the Earth Mover's Distance (EMD) implementation from the POT library.

    Args:
        x (Tensor): Source samples of shape (n, d).
        y (Tensor): Target samples of shape (m, d).

    Returns:
        float: The squared Wasserstein distance between x and y.
    """
    dists = ot.utils.dist(x, y)
    a = torch.ones(x.shape[0])/x.shape[0]
    b = torch.ones(y.shape[0])/y.shape[0]
    return ot.emd2(a, b, dists)

class GradientFlow(ABC):
    """
    Abstract base class for gradient flow-based optimization over distributions.
    """
    def __init__(self, learning_rate_flow, n_iter_flow):
        """
        Initializes the gradient flow parameters.

        Args:
            learning_rate_flow (float): Learning rate for the gradient flow.
            n_iter_flow (int): Number of iterations for the gradient flow.
        """
        self.learning_rate_flow = learning_rate_flow
        self.n_iter_flow = n_iter_flow
    
    def init(self):
        """
        Optional initialization method for subclasses.
        """
        pass

    def inner_fit(self, source, target):
        """
        Optional inner-loop optimization for learning additional parameters
        (e.g., neural networks). Default: no-op.
        """
        pass

    @abstractmethod
    def forward(self, source, target):
        """
        Computes the loss function to drive the gradient flow.

        Args:
            source (Tensor): Source distribution samples.
            target (Tensor): Target distribution samples.

        Returns:
            Tensor: Loss value.
        """
        raise NotImplementedError

    def fit(self, source, target):
        """
        Applies gradient flow to move source towards target distribution.

        Args:
            source (Tensor): Initial source distribution.
            target (Tensor): Target distribution.

        Returns:
            sources (list): Sequence of source positions.
            losses (list): List of objective values.
            losses_wass (list): List of true Wasserstein distances to target.
        """
        losses = []
        losses_wass = []
        source = torch.tensor(source.clone().detach().numpy(), 
                              dtype=torch.float32, requires_grad=True)
        sources = [source.clone().detach().numpy()]
        opt_source = SGD([source], lr=self.learning_rate_flow)
        for _ in range(self.n_iter_flow):
            self.inner_fit(source.detach(), target.detach())
            loss = self.forward(source, target)
            loss.backward()
            losses.append(loss.item())
            opt_source.step()
            opt_source.zero_grad()
            sources.append(source.clone().detach().numpy())
            losses_wass.append(wass(source, target).detach().numpy())
        return sources, losses, losses_wass

class DifferentiableGeneralizedWassersteinPlanGradientFlow(GradientFlow):
    """
    Implements a gradient flow using a learned transport plan via a neural network model.
    """
    def __init__(self, learning_rate_flow, n_iter_flow, model, 
                 n_iter_inner, learning_rate_inner=0.1, epsilon=5e-2, n_samples_stein=10):
        """
        Args:
            model (nn.Module): Neural transport model.
            n_iter_inner (int): Inner-loop optimization steps for model.
            learning_rate_inner (float): Learning rate for model training.
            epsilon (float): Noise scale (parameter $\\varepsilon$).
            n_samples_stein (int): Number of samples for gradient estimation (parameter $N$).
        """
        super().__init__(learning_rate_flow, n_iter_flow)
        self.epsilon = epsilon
        self.n_samples = n_samples_stein
        self.learning_rate_inner = learning_rate_inner
        self.model = model
        self.n_iter_inner = n_iter_inner
        self.opt_model = Adam(self.model.parameters(), lr=self.learning_rate_inner)
    
    def init(self):
        """
        Initializes the model optimizer and optionally the model.
        """
        if hasattr(self.model, "init"):
            self.model.init()
        self.opt_model = Adam(self.model.parameters(), lr=self.learning_rate_inner)
    
    def inner_fit(self, source, target):
        """
        Optimizes the projection model using the $h_\\varepsilon$ loss.
        """
        mem_loss_inner = []
        mem_params = []
        for _ in range(self.n_iter_inner):
            loss = H_eps(self.model, source, target, 
                         fun=H_module, n_samples=self.n_samples, 
                         epsilon=self.epsilon)
            mem_loss_inner.append(loss.item())
            mem_params.append(self.model.state_dict())
            loss.backward()
            self.opt_model.step()
            self.opt_model.zero_grad()
        self.model.load_state_dict(mem_params[np.argmin(mem_loss_inner)])

    def forward(self, source, target):
        """
        Computes the non-perturbed loss $h$ using the learned projection model.
        """
        return H_module(source, target, self.model)


class DifferentiableManifoldWassersteinPlanGradientFlow(DifferentiableGeneralizedWassersteinPlanGradientFlow):
    def __init__(self, learning_rate_flow, n_iter_flow, model, manifold="Euclidean", n_iter_inner=50, learning_rate_inner=.1, epsilon=5e-2, n_samples_stein=10):
        super().__init__(learning_rate_flow, n_iter_flow, model, n_iter_inner, learning_rate_inner, epsilon, n_samples_stein)

        self.manifold = manifold
        if self.manifold == "poincare" or self.manifold == "poincaré":
            self.fundist = F_nn_poincare
            self.fundist.__name__ = "F_nn_poincare"
        else:
            self.fundist = F_nn_sqeuc
            self.fundist.__name__ = "F_nn_sqeuc"

        self.opt_model = RiemannianSGD(self.model.parameters(), lr=self.learning_rate_inner) 

    def fit(self, source, target):
        losses = []
        losses_wass = []

        if self.manifold == "poincare":
            manifold = geoopt.PoincareBall()
        else:
            print("only the Poincaré manifold is implemented: running on Euclidean one")
            manifold = geoopt.Euclidean()
        source = torch.tensor(source.clone().detach().numpy(), 
                              dtype=source.dtype, requires_grad=True)
        source = geoopt.ManifoldTensor(source.clone().detach().numpy(), manifold=manifold, requires_grad=True)
        source = geoopt.ManifoldParameter(source, manifold=manifold)
        opt_source = RiemannianSGD([source], lr=self.learning_rate_flow, momentum=0.9)#, stabilize=1) #stabilize to avoid numerical instabilities
        
        sources = [source.clone().detach().numpy()]

        for _ in range(self.n_iter_flow):
            self.inner_fit(source.detach(), target.detach()) #optimize the direction
            opt_source.zero_grad()
            loss = self.forward(source, target) 
            #loss.backward() #retain_graph=True
            #losses.append(loss.item())

            prev_source = source.clone().detach()

            #to have the same setting than Bonet et al.
            grad_x0_swgg = torch.autograd.grad(loss, source)[0]
            norm_x = torch.norm(source, dim=-1, keepdim=True)
            z = (1-norm_x**2)**2/4
            if grad_x0_swgg.isnan().any(): # to deal with numerical issues
                with torch.no_grad():
                    posNan = np.where(torch.isnan(grad_x0_swgg))[0]
            source = exp_poincare(-self.learning_rate_flow * z * grad_x0_swgg, source)

            #opt_source.step()
            if source.isnan().any(): #Optimizing on manifolds is prone to numerical instabilities
            #    #in that case, we do not update the source
                #print("Nan's in source, not updating")
                with torch.no_grad():
                    posNan = np.where(torch.isnan(source))[0]
                    source[np.where(torch.isnan(source))[0]] = prev_source[np.where(torch.isnan(source))[0]]

            sources.append(source.clone().detach().numpy())
            losses_wass.append(wass_poincare(source, target).detach().numpy())
            opt_source.zero_grad()
        return sources, losses, losses_wass

class SlicedWassersteinGradientFlow(GradientFlow):
    """
    Implements sliced Wasserstein gradient flow using random projections.
    """
    def __init__(self, learning_rate_flow, n_iter_flow, n_directions):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_directions = n_directions

    def forward(self, source, target):
        """
        Computes the average sliced Wasserstein loss across random directions.
        """
        directions = torch.randn((self.n_directions, source.shape[1]), requires_grad=False)
        directions /= torch.norm(directions, dim=-1, keepdim=True)
        ordered_sources = torch.sort(directions @ source.T, dim=-1)[0]
        ordered_targets = torch.sort(directions @ target.T, dim=-1)[0]
        return torch.mean(torch.abs(ordered_sources - ordered_targets) ** 2)

class MaxSlicedWassersteinGradientFlow(GradientFlow):
    """
    Gradient flow using the maximum sliced Wasserstein direction.
    """
    def __init__(self, learning_rate_flow, n_iter_flow, d, n_iter_inner, learning_rate_inner):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_iter_inner = n_iter_inner
        self.learning_rate_inner = learning_rate_inner
        self.d = d
    
    def init(self):
        """
        Initializes a direction vector for inner-loop maximization.
        """
        self.theta_ = torch.randn(self.d, dtype=torch.float32, requires_grad=False)
        self.theta_ /= torch.norm(self.theta_, p=2)
        self.theta_.requires_grad_()
        self.opt_inner = Adam([self.theta_], lr=self.learning_rate_inner)

    def inner_fit(self, source, target):
        """
        Optimizes the projection direction to maximize the sliced Wasserstein distance.
        """
        for _ in range(self.n_iter_inner):
            loss = - self.forward(source, target)
            loss.backward()
            self.opt_inner.step()
            self.opt_inner.zero_grad()

    def forward(self, source, target):
        """
        Computes the sliced Wasserstein loss in the optimized direction.
        """
        ordered_sources = torch.sort(self.theta_ @ source.T, dim=-1)[0]
        ordered_targets = torch.sort(self.theta_ @ target.T, dim=-1)[0]
        return torch.mean(torch.abs(ordered_sources - ordered_targets) ** 2)

class RandomSearchSWGGGradientFlow(GradientFlow):
    """
    Uses SWGG and random directions for gradient flow.
    """
    def __init__(self, learning_rate_flow, n_iter_flow, n_directions):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_directions = n_directions

    def forward(self, source, target):
        """
        Computes a sliced loss using random directions and returns the best.
        """
        directions = torch.randn((self.n_directions, source.shape[1]), requires_grad=False)
        directions /= torch.norm(directions, dim=-1, keepdim=True)
        ordered_sources = source[torch.argsort(directions @ source.T, dim=-1)]
        ordered_targets = target[torch.argsort(directions @ target.T, dim=-1)]
        return torch.min(torch.mean(torch.sum(torch.abs(ordered_sources - ordered_targets) ** 2, dim=2), dim=1))

class SWGGGradientFlow(GradientFlow):
    """
    SWGG Gradient Flow (optimized version).
    """
    def __init__(self, learning_rate_flow, n_iter_flow, 
                 n_iter_inner, learning_rate_inner, d,
                 s=10, epsilon=.5, device="cpu"):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_iter_inner = n_iter_inner
        self.learning_rate_inner = learning_rate_inner
        self.s = s
        self.epsilon = epsilon
        self.device = device
        self.d = d
    
    def init(self):
        """
        Initializes the direction vector and optimizer for slicing.
        """
        self.theta_ = torch.randn(self.d, dtype=torch.float32, requires_grad=False)
        self.theta_ /= torch.norm(self.theta_, p=2)
        self.theta_.requires_grad_()
        self.opt_inner = Adam([self.theta_], lr=self.learning_rate_inner)

    def inner_fit(self, source, target):
        """
        Optimizes the slicing direction using the SWGG objective.
        """
        for _ in range(self.n_iter_inner):
            with torch.no_grad():
                self.theta_ /= torch.norm(self.theta_, p=2)
            loss = self.forward(source, target, s=self.s, epsilon=self.epsilon)
            loss.backward()
            self.opt_inner.step()
            self.opt_inner.zero_grad()

    def forward(self, source, target, s=1, epsilon=0.):
        """
        Computes the SWGG loss.

        Args:
            s (int): Number of repetitions for Gaussian blurring.
            epsilon (float): Blur intensity.

        Returns:
            Tensor: The SWGG loss.
        """
        n,dim=source.shape
        
        X_line=torch.matmul(source, self.theta_)
        Y_line=torch.matmul(target, self.theta_)
        
        X_line_sort,u=torch.sort(X_line,axis=0)
        Y_line_sort,v=torch.sort(Y_line,axis=0)
        
        X_sort=source[u]
        Y_sort=target[v]
        
        Z_line=(X_line_sort+Y_line_sort)/2
        Z=Z_line[:,None] * self.theta_[None,:]
        
        W_XZ=torch.sum((X_sort-Z)**2)/n
        W_YZ=torch.sum((Y_sort-Z)**2)/n
        
        X_line_extend = X_line_sort.repeat_interleave(s,dim=0)
        X_line_extend_blur = X_line_extend + 0.5 * epsilon * torch.randn(X_line_extend.shape,device=self.device)
        Y_line_extend = Y_line_sort.repeat_interleave(s,dim=0)
        Y_line_extend_blur = Y_line_extend + 0.5 * epsilon * torch.randn(Y_line_extend.shape,device=self.device)
        
        X_line_extend_blur_sort,u_b=torch.sort(X_line_extend_blur,axis=0)
        Y_line_extend_blur_sort,v_b=torch.sort(Y_line_extend_blur,axis=0)
        
        X_extend=X_sort.repeat_interleave(s,dim=0)
        Y_extend=Y_sort.repeat_interleave(s,dim=0)
        X_sort_extend=X_extend[u_b]
        Y_sort_extend=Y_extend[v_b]
        
        bary_extend=(X_sort_extend+Y_sort_extend)/2
        bary_blur=torch.mean(bary_extend.reshape((n,s,dim)),dim=1)
        
        W_baryZ=torch.sum((bary_blur-Z)**2)/n
        return -4*W_baryZ+2*W_XZ+2*W_YZ

class AugmentedSlicedWassersteinGradientFlow(GradientFlow):
    """
    Augmented sliced Wasserstein flow with a learned feature transformation.
    """
    def __init__(self, learning_rate_flow, n_iter_flow, n_directions, model, lambda_,
                 learning_rate_inner=.01, n_iter_inner=10):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_directions = n_directions
        self.lambda_ = lambda_
        self.learning_rate_inner = learning_rate_inner
        self.model = model
        self.n_iter_inner = n_iter_inner
        self.opt_model = Adam(self.model.parameters(), lr=self.learning_rate_inner)
    
    def init(self):
        """
        Initializes the model and its optimizer.
        """
        if hasattr(self.model, "init"):
            self.model.init()
        self.opt_model = Adam(self.model.parameters(), lr=self.learning_rate_inner)
    
    def inner_fit(self, source, target):
        """
        Optimizes the transformation model to minimize augmented SW loss.
        """
        for _ in range(self.n_iter_inner):
            loss_sw, m_source, m_target = self._forward(source, target)
            reg = self.lambda_ * torch.mean(torch.norm(m_source, p=2, dim=1)
                                            + torch.norm(m_target, p=2, dim=1))
            loss = reg - loss_sw
            loss.backward()
            self.opt_model.step()
            self.opt_model.zero_grad()

    def _forward(self, source, target):
        """
        Projects source and target using the model and computes SW distance.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: loss value, mapped source, mapped target.
        """
        m_source = self.model(source)
        m_target = self.model(target)
        directions = torch.randn((self.n_directions, m_source.shape[1]), requires_grad=False)
        directions /= torch.norm(directions, dim=-1, keepdim=True)
        ordered_sources = torch.sort(directions @ m_source.T, dim=-1)[0]
        ordered_targets = torch.sort(directions @ m_target.T, dim=-1)[0]
        return torch.sqrt(torch.sum(torch.abs(ordered_sources - ordered_targets) ** 2, dim=-1).mean()), m_source, m_target

    def forward(self, source, target):
        """
        Computes the augmented sliced Wasserstein loss using the learned mapping.
        """
        return self._forward(source, target)[0]
