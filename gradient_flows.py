from torch.optim import SGD, Adam
import numpy as np
import torch
import ot
from abc import ABC, abstractmethod

from losses import F_eps

torch.manual_seed(0)

def wass(x, y):
    dists = ot.utils.dist(x, y)
    a = torch.ones(x.shape[0])/x.shape[0]
    b = torch.ones(y.shape[0])/y.shape[0]
    return ot.emd2(a, b, dists)

def F_nn(x, y, model, p):
    pos_x_1d = torch.argsort(model(x).flatten())
    pos_y_1d = torch.argsort(model(y).flatten())
    return torch.mean(torch.sum(torch.abs(x[pos_x_1d] - y[pos_y_1d]) ** p, dim=-1), dim=0)

F_nn_sqeuc = lambda x, y, model: F_nn(x, y, model, p=2)


class GradientFlow(ABC):
    def __init__(self, learning_rate_flow, n_iter_flow):
        self.learning_rate_flow = learning_rate_flow
        self.n_iter_flow = n_iter_flow

    def inner_fit(self, source, target):
        # Only useful for methods that will **optimize** an inner model
        # eg. useless for Sliced Wass or SWGG without optim
        pass

    @abstractmethod
    def forward(self, source, target):
        raise NotImplementedError

    def fit(self, source, target):
        losses = []
        losses_wass = []
        source = torch.tensor(source.clone().detach().numpy(), 
                              dtype=torch.float32, requires_grad=True)
        sources = [source.clone().detach().numpy()]
        opt_source = Adam([source], lr=self.learning_rate_flow)
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

class GeneralizedSWGGGradientFlow(GradientFlow):
    def __init__(self, learning_rate_flow, n_iter_flow, model, 
                 n_iter_inner, learning_rate_inner=.01, epsilon=5e-2, n_samples_stein=10):
        super().__init__(learning_rate_flow, n_iter_flow)

        self.epsilon = epsilon
        self.n_samples = n_samples_stein
        self.learning_rate_inner = learning_rate_inner

        self.model = model
        self.n_iter_inner = n_iter_inner
        self.opt_model = Adam(self.model.parameters(), lr=self.learning_rate_inner)
    
    def inner_fit(self, source, target):
        mem_loss_inner = []
        mem_params = []
        for _ in range(self.n_iter_inner):
            loss = F_eps(self.model, source, target, 
                         fun=F_nn_sqeuc, n_samples=self.n_samples, 
                         epsilon=self.epsilon)
            mem_loss_inner.append(loss.item())
            mem_params.append(self.model.state_dict())
            loss.backward()
            self.opt_model.step()
            self.opt_model.zero_grad()
        self.model.load_state_dict(mem_params[np.argmin(mem_loss_inner)])

    def forward(self, source, target):
        return F_nn_sqeuc(source, target, self.model)

class SlicedWassersteinGradientFlow(GradientFlow):
    def __init__(self, learning_rate_flow, n_iter_flow, n_directions):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_directions = n_directions

    def forward(self, source, target):
        directions = torch.randn((self.n_directions, source.shape[1]), requires_grad=False)
        directions /= torch.norm(directions, dim=-1, keepdim=True)
        ordered_sources = torch.sort(directions @ source.T, dim=-1)[0]  # n_dir, n
        ordered_targets = torch.sort(directions @ target.T, dim=-1)[0]  # n_dir, n
        return torch.mean(torch.abs(ordered_sources - ordered_targets) ** 2)

class MaxSlicedWassersteinGradientFlow(GradientFlow):
    def __init__(self, learning_rate_flow, n_iter_flow, d, n_iter_inner, learning_rate_inner):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_iter_inner = n_iter_inner
        self.learning_rate_inner = learning_rate_inner
        self.theta_ = torch.randn(d, dtype=torch.float32, requires_grad=False)
        self.theta_ /= torch.norm(self.theta_, p=2)
        self.theta_.requires_grad_()
        self.opt_inner = Adam([self.theta_], lr=self.learning_rate_inner)

    def inner_fit(self, source, target):
        for _ in range(self.n_iter_inner):
            loss = - self.forward(source, target)
            loss.backward()
            self.opt_inner.step()
            self.opt_inner.zero_grad()

    def forward(self, source, target):
        ordered_sources = torch.sort(self.theta_ @ source.T, dim=-1)[0]  # n, 
        ordered_targets = torch.sort(self.theta_ @ target.T, dim=-1)[0]  # n, 
        return torch.mean(torch.abs(ordered_sources - ordered_targets) ** 2)

class UnOptimizedSWGGGradientFlow(GradientFlow):
    def __init__(self, learning_rate_flow, n_iter_flow, n_directions):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_directions = n_directions

    def forward(self, source, target):
        directions = torch.randn((self.n_directions, source.shape[1]), requires_grad=False)
        directions /= torch.norm(directions, dim=-1, keepdim=True)
        ordered_sources = source[torch.argsort(directions @ source.T, dim=-1)]  # n_dir, n, d
        ordered_targets = target[torch.argsort(directions @ target.T, dim=-1)]  # n_dir, n, d
        return torch.min(torch.mean(torch.sum(torch.abs(ordered_sources - ordered_targets) ** 2, dim=2), dim=1))

class SWGGGradientFlow(GradientFlow):
    def __init__(self, learning_rate_flow, n_iter_flow, 
                 n_iter_inner, learning_rate_inner, d,
                 s=10, epsilon=.5, device="cpu"):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_iter_inner = n_iter_inner
        self.learning_rate_inner = learning_rate_inner
        self.s = s
        self.epsilon = epsilon
        self.device = device
        self.theta_ = torch.randn(d, dtype=torch.float32, requires_grad=False)
        self.theta_ /= torch.norm(self.theta_, p=2)
        self.theta_.requires_grad_()
        self.opt_inner = Adam([self.theta_], lr=self.learning_rate_inner)

    def inner_fit(self, source, target):
        for _ in range(self.n_iter_inner):
            loss = self.forward(source, target, s=self.s, epsilon=self.epsilon)
            loss.backward()
            self.opt_inner.step()
            self.opt_inner.zero_grad()


    def forward(self, source, target, s=1, epsilon=0.):
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
    def __init__(self, learning_rate_flow, n_iter_flow, n_directions, model, lambda_,
                 learning_rate_inner=.01, n_iter_inner=10):
        super().__init__(learning_rate_flow, n_iter_flow)
        self.n_directions = n_directions
        self.lambda_ = lambda_

        self.learning_rate_inner = learning_rate_inner
        self.model = model
        self.n_iter_inner = n_iter_inner
        self.opt_model = Adam(self.model.parameters(), lr=self.learning_rate_inner)
    
    def inner_fit(self, source, target):
        for _ in range(self.n_iter_inner):
            loss_sw, m_source, m_target = self._forward(source, target)
            reg = self.lambda_ * torch.mean(torch.norm(m_source, p=2, dim=1)
                                            + torch.norm(m_target, p=2, dim=1))
            loss = reg - loss_sw
            loss.backward()
            self.opt_model.step()
            self.opt_model.zero_grad()

    def _forward(self, source, target):
        m_source = self.model(source)
        m_target = self.model(target)
        directions = torch.randn((self.n_directions, m_source.shape[1]), requires_grad=False)
        directions /= torch.norm(directions, dim=-1, keepdim=True)
        ordered_sources = torch.sort(directions @ m_source.T, dim=-1)[0]  # n_dir, n
        ordered_targets = torch.sort(directions @ m_target.T, dim=-1)[0]  # n_dir, n
        return torch.sqrt(torch.sum(torch.abs(ordered_sources - ordered_targets) ** 2, dim=-1).mean()), m_source, m_target

    def forward(self, source, target):
        return self._forward(source, target)[0]