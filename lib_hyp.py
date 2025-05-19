import torch
import math

import torch.nn.functional as F
import torch.distributions as D
from typing import Any, Tuple
from torch import Tensor

eps = 1e-7
max_clamp_norm = 40
max_norm = 85
ln_2: torch.Tensor = math.log(2)

def minkowski_ip(x, y, keepdim=True):
    if len(x.shape)==1:
        x = x.reshape(1,-1)
    if len(y.shape)==1:
        y = y.reshape(1,-1)
    
    if x.shape[0] != y.shape[0]:
        return -x[...,0][None]*y[...,0][:,None] + torch.sum(x[...,1:][None]*y[...,1:][:,None], axis=-1)
    else:
        return (-x[...,0]*y[...,0])[:,None] + torch.sum(x[...,1:]*y[...,1:], axis=-1, keepdim=True)

def parallelTransport(v, x0, x1):
    """
        Transport v\in T_x0 H to u\in T_x1 H by following the geodesics by parallel transport
    """
    n, d = v.shape
    if len(x0.shape)==1:
        x0 = x0.reshape(-1, d)
    if len(x1.shape)==1:
        x1 = x1.reshape(-1, d)
        
    u = v + minkowski_ip(x1, v)*(x0+x1)/(1-minkowski_ip(x1, x0))
    return u

def expMap(u, x):
    """
        Project u\in T_x H to the surface
    """
    
    if len(x.shape)==1:
        x = x.reshape(1, -1)
    
    norm_u = minkowski_ip(u,u)**(1/2)
    y = torch.cosh(norm_u)*x + torch.sinh(norm_u) * u/norm_u
    return y


def sampleWrappedNormal(mu, Sigma, n):
    device = mu.device
    
    d = len(mu)
    normal = D.MultivariateNormal(torch.zeros((d-1,), device=device),Sigma)
    x0 = torch.zeros((1,d), device=device)
    x0[0,0] = 1
    
    ## Sample in T_x0 H
    v_ = normal.sample((n,))
    v = F.pad(v_, (1,0))
    
    ## Transport to T_\mu H and project on H
    u = parallelTransport(v, x0, mu)    
    y = expMap(u, mu)
    
    return y 


def lorentz_to_poincare(y, r=1):
    return r*y[...,1:]/(r+y[...,0][:,None])


def expand_proj_dims(x: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(x.shape[:-1] + torch.Size([1])).to(x.device).to(x.dtype)
    return torch.cat((zeros, x), dim=-1)

# We will use this clamping technique to ensure numerical stability of the Exp and Log maps
class LeakyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None

def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)

def cosh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.cosh(x)

def sinh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.sinh(x)

# Exp map for the origin has a special form which doesn't need the Lorentz norm
def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    assert x[..., 0].allclose(torch.zeros_like(x[..., 0]))
    x = x[..., 1:]
    x_norm = torch.norm(x, p=2, keepdim=True, dim=-1) / radius
    x_normed = F.normalize(x, p=2, dim=-1) * radius
    ret = torch.cat((cosh(x_norm) * radius, sinh(x_norm) * x_normed), dim=-1)
    assert torch.isfinite(ret).all()
    return ret

class BusemannMap(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        init_dir = torch.randn(size=(1,input_size)) #no bias here
        self.direction = torch.nn.Parameter(init_dir/(torch.norm(init_dir, dim=-1)**2), requires_grad=True)
        
    def busemann_poincare(self, x): #could be performed on Lorentz space as well
        with torch.no_grad():
            self.direction.div_(torch.norm(self.direction, dim=-1, keepdim=True)**2)
        return torch.log(torch.norm(self.direction[None]-x[:,None], dim=-1)**2 /
                         (1-torch.norm(x,dim=-1,keepdim=True)**2))

    def forward(self, x):
        x = self.busemann_poincare(x)
        return x