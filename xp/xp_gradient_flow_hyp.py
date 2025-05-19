import torch
from torch import nn
import ot
import argparse

import numpy as np

from torch.utils.data import DataLoader
from itertools import cycle
from copy import deepcopy

import geoopt


from lib_hyp import (sampleWrappedNormal,
                     sliced_wasserstein,
                     horo_hyper_sliced_wasserstein_poincare,
                     poincare_to_lorentz,
                     lorentz_to_poincare,
                     minkowski_ip2,
                     exp_poincare)

from dgswp import PoincareDGSWPGradientFlow

#most of the code is taken from the original code of the paper of Bonet, 2023
#https://github.com/clbonet/Hyperbolic_Sliced-Wasserstein_via_Geodesic_and_Horospherical_Projections
    
class BusemannMap(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.init()
    
    def init(self):
        manifold_inner = geoopt.Sphere()
        direction = torch.randn(size=(1, self.input_size), dtype=torch.float64) #no bias here
        direction = geoopt.ManifoldTensor(direction, manifold=manifold_inner)
        direction.proj_() #make them belonging to the manifold
        self.direction  = geoopt.ManifoldParameter(direction)
        
    def busemann_poincare(self, x): #could be performed on Lorentz space as well
        return torch.log(torch.norm(self.direction[None]-x[:,None], dim=-1)**2 /
                         (torch.clamp(1-torch.norm(x,dim=-1,keepdim=True)**2, min=1e-10)))
    
    def manifold_paramter(self):
        return self.direction

    def forward(self, x):
        x = self.busemann_poincare(x)
        return x


parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default="center", 
                    help="Which target to use - border or center")
args, unknown = parser.parse_known_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

if args.target == "center":
    mu = torch.tensor([1.5, np.sqrt(1.5**2-1), 0], dtype=torch.float64, device=device)
elif args.target == "border":
    mu = torch.tensor([8, np.sqrt(63), 0], dtype=torch.float64, device=device)
Sigma = 0.1 * torch.tensor([[1,0],[0,1]], dtype=torch.float, device=device)

num_projections = 100 #instead of 1000
base_lr = 2.5
lr_sw = base_lr*7 # such that the speed is roughly the same 
lr_hhsw = base_lr
lr_dgswp = base_lr/3
n_repeat = 10
n_epochs = 300

n_samples = 50

mu0 = torch.tensor([1,0,0], dtype=torch.float64, device=device)
Sigma0 = torch.eye(2, dtype=torch.float, device=device)

L_hhsw = np.zeros((n_repeat, n_epochs))
L_swp = np.zeros((n_repeat, n_epochs)) 
L_dgswp = np.zeros((n_repeat, n_epochs))


model_dgswp = PoincareDGSWPGradientFlow(learning_rate_flow=lr_dgswp,
                                        n_iter_flow=n_epochs,
                                        model=BusemannMap(input_size=2),
                                        n_iter_inner=num_projections)    

for k in range(n_repeat):
    print("-----------------try #",k)
    torch.manual_seed(k)
    np.random.seed(k)

    X_target = sampleWrappedNormal(mu, Sigma, n_samples)#instead of 10000
        
    train_dl = DataLoader(X_target, batch_size=n_samples, shuffle=True)
    dataiter = iter(cycle(train_dl))
    
    x0 = sampleWrappedNormal(mu0, Sigma0, n_samples)
    
    x_hhsw = deepcopy(lorentz_to_poincare(x0))
    x_swp = deepcopy(lorentz_to_poincare(x0))
    x_dgswp = deepcopy(lorentz_to_poincare(x0))

    x_hhsw.requires_grad_(True)
    x_swp.requires_grad_(True)
    x_dgswp.requires_grad_(True)
        
    for e in range(n_epochs):
        X_target = next(dataiter).type(torch.float64).to(device)

        #-------------------horo_hyper_sliced_wasserstein_poincare
        hhsw = horo_hyper_sliced_wasserstein_poincare(x_hhsw, lorentz_to_poincare(X_target), 
                                            num_projections, device, p=2)
        grad_x0_hhsw = torch.autograd.grad(hhsw, x_hhsw)[0]
        norm_x = torch.norm(x_hhsw, dim=-1, keepdim=True)
        z = (1-norm_x**2)**2/4
        x_hhsw = exp_poincare(-lr_hhsw * z * grad_x0_hhsw, x_hhsw)

        #-------------------sliced wasserstein
        swp = sliced_wasserstein(x_swp, lorentz_to_poincare(X_target), 
                                num_projections, device, p=2)
        grad_x0_swp = torch.autograd.grad(swp, x_swp)[0]
        norm_x = torch.norm(x_swp, dim=-1, keepdim=True)
        z = (1-norm_x**2)**2/4
        x_swp = exp_poincare(-lr_hhsw * z * grad_x0_swp, x_swp)

        #-------------------generalized sliced wasserstein plans
        if e == 0:
            model_dgswp.init()
            x_dgswp, _, loss = model_dgswp.fit(source=x_dgswp, 
                                                target=lorentz_to_poincare(
                                                    X_target))
                
        n = n_samples           
        x_test = sampleWrappedNormal(mu, Sigma, n)
            
        a = torch.ones((n,), device=device)/n
        b = torch.ones((n,), device=device)/n

        for x, L in zip([x_hhsw, x_swp, x_dgswp],
                        [L_hhsw, L_swp, L_dgswp]):
            if torch.any(torch.isnan(x)):
                L[k, e] = np.inf
            else:
                M = torch.arccosh(torch.clamp(-minkowski_ip2(X_target, poincare_to_lorentz(x)), min=1+1e-15))**2
                L[k, e] = ot.emd2(a, b, M).item()

np.savetxt("results_hyp/hhsw_loss_wnd_"+args.target, L_hhsw)
np.savetxt("results_hyp/swp_loss_wnd_"+args.target, L_swp)
np.savetxt("results_hyp/dgswp_loss_wnd_"+args.target, L_dgswp)
