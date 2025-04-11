import torch
from torch import nn
import ot
import argparse

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from itertools import cycle
from tqdm.auto import trange
from copy import deepcopy

import geoopt


from lib_hyp.utils_hyperbolic import *
from lib_hyp.distributions import sampleWrappedNormal
from lib_hyp.hsw import hyper_sliced_wasserstein
from lib_hyp.sw import sliced_wasserstein
from lib_hyp.hhsw import horo_hyper_sliced_wasserstein_poincare

from gradient_flows import (GeneralizedSWGGGradientFlowOnManifolds)

#most of the code is taken from the original code of the paper of Bonet, 2023
#https://github.com/clbonet/Hyperbolic_Sliced-Wasserstein_via_Geodesic_and_Horospherical_Projections
    
class BusemannMap(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        manifold_inner = geoopt.Sphere()
        self.direction = torch.randn(size=(1, input_size), dtype=torch.float64) #no bias here
        self.direction = geoopt.ManifoldTensor(self.direction, manifold=manifold_inner)
        self.direction.proj_() #make them belonging to the manifold
        self.direction  = geoopt.ManifoldParameter(self.direction)

        
    def busemann_poincare(self, x): #could be performed on Lorentz space as well
        return torch.log(torch.norm(self.direction[None]-x[:,None], dim=-1)**2 /
                         (torch.clamp(1-torch.norm(x,dim=-1,keepdim=True)**2, min=1e-10)))
    
    def manifold_paramter(self):
        return self.direction

    def forward(self, x):
        x = self.busemann_poincare(x)
        return x
    

    


parser = argparse.ArgumentParser()



parser.add_argument("--type_target", type=str, default="wnd_ddim", help="wnd or mwnd")
parser.add_argument("--target", type=str, default="center", help="Which target to use - border or center")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
parser.add_argument("--ntry", type=int, default=2, help="number of restart")
parser.add_argument("--lr", type=float, default=2.5, help="Learning rate")
parser.add_argument("--n_epochs", type=int, default=101, help="Number of epochs")
#args = parser.parse_args()
args, unknown = parser.parse_known_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    if args.type_target == "wnd":
        
        if args.target == "center":
            mu = torch.tensor([1.5, np.sqrt(1.5**2-1), 0], dtype=torch.float64, device=device)
        elif args.target == "border":
            mu = torch.tensor([8, np.sqrt(63), 0], dtype=torch.float64, device=device)
        Sigma = 0.1 * torch.tensor([[1,0],[0,1]], dtype=torch.float, device=device)

    elif args.type_target == "mwnd":
        ps = np.ones(5)/5
        if args.target == "center":
            mus_lorentz = torch.tensor([[0,0.5],[0.5,0],[0,-0.5],[-0.5,0], [0,0.1]], dtype=torch.float)
        elif args.target == "border":
            mus_lorentz = torch.tensor([[0,0.9],[0.9,0],[0,-0.9],[-0.9,0], [0,0.1]], dtype=torch.float)

        mus = poincare_to_lorentz(mus_lorentz)
        sigma = 0.01 * torch.tensor([[1,0],[0,1]], dtype=torch.float)
    elif args.type_target == "wnd_ddim":
        dim = 3
        mus = torch.ones(dim+1, dtype=torch.float64, device=device)
        mus *=1.5
        mus[-1] = 0
        sigma = 0.01 * torch.eye(dim, dtype=torch.float, device=device)
    
        
        
    J = torch.diag(torch.tensor([-1,1,1], device=device, dtype=torch.float64))
    
    num_projections = 100 #instead of 1000
    lr_sw = args.lr*7 # such that the speed is roughly the same 
    lr_hhsw = args.lr
    lr_swgg = args.lr/3
    n_epochs = args.n_epochs
    n_try = args.ntry

    n_samples = 50
    
    mu0 = torch.tensor([1,0,0], dtype=torch.float64, device=device)
    Sigma0 = torch.eye(2, dtype=torch.float, device=device)
    
    L_hsw = np.zeros((n_try, n_epochs))
    L_sw = np.zeros((n_try, n_epochs))
    L_hhsw = np.zeros((n_try, n_epochs))
    L_swp = np.zeros((n_try, n_epochs)) 
    L_sswgg = np.zeros((n_try, n_epochs)) 
    L_sswgg_euc = np.zeros((n_try, n_epochs)) 

    
    model_swgg = GeneralizedSWGGGradientFlowOnManifolds(learning_rate_flow=lr_swgg,
                                          n_iter_flow=n_epochs,
                                          model=BusemannMap(input_size=2),
                                          manifold="poincare",
                                          n_iter_inner=num_projections)
    
    #model_swgg_euc = GeneralizedSWGGGradientFlowOnManifolds(learning_rate_flow=lr_swgg,
    #                                      n_iter_flow=n_epochs,
    #                                      model=BusemannMap(input_size=2),
    #                                      manifold="euclidean",
    #                                      n_iter_inner=num_projections)
    
    
    
    for k in range(n_try):
        print("-----------------try #",k)
        if args.type_target == "wnd" or args.target == "wnd_ddim":
            X_target = sampleWrappedNormal(mu, Sigma, n_samples)#instead of 10000
        elif args.type_target == "mwnd":
            Z = np.random.multinomial(n_samples, ps)#instead of 10000
            X = []
            for l in range(len(Z)):
                if Z[l]>0:
                    samples = sampleWrappedNormal(mus[l], sigma, int(Z[l])).numpy()
                    X += list(samples)

            X_target = torch.tensor(X, device=device, dtype=torch.float)
            
        train_dl = DataLoader(X_target, batch_size=n_samples, shuffle=True)
        dataiter = iter(cycle(train_dl))
        
        x0 = sampleWrappedNormal(mu0, Sigma0, n_samples)
        
        x_hhsw = deepcopy(lorentz_to_poincare(x0))
        x_swp = deepcopy(lorentz_to_poincare(x0))
        x_swgg = deepcopy(lorentz_to_poincare(x0))
        x_swgg_euc = deepcopy(lorentz_to_poincare(x0))

        x_hhsw.requires_grad_(True)
        x_swp.requires_grad_(True)
        x_swgg.requires_grad_(True)
        x_swgg_euc.requires_grad_(True)
        
        if args.pbar:
            bar = trange(n_epochs)
        else:
            bar = range(n_epochs)
            
        for e in bar:
            X_target = next(dataiter).type(torch.float64).to(device)

            #-------------------horo_hyper_sliced_wasserstein_poincare
            if True:
                hhsw = horo_hyper_sliced_wasserstein_poincare(x_hhsw, lorentz_to_poincare(X_target), 
                                                    num_projections, device, p=2)
                grad_x0_hhsw = torch.autograd.grad(hhsw, x_hhsw)[0]
                norm_x = torch.norm(x_hhsw, dim=-1, keepdim=True)
                z = (1-norm_x**2)**2/4
                x_hhsw = exp_poincare(-lr_hhsw * z * grad_x0_hhsw, x_hhsw)

            #-------------------sliced wasserstein
            if True:
                swp = sliced_wasserstein(x_swp, lorentz_to_poincare(X_target), 
                                        num_projections, device, p=2)
                grad_x0_swp = torch.autograd.grad(swp, x_swp)[0]
                norm_x = torch.norm(x_swp, dim=-1, keepdim=True)
                z = (1-norm_x**2)**2/4
                x_swp = exp_poincare(-lr_hhsw * z * grad_x0_swp, x_swp)

            #-------------------generalized sliced wasserstein plans
            if e == 0 and True:
                #no optim means on busemann
                
                x_swgg, _, loss = model_swgg.fit(source=x_swgg, 
                                                 target=lorentz_to_poincare(
                                                     X_target.float()))
                
                #x_swgg_euc, _, loss = model_swgg_euc.fit(source=x_swgg_euc, 
                #                                 target=lorentz_to_poincare(
                #                                     X_target.float()))

                  
            n = n_samples           
            if args.type_target == "wnd":
                x_test = sampleWrappedNormal(mu, Sigma, n)
            elif args.type_target == "mwnd":
                Z = np.random.multinomial(n, ps)
                X = []
                for l in range(len(Z)):
                    if Z[l]>0:
                        samples = sampleWrappedNormal(mus[l], sigma, int(Z[l])).numpy()
                        X += list(samples)
                x_test = torch.tensor(X, device=device, dtype=torch.float)
                
            a = torch.ones((n,), device=device)/n
            b = torch.ones((n,), device=device)/n


            if True:
                if torch.any(torch.isnan(x_hhsw)):
                    L_hhsw[k, e] = np.inf
                else:
                    M_hhsw = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, poincare_to_lorentz(x_hhsw)), min=1+1e-15))**2
                    w_hhsw = ot.emd2(a, b, M_hhsw)
                    L_hhsw[k, e] = w_hhsw.item()
                
                if torch.any(torch.isnan(x_swp)):
                    L_swp[k, e] = np.inf
                else:
                    M_swp = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, poincare_to_lorentz(x_swp)), min=1+1e-15))**2
                    w_swp = ot.emd2(a, b, M_swp)
                    L_swp[k, e] = w_swp.item()

            if True:
                M_swgg = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, poincare_to_lorentz(torch.tensor(x_swgg[e]))), min=1+1e-15))**2
                w_swgg = ot.emd2(a, b, M_swgg)
                L_sswgg[k, e] = w_swgg.item()
                #M_swgg_euc = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, poincare_to_lorentz(torch.tensor(x_swgg_euc[e]))), min=1+1e-15))**2
                #w_swgg = ot.emd2(a, b, M_swgg_euc)
                #L_sswgg_euc[k, e] = w_swgg.item()




    L_hhsw10 = np.log10(L_hhsw)
    L_swp10 = np.log10(L_swp)
    L_swgg10 = np.log10(L_sswgg)

    mean_hhsw = np.mean(L_hhsw10, axis=0)
    std_hhsw = np.std(L_hhsw10, axis=0)

    mean_swp = np.mean(L_swp10, axis=0)
    std_swp = np.std(L_swp10, axis=0)

    mean_swgg = np.mean(L_swgg10, axis=0)
    std_swgg = np.std(L_swgg10, axis=0)

    n_epochs = len(mean_hhsw)
    iterations = range(n_epochs)

    plt.plot(iterations, mean_hhsw, label="HHSW")
    plt.fill_between(iterations, mean_hhsw-std_hhsw, mean_hhsw+std_hhsw, alpha=0.5)
    plt.plot(iterations, mean_swp, label="SW")
    plt.fill_between(iterations, mean_swp-std_swp, mean_swp+std_swp, alpha=0.5)
    plt.plot(iterations, mean_swgg, label="SWGG")
    plt.fill_between(iterations, mean_swgg-std_swgg, mean_swgg+std_swgg, alpha=0.5)

    plt.xlabel("Iterations", fontsize=13)
    plt.ylabel(r"$\log_{10}(W_2^2(\hat{\mu}_n,\nu))$", fontsize=13)
    plt.grid(True)
    plt.legend()

    np.savetxt("./Results/hhsw_loss_"+args.type_target+"_"+args.target, L_hhsw)
    np.savetxt("./Results/swp_loss_"+args.type_target+"_"+args.target, L_swp)
    np.savetxt("./Results/swgg_loss_"+args.type_target+"_"+args.target, L_sswgg)
    
    #np.savetxt("./Results/swgg_loss_"+args.type_target+"_"+args.target, L_sswgg_euc)
    
            
