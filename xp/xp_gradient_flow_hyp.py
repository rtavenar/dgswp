import torch
from torch import nn
import ot
import argparse

import numpy as np

from torch.utils.data import DataLoader
from itertools import cycle
from tqdm.auto import trange
from copy import deepcopy


from lib_hyp import sampleWrappedNormal, hyper_sliced_wasserstein, sliced_wasserstein, horo_hyper_sliced_wasserstein_poincare, BusemannMap

from dgswp import GeneralizedSWGGGradientFlow

#most of the code is taken from the original code of the paper of Bonet, 2023
#https://github.com/clbonet/Hyperbolic_Sliced-Wasserstein_via_Geodesic_and_Horospherical_Projections
    

    


parser = argparse.ArgumentParser()



parser.add_argument("--type_target", type=str, default="mwnd", help="wnd or mwnd")
parser.add_argument("--target", type=str, default="center", help="Which target to use - border or center")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
parser.add_argument("--ntry", type=int, default=2, help="number of restart")
parser.add_argument("--lr", type=float, default=5, help="Learning rate")
parser.add_argument("--n_epochs", type=int, default=251, help="Number of epochs")
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
        
        
    J = torch.diag(torch.tensor([-1,1,1], device=device, dtype=torch.float64))
    
    num_projections = 100 #instead of 1000
    lr_hsw = args.lr
    lr_sw = args.lr
    lr_hhsw = args.lr
    lr_swgg = args.lr
    n_epochs = args.n_epochs
    n_try = args.ntry

    #n_samples = 200
    
    n_samples = 100
    
    mu0 = torch.tensor([1,0,0], dtype=torch.float64, device=device)
    Sigma0 = torch.eye(2, dtype=torch.float, device=device)
    
    L_hsw = np.zeros((n_try, n_epochs))
    L_sw = np.zeros((n_try, n_epochs))
    L_hhsw = np.zeros((n_try, n_epochs))
    L_swp = np.zeros((n_try, n_epochs)) 
    L_sswgg = np.zeros((n_try, n_epochs)) 
    L_sswgg_no_optim = np.zeros((n_try, n_epochs)) 



    model_swgg_optim = GeneralizedSWGGGradientFlow(learning_rate_flow=.1,
                                          n_iter_flow=n_epochs,
                                          model=BusemannMap(input_size=2),
                                          n_iter_inner=num_projections)
    
    model_swgg_no_optim = GeneralizedSWGGGradientFlow_busemann(learning_rate_flow=1,
                                          n_iter_flow=n_epochs,
                                          model=BusemannMap(input_size=2),
                                          n_iter_inner=num_projections, lr_inner = .01, n_samples=10, eps= 5e-3)
    
    
    
        
    for k in range(n_try):
        print("-----------------try #",k)
        if args.type_target == "wnd":
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
        
        x_hsw = deepcopy(x0)
        x_sw = deepcopy(x0)
        x_hhsw = deepcopy(lorentz_to_poincare(x0))
        x_swp = deepcopy(lorentz_to_poincare(x0))

        #with SWGG
        x_swgg = deepcopy(lorentz_to_poincare(x0))
        x_swgg_no_optim = deepcopy(lorentz_to_poincare(x0))

        x_hsw.requires_grad_(True)
        x_sw.requires_grad_(True)
        x_hhsw.requires_grad_(True)
        x_swp.requires_grad_(True)
        x_swgg.requires_grad_(True)
        x_swgg_no_optim.requires_grad_(True)
        
        if args.pbar:
            bar = trange(n_epochs)
        else:
            bar = range(n_epochs)
            
        for e in bar:
            print(e)
            X_target = next(dataiter).type(torch.float64).to(device)
            #-------------------hyperbolic sliced wasserstein
            if False:
                hsw = hyper_sliced_wasserstein(x_hsw, X_target, num_projections, device, p=2)
                grad_x0_hsw = torch.autograd.grad(hsw, x_hsw)[0]
                z_hsw = torch.matmul(grad_x0_hsw, J)
                proj_hsw = z_hsw + minkowski_ip(z_hsw, x_hsw) * x_hsw
                x_hsw = expMap(-lr_hsw*proj_hsw, x_hsw)
            #-------------------sliced wasserstein
            if False:
                sw = sliced_wasserstein(x_sw, X_target, num_projections, device, p=2)
                grad_x0_sw = torch.autograd.grad(sw, x_sw)[0]
                z_sw = torch.matmul(grad_x0_sw, J)
                proj_sw = z_sw + minkowski_ip(z_sw, x_sw) * x_sw
                x_sw = expMap(-lr_sw*proj_sw, x_sw)

            #-------------------horo_hyper_sliced_wasserstein_poincare
            if False:
                hhsw = horo_hyper_sliced_wasserstein_poincare(x_hhsw, lorentz_to_poincare(X_target), 
                                                    num_projections, device, p=2)
                grad_x0_hhsw = torch.autograd.grad(hhsw, x_hhsw)[0]
                norm_x = torch.norm(x_hhsw, dim=-1, keepdim=True)
                z = (1-norm_x**2)**2/4
                x_hhsw = exp_poincare(-lr_hhsw * z * grad_x0_hhsw, x_hhsw)

            #-------------------sliced wasserstein
            if False:
                swp = sliced_wasserstein(x_swp, lorentz_to_poincare(X_target), 
                                        num_projections, device, p=2)
                grad_x0_swp = torch.autograd.grad(swp, x_swp)[0]
                norm_x = torch.norm(x_swp, dim=-1, keepdim=True)
                z = (1-norm_x**2)**2/4
                x_swp = exp_poincare(-lr_hhsw * z * grad_x0_swp, x_swp)

            #-------------------generalized sliced wasserstein plans
            if e == 0 and True:
                #no optim means on busemann
                x_swgg_no_optim, _, loss_no_optim = model_swgg_no_optim.fit(source=x_swgg_no_optim, target=lorentz_to_poincare(X_target.float()))
                x_swgg, _, loss = model_swgg_optim.fit(source=x_swgg, target=lorentz_to_poincare(X_target.float()))
                
            
            #n = 500
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

            if False:
                if torch.any(torch.isinf(x_sw)):
                    L_sw[k, e] = np.inf
                else:
                    M_sw = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, x_sw), min=1+1e-15))**2
                    w_sw = ot.emd2(a, b, M_sw)
                    L_sw[k, e] = w_sw.item()

                
                if torch.any(torch.isnan(x_hsw)):
                    L_hsw[k, e] = np.inf
                else:
                    M_hsw = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, x_hsw), min=1+1e-15))**2
                    w_hsw = ot.emd2(a, b, M_hsw)
                    L_hsw[k, e] = w_hsw.item()
                    
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

            if True:#torch.any(torch.isnan(x_swgg)):
                #x_test is in Lorentz space, x_swgg on Poincare
                M_swgg = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, poincare_to_lorentz(torch.tensor(x_swgg[e]))), min=1+1e-15))**2
                w_swgg = ot.emd2(a, b, M_swgg)
                L_sswgg[k, e] = w_swgg.item()
                
                M_swgg_no_optim = torch.arccosh(torch.clamp(-minkowski_ip2(x_test, poincare_to_lorentz(torch.tensor(x_swgg_no_optim[e]))), min=1+1e-15))**2
                w_swgg_no_optim = ot.emd2(a, b, M_swgg_no_optim)
                L_sswgg_no_optim[k, e] = w_swgg_no_optim.item()


                
                
    #np.savetxt("./Results/sw_loss_"+args.type_target+"_"+args.target, L_sw)
    #np.savetxt("./Results/hsw_loss_"+args.type_target+"_"+args.target, L_hsw)
    #np.savetxt("./Results/hhsw_loss_"+args.type_target+"_"+args.target, L_hhsw)
    #np.savetxt("./Results/swp_loss_"+args.type_target+"_"+args.target, L_swp)
    np.savetxt("./Results/swgg_loss_"+args.type_target+"_"+args.target, L_sswgg)
    np.savetxt("./Results/swgg_loss_no_optim_"+args.type_target+"_"+args.target, L_sswgg_no_optim)
    
            
