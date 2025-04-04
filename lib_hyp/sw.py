import torch

import numpy as np
import torch.nn.functional as F

import lib_hyp.utils_hyperbolic as hyp

from losses import F_eps


device = "cuda" if torch.cuda.is_available() else "cpu"


def emd1D(u_values, v_values, u_weights=None, v_weights=None,p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    zero = torch.zeros(1, dtype=dtype, device=device)
    
    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
    
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    v_index = torch.searchsorted(v_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]
    
    if p == 1:
        return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
    if p == 2:
        return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)  
    return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)


def g_circular(x, theta, radius=2):
    """
        https://github.com/kimiandj/gsw/blob/9f7f0ce6ae74049cb9ed753c34a6deff14cd4417/code/gsw/gsw.py#L149
    """
    theta = torch.stack([radius*th for th in theta])
    return torch.stack([torch.sqrt(torch.sum((x-th)**2,dim=1)) for th in theta],1)


def get_powers(dim, degree):
    '''
    This function calculates the powers of a homogeneous polynomial
    e.g.
    list(get_powers(dim=2,degree=3))
    [(0, 3), (1, 2), (2, 1), (3, 0)]
    list(get_powers(dim=3,degree=2))
    [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    
    https://github.com/kimiandj/gsw/blob/9f7f0ce6ae74049cb9ed753c34a6deff14cd4417/code/gsw/gsw.py#L149
    '''
    if dim == 1:
        yield (degree,)
    else:
        for value in range(degree + 1):
            for permutation in get_powers(dim - 1,degree - value):
                yield (value,) + permutation
                

def homopoly(dim, degree):
    '''
    calculates the number of elements in a homogeneous polynomial
    
    https://github.com/kimiandj/gsw/blob/9f7f0ce6ae74049cb9ed753c34a6deff14cd4417/code/gsw/gsw.py#L149
    '''
    return len(list(get_powers(dim,degree)))


def g_poly(X, theta, device, degree=3):
    ''' The polynomial defining function for generalized Radon transform
        Inputs
        X:  Nxd matrix of N data samples
        theta: Lxd vector that parameterizes for L projections
        degree: degree of the polynomial
        
        https://github.com/kimiandj/gsw/blob/9f7f0ce6ae74049cb9ed753c34a6deff14cd4417/code/gsw/gsw.py#L149
    '''
    N, d = X.shape
    assert theta.shape[1]==homopoly(d, degree)
    powers=list(get_powers(d, degree))
    HX=torch.ones((N, len(powers))).to(device)
    for k,power in enumerate(powers):
        for i,p in enumerate(power):
            HX[:,k]*=X[:,i]**p
    if len(theta.shape)==1:
        return torch.matmul(HX,theta)
    else:
        return torch.matmul(HX,theta.t())


def sliced_cost(Xs, Xt, ftype="linear", projections=None, u_weights=None, v_weights=None, p=1, degree=3):

    if projections is not None and ftype == "linear":
        Xps = (Xs @ projections).T
        Xpt = (Xt @ projections).T
    elif projections is not None and ftype == "circular":
        Xps = g_circular(Xs, projections.T).T
        Xpt = g_circular(Xt, projections.T).T
    elif projections is not None and ftype=="poly":
        Xps = g_poly(Xs, projections.T, device=Xs.device, degree=degree).T
        Xpt = g_poly(Xt, projections.T, device=Xt.device, degree=degree).T
    else:
        Xps = Xs.T
        Xpt = Xt.T
                
    return torch.mean(emd1D(Xps,Xpt,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p))


def sliced_wasserstein(Xs, Xt, num_projections, device,
                       u_weights=None, v_weights=None, p=1, 
                       ftype="linear", degree=3):
    num_features = Xs.shape[1]

    # Random projection directions, shape (num_features, num_projections)
    if ftype=="poly":
        dpoly = homopoly(num_features, degree)
        projections = np.random.normal(size=(dpoly, num_projections))
    else:
        projections = np.random.normal(size=(num_features, num_projections))
    projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)

    return sliced_cost(Xs,Xt,projections=projections,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p, ftype=ftype, degree=degree)


def swgg_hyperbolic(Xs, Xt, num_directions, p=2):
    num_features = Xs.shape[1]
    projections = torch.randn(size=(num_directions, num_features), requires_grad=False)
    #projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)
    pos_x_1d = torch.argsort(hyp.busemann_poincare2(projections, Xs))
    pos_y_1d = torch.argsort(hyp.busemann_poincare2(projections, Xt))
    return torch.min(torch.mean(torch.sum(torch.square(Xs[pos_x_1d] -Xt[pos_y_1d]), dim=-1), dim=0)) #only for p = 2 for now


def F_nn(Xs, Xt, projections, p=2):
    pos_x_1d = torch.argsort(hyp.busemann_poincare2(projections, Xs))
    pos_y_1d = torch.argsort(hyp.busemann_poincare2(projections, Xt))
    return torch.min(torch.mean(torch.sum(torch.square(Xs[pos_x_1d] -Xt[pos_y_1d]), dim=-1), dim=0))#p=2

F_nn_sqeuc = lambda x, y, model: F_nn(x, y, model, p=2)

def swgg_hyperbolic_optim(Xs, Xt, num_directions, p=2):
    num_features = Xs.shape[1]
    projections = torch.randn(size=(num_directions, num_features), requires_grad=True)
    #projections = F.normalize(torch.from_numpy(projections), p=2, dim=0).type(Xs.dtype).to(device)

    num_iter = 20 
    optimizer = torch.optim.SGD([projections], lr=1e-2)
    loss_l=torch.empty(num_iter)
    #proj_l=torch.empty((num_iter,X.shape[1]))
    for i in range(num_iter):
            projections.data/=torch.norm(projections.data)
            optimizer.zero_grad()
            loss = F_eps(self.model, source, target, 
                         fun=F_nn_sqeuc, n_samples=self.n_samples, 
                         epsilon=self.epsilon)
            #loss = self.SWGG_smooth(X,Y,theta,s=s,std=std)
            loss.backward()
            optimizer.step()
        
            loss_l[i]=loss.data
            #proj_l[i,:]=theta.data
    res=self.SWGG_smooth(X,Y,theta.data.float(),s=1,std=0)
    return res

    pos_x_1d = torch.argsort(hyp.busemann_poincare2(projections, Xs))
    pos_y_1d = torch.argsort(hyp.busemann_poincare2(projections, Xt))
    return torch.min(torch.mean(torch.sum(torch.square(Xs[pos_x_1d] -Xt[pos_y_1d]), dim=-1), dim=0)) #only for p = 2 for now
