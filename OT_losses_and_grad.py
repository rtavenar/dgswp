import numpy as np
from scipy.ndimage import gaussian_filter1d
import ot



def theta2w(theta):
    if theta.shape == (): #in 1d
        return np.array([np.cos(theta), np.sin(theta)])
    else:
        return theta.T


def G(x, y, theta): #inner problem
    w = theta2w(theta)
    x_1d = np.sort(x.dot(w))
    y_1d = np.sort(y.dot(w))
    return np.sum((x_1d - y_1d) ** 2)/len(x_1d)


def G_eps(x, y, theta, epsilon, n_samples):
    if  theta.shape == ():
        return np.mean([G(x, y, theta + epsilon * z) for z in np.random.randn(n_samples)])
    else:
        d = x.shape[1]
        return np.mean([G(x, y, theta + epsilon * z) for z in np.random.randn(n_samples, d)])
    

def grad_G_eps_Stein(x, y, theta, epsilon, n_samples):
    G0 = G(x, y, theta)
    if  theta.shape == ():
        return np.mean([(G(x, y, theta + epsilon * z) - G0) * z / epsilon for z in np.random.randn(n_samples)])
    else:
        d = x.shape[1]
        return np.mean([(G(x, y, theta + epsilon * z) - G0) * z / epsilon for z in np.random.randn(n_samples, d)], axis=0)


def F(x, y, theta): #outer problem
    w = theta2w(theta)
    pos_x_1d = np.argsort(x.dot(w))
    pos_y_1d = np.argsort(y.dot(w))
    return np.mean(np.sum(np.square(x[pos_x_1d] - y[pos_y_1d]), axis = -1), axis = 0)

def F_eps(x, y, theta, epsilon, n_samples):
    if  theta.shape == ():
        ths = theta + epsilon * np.random.randn(n_samples)
        return np.mean([F(x, y, th) for th in ths])
    else:
        d = x.shape[1]
        ths = theta + epsilon * np.random.randn(n_samples, d)
        return np.mean([F(x, y, th) for th in ths])


def grad_F_eps(x, y, theta, epsilon, n_samples): #making the assumption that it is smooth and non nul gradient
    F0 =  F(x, y, theta)
    if  theta.shape == ():
        return np.mean([(F(x, y, theta + epsilon * z) - F0) * z / epsilon for z in np.random.randn(n_samples)])
    else:
        d = x.shape[1]
        return np.mean([(F(x, y, theta + epsilon * z) - F0) * z / epsilon for z in np.random.randn(n_samples, d)], axis=0)




