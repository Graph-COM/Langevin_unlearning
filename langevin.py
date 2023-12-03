import numpy as np
import torch
from tqdm import tqdm



# unadjusted langevin dynamicx
def unadjusted_langevin_algorithm(init_point, dim_w, X, y, lam, sigma, device, potential, burn_in = 10000, len_list = 1, step=0.1, M = 1):
    # randomly sample from N(0, I)
    if init_point == None:
        w0 = torch.randn(dim_w).to(device)
    else:
        w0 = init_point.to(device)
    wi = w0
    samples = []
    for i in range(len_list + burn_in):
        wi.requires_grad_()
        u = potential(wi, X, y, lam)
        grad = torch.autograd.grad(u, wi)[0]
        # clip the grad to norm = 1
        grad_norm = torch.norm(grad)
        if grad_norm > M:
            grad = grad / grad_norm
        wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
        samples.append(wi.detach().cpu().numpy())
    #return samples
    return samples[burn_in:]




# ther version below is about temperature and 2*Sf
'''def unadjusted_langevin_algorithm(init_point, dim_w, X, y, lam, temp, device, potential, burn_in = 10000, len_list = 1, step=0.1, Sf = 1):
    # randomly sample from N(0, I)
    if init_point == None:
        w0 = torch.randn(dim_w).to(device)
    else:
        w0 = init_point.to(device)
    wi = w0
    samples = []
    for i in range(len_list + burn_in):
        wi.requires_grad_()
        u = potential(wi, X, y, lam, temp, Sf)
        #u = potential(wi)
        grad = torch.autograd.grad(u, wi)[0]
        wi = wi.detach() - step * grad + np.sqrt(2 * step) * torch.randn(dim_w).to(device)
        samples.append(wi.detach().cpu().numpy())
    #return samples
    return samples[burn_in:]'''
