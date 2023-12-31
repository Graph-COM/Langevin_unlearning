import numpy as np
import torch
from tqdm import tqdm
from opacus.optimizers.optimizer import DPOptimizer
import torch.nn as nn
import torch.nn.functional as F



# below is the slow version of per sample gradient
# on Dec 30: add clip the weight after each update, (projection function)
def unadjusted_langevin_algorithm(init_point, dim_w, X, y, lam, sigma, device, potential, burn_in = 10000, len_list = 1, step=0.1, M = 1, projection = 0, batch_size = 0):
    # randomly sample from N(0, I)
    if init_point == None:
        w0 = torch.randn(dim_w).to(device)
    else:
        w0 = init_point.to(device)
    wi = w0
    samples = []
    if batch_size == 0:
        for i in range(len_list + burn_in):
            z = torch.sigmoid(y * X.mv(wi))
            per_sample_grad = X * ((z-1) * y).unsqueeze(-1) + lam * wi.repeat(X.size(0),1)
            row_norms = torch.norm(per_sample_grad,dim=1)
            clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
            clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
            grad = clipped_grad.mean(0)
            wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
            if projection != 0:
                w_norm = torch.norm(wi, p=2)
                wi = (wi / w_norm) * projection
            samples.append(wi.detach().cpu().numpy())
        return samples[burn_in:]
    else:
        # batch stochastic sgd for langevin
        for i in range(len_list + burn_in):
            z = torch.sigmoid(y * X.mv(wi))
            per_sample_grad = X * ((z-1) * y).unsqueeze(-1) + lam * wi.repeat(X.size(0),1)
            row_norms = torch.norm(per_sample_grad,dim=1)
            clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
            clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
            grad = clipped_grad.mean(0)
            wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
            if projection != 0:
                w_norm = torch.norm(wi, p=2)
                wi = (wi / w_norm) * projection
            samples.append(wi.detach().cpu().numpy())
        return samples[burn_in:]


# below is whole gradient clip (wrong version)
# the code below only clips the gradient of all samples, but indeed we need to characterize the per-sample gradient
'''def unadjusted_langevin_algorithm(init_point, dim_w, X, y, lam, sigma, device, potential, burn_in = 10000, len_list = 1, step=0.1, M = 1):
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
    return samples[burn_in:]'''




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
