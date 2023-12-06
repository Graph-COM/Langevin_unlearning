import numpy as np
import torch
from tqdm import tqdm
from opacus.optimizers.optimizer import DPOptimizer
import torch.nn as nn
import torch.nn.functional as F


# unadjusted langevin dynamicx

'''class Model_w(nn.Module):
    def __init__(self, dim_w):
        super(Model_w, self).__init__()
        self.param = nn.Parameter(torch.Tensor(dim_w))
        nn.init.normal_(self.param, mean=0.0, std=1.0)
    def forward(self, x):
        return x.mv(self.param), self.param.pow(2).sum()

def unadjusted_langevin_algorithm(init_point, dim_w, X, y, lam, sigma, device, potential, burn_in = 10000, len_list = 1, step=0.1, M = 1):
    # randomly sample from N(0, I)
    wi = Model_w(dim_w)
    wi.to(device)
    if init_point is not None:
        wi.param.data = init_point.to(device)
    gd_optimizer = torch.optim.SGD(wi.parameters(), lr=step)
    gd_dp_optimizer = DPOptimizer(optimizer=gd_optimizer, noise_multiplier=0.0, max_grad_norm=M, expected_batch_size=11982)
    samples = []
    for i in range(len_list + burn_in):
        gd_dp_optimizer.zero_grad()
        term1, term2 = wi(X)
        loss =  - F.logsigmoid(y *term1).mean() + lam * X.size(0) * term2 / 2
        loss.backward()
        import pdb; pdb.set_trace()
        gd_dp_optimizer.step()
        print(wi)
        wi = wi + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
        import pdb; pdb.set_trace()
        samples.append(wi.detach().cpu().numpy())
    return samples[burn_in:]'''


# below is the slow version of per sample gradient
def unadjusted_langevin_algorithm(init_point, dim_w, X, y, lam, sigma, device, potential, burn_in = 10000, len_list = 1, step=0.1, M = 1):
    # randomly sample from N(0, I)
    if init_point == None:
        w0 = torch.randn(dim_w).to(device)
    else:
        w0 = init_point.to(device)
    wi = w0
    samples = []
    for i in range(len_list + burn_in):
        z = torch.sigmoid(y * X.mv(wi))
        per_sample_grad = X * ((z-1) * y).unsqueeze(-1) + lam * wi.repeat(X.size(0),1)
        row_norms = torch.norm(per_sample_grad,dim=1)
        clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
        clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
        grad = clipped_grad.mean(0)
        wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
        samples.append(wi.detach().cpu().numpy())
    return samples[burn_in:]


# below is batch gradient clip (wrong version)
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
