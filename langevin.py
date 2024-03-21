import numpy as np
from tqdm import tqdm
from opacus.optimizers.optimizer import DPOptimizer
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def unadjusted_langevin_algorithm(init_point, dim_w, X, y, lam, sigma, device, burn_in = 10000, len_list = 1, step=0.1, M = 1, m = 0, projection = None, batch_size = None):
    # randomly sample from N(0, C_lsi)
    if init_point == None:
        if m == 0:
            print('m not assigned, please check!')
        var = (2 * sigma**2) / m
        std = torch.sqrt(torch.tensor(var))
        w0 = torch.normal(mean=1000, std=std, size=(dim_w,)).reshape(-1).to(device)
    else:
        w0 = init_point.to(device)
    wi = w0
    samples = []
    if batch_size is None:
        for i in range(len_list + burn_in):
            z = torch.sigmoid(y * X.mv(wi))
            per_sample_grad = (X * ((z-1) * y).unsqueeze(-1) + lam * wi.repeat(X.size(0),1))
            row_norms = torch.norm(per_sample_grad,dim=1)
            clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
            clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
            grad = clipped_grad.mean(0)
            wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
            if projection is not None:
                w_norm = torch.norm(wi, p=2)
                wi = (wi / w_norm) * projection
            samples.append(wi.detach().cpu().numpy())
        return samples[burn_in:]
    else:
        # batch stochastic sgd for langevin
        # first sample a batch of y and X
        batch_list = random.sample(list(range(y.shape[0])), batch_size)
        X_batch = X[batch_list]
        y_batch = y[batch_list]
        for i in range(len_list + burn_in):
            z = torch.sigmoid(y_batch * X_batch.mv(wi))
            per_sample_grad = (X_batch * ((z-1) * y_batch).unsqueeze(-1) + lam * wi.repeat(X_batch.size(0),1))
            row_norms = torch.norm(per_sample_grad,dim=1)
            clipped_grad = per_sample_grad * ( M / row_norms).view(-1,1)
            clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
            grad = clipped_grad.mean(0)
            wi = wi.detach() - step * grad + np.sqrt(2 * step * sigma**2) * torch.randn(dim_w).to(device)
            if projection is not None:
                w_norm = torch.norm(wi, p=2)
                wi = (wi / w_norm) * projection
            samples.append(wi.detach().cpu().numpy())
        return samples[burn_in:]
