import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# why the original loss function here do not have a X.size(0) term in the 2nd term?
def lr_loss(w, X, y, lam):
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2




def logistic_density(w, X, y, lam):
    # w: weight (vector)
    # X: training data
    # y: training label {-1, +1}
    # lam: term in regularization that controls Lipschitz smoothness
    f = -F.logsigmoid(y * X.mv(w)).mean() + lam * X.size(0) * w.pow(2).sum() / 2
    f = 100 * f
    return torch.exp(-f)


def logistic_potential(w, X, y, lam):
    # w: weight (vector)
    # X: training data
    # y: training label {-1, +1}
    # lam: term in regularization that controls Lipschitz smoothness
    f = -F.logsigmoid(y * X.mv(w)).mean() + lam * X.size(0) * w.pow(2).sum() / 2
    f = 100 * f
    return f





# below are the versions with temperature and sf
'''def logistic_density(w, X, y, lam, temp, Sf = 1):
    # w: weight (vector)
    # X: training data
    # y: training label {-1, +1}
    # lam: term in regularization that controls Lipschitz smoothness
    # temp: the temperature that controls the landscape of density
    # Sf: here sup_D sup_w |f(w;D) - f(w;D')|, in binary case should be 1
    f = -F.logsigmoid(y * X.mv(w)).mean() + lam * X.size(0) * w.pow(2).sum() / 2
    # note here temp is on numerator, temp->0, uniform; temp -> \infty, landscape
    u = (f * temp) / (2 * Sf)
    return torch.exp(-u)


def logistic_potential(w, X, y, lam, temp, Sf = 1):
    # w: weight (vector)
    # X: training data
    # y: training label {-1, +1}
    # lam: term in regularization that controls Lipschitz smoothness
    # temp: the temperature that controls the landscape of density
    # Sf: here sup_D sup_w |f(w;D) - f(w;D')|, in binary case should be 1
    f = -F.logsigmoid(y * X.mv(w)).mean() + lam * X.size(0) * w.pow(2).sum() / 2
    # note here temp is on numerator, temp->0, uniform; temp -> \infty, landscape
    #return torch.exp( (-f * temp) / (2 * Sf) )
    u = (f * temp) / (2 * Sf)
    return u'''




