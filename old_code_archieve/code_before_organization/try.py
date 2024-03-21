from hmac import new
import time
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy.optimize import minimize_scalar
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from utils import load_features, generate_gaussian, plot_2dgaussian, plot_w_2dgaussian, create_nested_folder
from langevin import unadjusted_langevin_algorithm
from density import logistic_density, logistic_potential


class Runner():
    def __init__(self, args):
        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.args = args
        if args.dataset == 'MNIST':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.dim_w = 784
        elif args.dataset == 'CIFAR10':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.dim_w = 512
        # make the norm of x = 1, MNIST naturally satisfys
        self.X_train_norm = self.X_train.norm(dim=1, keepdim=True)
        self.X_train = self.X_train / self.X_train_norm
        self.X_test_norm = self.X_test.norm(dim=1, keepdim=True)
        self.X_test = self.X_test / self.X_test_norm
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
    def get_metadata(self):
        # num of training data
        self.n = len(self.X_train)
        print('number training data:'+str(self.n))
        # L-smoothness constant
        X = self.X_train.cpu().numpy()
        self.L = np.max(np.linalg.eigvalsh(X.T @ X / self.n)) / 4 + self.args.lam * self.n
        self.L = self.L
        print('L smooth constant'+str(self.L))
        # m-strongly convex constant
        self.m = self.args.lam * self.n
        self.m = self.m
        print('m strongly convex:'+str(self.m))
        # M-Lipschitz constant
        self.M = self.args.M
        print('M lipschitz constant:'+str(self.M))
        # calculate step size
        #max_eta = min( 2 * (1 - (self.m / 2)*(1/self.L + 1/self.m)) * (1/self.L + 1/self.m), 2 / (self.L + self.m) ) 
        #self.eta = min(max_eta, 1)
        self.eta = 1 / self.L
        print('step size eta:'+str(self.eta))
        # calculate RDP delta
        self.delta = 1 / self.n
        print('RDP constant delta:'+str(self.delta))

    def train(self):
        import pdb; pdb.set_trace()

def main():
    parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--result-dir', type=str, default='./result', help='directory for saving results')
    parser.add_argument('--dataset', type=str, default='MNIST', help='[MNIST, 2dgaussian, kdgaussian]')
    parser.add_argument('--extractor', type=str, default='raw_feature', help='extractor type')
    parser.add_argument('--lam', type=float, default=1e-6, help='L2 regularization')
    parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
    parser.add_argument('--num-steps', type=int, default=10000, help='number of optimization steps')
    parser.add_argument('--train-mode', type=str, default='binary', help='train mode [ovr/binary]')
    parser.add_argument('--M', type = float, default = 1, help = 'set M-Lipschitz constant (norm of gradient)')

    parser.add_argument('--gpu', type = int, default = 6, help = 'gpu')
    parser.add_argument('--sigma', type = float, default = 0.1, help = 'the parameter sigma')
    parser.add_argument('--burn_in', type = int, default = 10000, help = 'burn in step number of LMC')
    parser.add_argument('--gaussian_dim', type = int, default = 10, help = 'dimension of gaussian task')
    parser.add_argument('--len_list', type = int, default = 10000, help = 'length of w to paint in 2D gaussian')
    parser.add_argument('--finetune_step', type = int, default = 50, help = 'steps to finetune on the new removed data')
    parser.add_argument('--search_burnin', type = int, default = 0, help = 'whether grid search to paint for burn-in')
    parser.add_argument('--search_finetune', type = int, default = 0, help = 'whether to grid search finetune')
    parser.add_argument('--search_burnin_newdata', type = int, default = 0, help = 'search burn in on new data')
    parser.add_argument('--run_baseline', type = int, default = 1, help = 'run the baseline')
    parser.add_argument('--sequential', type = int, default = 0, help = 'whether test sequential unlearning')
    parser.add_argument('--find_k', type = int, default = 0, help = 'find the k')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()
    # here requires to find sigma by hand
    #runner.find_sigma()
    if args.sequential == 1:
        runner.sequential()
    elif args.find_k == 1:
        target_k_list = [1, 2, 5]
        #epsilon_list = [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
        epsilon_list = [0.05]
        for target_k in target_k_list:
            result_list = []
            for epsilon in epsilon_list:
                result_sigma = runner.search_k(target_k, epsilon)
                result_list.append(result_sigma)
            print('target k:'+str(target_k))
            print(result_list)

    else:
        runner.train()
    


if __name__ == '__main__':
    main()