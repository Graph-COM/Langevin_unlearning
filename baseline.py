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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from utils import load_features, generate_gaussian, plot_2dgaussian, plot_w_2dgaussian
from langevin import unadjusted_langevin_algorithm
from density import logistic_density, logistic_potential


class Runner():
    def __init__(self, args):
        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.args = args
        if args.dataset == 'MNIST':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.dim_w = 784
        elif args.dataset == '2dgaussian':
            mean_1 = torch.tensor([-2, -2])
            mean_2 = torch.tensor([2, 2])
            std = torch.tensor([1, 1])
            self.X_train, self.y_train = generate_gaussian(2, 10000, mean_1, mean_2, std)
            self.X_test, self.y_test = generate_gaussian(2, 1000, mean_1, mean_2, std)
            self.dim_w = 2
        elif args.dataset == 'kdgaussian':
            mean_1 = torch.ones(args.gaussian_dim) * -2
            mean_2 = torch.ones(args.gaussian_dim) * 2
            std = torch.ones(args.gaussian_dim)
            self.X_train, self.y_train = generate_gaussian(args.gaussian_dim, 10000, mean_1, mean_2, std)
            self.X_test, self.y_test = generate_gaussian(args.gaussian_dim, 1000, mean_1, mean_2, std)
            self.dim_w = args.gaussian_dim
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
        max_eta = min( 2 * (1 - (self.m / 2)*(1/self.L + 1/self.m)) * (1/self.L + 1/self.m), 2 / (self.L + self.m) ) 
        #self.eta = min(max_eta, 1)
        self.eta = max_eta
        print('step size eta:'+str(self.eta))
        # calculate RDP delta
        self.delta = 1 / self.n
        print('RDP constant delta:'+str(self.delta))
        
    def train(self):
        if self.args.run_baseline:
            epsilon_list = [0.1, 0.5, 1, 2, 5]
            # first run algorithm #1 to learn and get parameters
            baseline_step_size = 2 / (self.L + self.m)
            X_train_removed, y_train_removed = self.get_removed_data(1)
            baseline_learn_scratch_acc, mean_time, w_list = self.get_mean_baseline(self.X_train, self.y_train, baseline_step_size, self.args.burn_in, None, len_list = 1, return_w = True)
            print('baseline learn scratch acc: ' + str(baseline_learn_scratch_acc))
            baseline_unlearn_scratch_acc, mean_time = self.get_mean_baseline(X_train_removed, y_train_removed, baseline_step_size, self.args.burn_in, None, len_list = 1)
            print('baseline unlearn scratch acc: ' + str(baseline_unlearn_scratch_acc))
            _, mean_time, w_list_new = self.get_mean_baseline(X_train_removed, y_train_removed, baseline_step_size, 1, w_list, len_list = 1, return_w = True)
            baseline_unlearn_finetune_acc_list = []
            for epsilon in epsilon_list:
                baseline_sigma = self.calculate_baseline_sigma(epsilon)
                print('baseline sigma: ' + str(baseline_sigma))
                random_noise = np.random.normal(0, baseline_sigma, (100, 1, self.dim_w))
                w_list_new_totest = torch.tensor(w_list_new + random_noise).float()
                baseline_unlearn_finetune_acc_list = []
                for i in range(100):
                    accuracy = self.test_accuracy(w_list_new_totest[i])
                    baseline_unlearn_finetune_acc_list.append(accuracy)
                baseline_unlearn_finetune_acc = np.mean(baseline_unlearn_finetune_acc_list)
                print('baseline unlearn finetune acc: ' + str(baseline_unlearn_finetune_acc))
                baseline_unlearn_finetune_acc_list.append(baseline_unlearn_finetune_acc)

            # run langevin unlearning
            num_remove_list = [1]
            sigma_list = [0.094, 0.019, 0.0096, 0.0049, 0.0021]
            for epsilon, sigma in zip(epsilon_list, sigma_list):
                print('epsilon: ' + str(epsilon))
                self.args.sigma = sigma
                K_dict, _ = self.search_finetune_step(epsilon_list, num_remove_list, self.args.sigma)
                lmc_learn_scratch_acc, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
                print('LMc learn scratch acc: ' + str(lmc_learn_scratch_acc))
                lmc_unlearn_scratch_acc, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, self.args.sigma, None, len_list = 1)
                print('LMc unlearn scratch acc: ' + str(lmc_unlearn_scratch_acc))
                lmc_unlearn_finetune_acc_list = []
                lmc_unlearn_finetune_acc, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, int(K_dict[num_remove_list[0]][epsilon]), self.args.sigma, w_list, len_list = 1)
                print('LMc unlearn finetune acc: ' + str(lmc_unlearn_finetune_acc))
                lmc_unlearn_finetune_acc_list.append(lmc_unlearn_finetune_acc)
            import pdb; pdb.set_trace()
    def find_sigma(self):
        num_remove_list = [1]
        epsilon_list = [0.1, 0.5, 1, 2, 5]
        sigma = 0.1
        K_dict, _ = self.search_finetune_step(epsilon_list, num_remove_list, sigma)
        print(K_dict)
        #import pdb; pdb.set_trace()
    def get_mean_baseline(self, X, y, baseline_step_size, step, w_list, len_list = 1, return_w = False, num_trial = 100):
        new_w_list = []
        trial_list = []
        time_list = []
        if w_list is None:
            for trial_idx in tqdm(range(num_trial)):
                w_init, time = self.run_gradient_descent(None, X, y, baseline_step_size, step, len_list)
                time_list.append(time)
                w_init = np.vstack(w_init)
                new_w_list.append(w_init)
                accuracy = self.test_accuracy(w_init)
                trial_list.append(accuracy)
        else:
            for trial_idx in tqdm(range(num_trial)):
                w = w_list[trial_idx].reshape(-1)
                w = torch.tensor(w)
                new_w, time = self.run_gradient_descent(w, X, y, baseline_step_size, step, len_list = 1)
                time_list.append(time)
                new_w = np.vstack(new_w)
                new_w_list.append(new_w)
                accuracy = self.test_accuracy(new_w)
                trial_list.append(accuracy)
        avg_accuracy = np.mean(trial_list)
        mean_time = np.mean(time_list)

        if return_w:
            new_w_list = np.stack(new_w_list, axis=0)
            return avg_accuracy, mean_time, new_w_list
        else:
            return avg_accuracy, mean_time
    
    def get_mean_performance(self, X, y, step, sigma, w_list, len_list = 1, return_w = False, num_trial = 100):
        new_w_list = []
        trial_list = []
        time_list = []
        if w_list is None:
            for trial_idx in tqdm(range(num_trial)):
                w_init, time = self.run_unadjusted_langvin(None, X, y, step, sigma, len_list)
                time_list.append(time)
                w_init = np.vstack(w_init)
                new_w_list.append(w_init)
                accuracy = self.test_accuracy(w_init)
                trial_list.append(accuracy)
        else:
            for trial_idx in tqdm(range(num_trial)):
                w = w_list[trial_idx].reshape(-1)
                w = torch.tensor(w)
                new_w, time = self.run_unadjusted_langvin(w, X, y, step, sigma, len_list = 1)
                time_list.append(time)
                new_w = np.vstack(new_w)
                new_w_list.append(new_w)
                accuracy = self.test_accuracy(new_w)
                trial_list.append(accuracy)
        avg_accuracy = np.mean(trial_list)
        mean_time = np.mean(time_list)

        if return_w:
            new_w_list = np.stack(new_w_list, axis=0)
            return avg_accuracy, mean_time, new_w_list
        else:
            return avg_accuracy, mean_time
    
    def run_gradient_descent(self, init_point, X, y, baseline_step_size, burn_in, len_list):
        start_time = time.time()
        w_list = self.gradient_descent_algorithm(init_point, self.dim_w, X, y, self.args.lam, device = self.device, potential = logistic_potential, burn_in = burn_in, len_list = len_list, step=baseline_step_size, M = self.M)
        end_time = time.time()
        return w_list, end_time - start_time

    def gradient_descent_algorithm(self, init_point, dim_w, X, y, lam, device, potential, burn_in = 10000, len_list = 1, step=0.1, M = 1):
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
            wi = wi.detach() - step * grad
            #w_norm = torch.norm(wi)
            #wi = wi / w_norm
            samples.append(wi.detach().cpu().numpy())
        return samples[burn_in:]
    def get_removed_data(self, num_remove):
        X_train_removed = self.X_train[:-num_remove,:]
        y_train_removed = self.y_train[:-num_remove]
        new_X_train = torch.randn(num_remove, self.dim_w)
        norms = new_X_train.norm(dim=1, keepdim=True)
        new_X_train = new_X_train / norms
        new_X_train = new_X_train.to(self.device)
        new_y_train = torch.randint(0, 2, (1, num_remove)) * 2 - 1
        new_y_train = new_y_train.to(self.device).reshape(-1)
        X_train_removed = torch.cat((X_train_removed, new_X_train), 0)
        y_train_removed = torch.cat((y_train_removed, new_y_train))
        return X_train_removed, y_train_removed
        
    
    def calculate_baseline_sigma(self, I, D = 2, epsilon = 1):
        # calculate the noise for descent to delete
        gamma = (self.L - self.m) / (self.L + self.m)
        numerator = 4 * math.sqrt(2) * self.M * gamma**I
        dominator = self.m * self.n * (1 - gamma**I) * (math.sqrt(math.log(1 / self.delta) + epsilon) - math.sqrt(math.log(1 / self.delta)))
        return numerator / dominator
    
    def epsilon_expression(self, K, sigma, eta, C_lsi, alpha, S, M, m, n, delta):
        #part_1 = math.exp(- (2 * float(K) * float(sigma) **2 * float(eta)) / (alpha * float(C_lsi)))
        part_1 = math.exp(- (float(K) * m * float(eta)) / (alpha))
        part_2 = (4 * alpha * float(S)**2 * float(M)**2) / (float(m) * float(sigma)**2 * float(n)**2)
        part_3 = (math.log(1 / float(delta))) / (alpha - 1)
        epsilon = part_1 * part_2 + part_3
        return epsilon
    
    def search_finetune_step(self, epsilon_list, num_remove_list, sigma):
        C_lsi = 2 * sigma**2 / self.m
        K_dict = {}
        alpha_dict = {}
        for num_remove in num_remove_list:
            K_list = {}
            alpha_list = {}
            for target_epsilon in epsilon_list:
                K = 1
                epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                while min_epsilon_with_k.fun > target_epsilon:
                    K = K + 10
                    epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                    min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                K_list[target_epsilon] = K
                alpha_list[target_epsilon] = min_epsilon_with_k.x
                print('num remove:'+str(num_remove)+'target epsilon: '+str(target_epsilon)+'K: '+str(K)+'alpha: '+str(min_epsilon_with_k.x))
            K_dict[num_remove] = K_list
            alpha_dict[num_remove] = alpha_list
        return K_dict, alpha_dict

    def test_accuracy(self, w_list):
        w = torch.tensor(w_list[0])
        # test accuracy (before removal)
        pred = self.X_test.mv(w)
        accuracy = pred.gt(0).eq(self.y_test.gt(0)).float().mean()
        return accuracy
    def run_unadjusted_langvin(self, init_point, X, y, burn_in, sigma, len_list):
        start_time = time.time()
        w_list = unadjusted_langevin_algorithm(init_point, self.dim_w, X, y, self.args.lam, sigma = sigma, device = self.device, potential = logistic_potential, burn_in = burn_in, len_list = len_list, step=self.eta, M = self.M)
        end_time = time.time()
        return w_list, end_time - start_time

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
    parser.add_argument('--burn_in', type = int, default = 1000, help = 'burn in step number of LMC')
    parser.add_argument('--gaussian_dim', type = int, default = 10, help = 'dimension of gaussian task')
    parser.add_argument('--len_list', type = int, default = 10000, help = 'length of w to paint in 2D gaussian')
    parser.add_argument('--finetune_step', type = int, default = 50, help = 'steps to finetune on the new removed data')
    parser.add_argument('--search_burnin', type = int, default = 0, help = 'whether grid search to paint for burn-in')
    parser.add_argument('--search_finetune', type = int, default = 0, help = 'whether to grid search finetune')
    parser.add_argument('--search_burnin_newdata', type = int, default = 0, help = 'search burn in on new data')
    parser.add_argument('--run_baseline', type = int, default = 1, help = 'run the baseline')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()
    #runner.find_sigma()
    runner.train()


if __name__ == '__main__':
    main()