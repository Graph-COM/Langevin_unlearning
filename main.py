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
        self.L = 100 * self.L
        print('L smooth constant'+str(self.L))
        # m-strongly convex constant
        self.m = self.args.lam * self.n
        self.m = 100 * self.m
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
        if self.args.search_burnin:
            # check the burn_in step required to converge
            if self.args.dataset == '2dgaussian':
                # list for 2dgaussian
                sigma_list = [0.01, 0.1, 1, 2, 5]
                burn_in_list = [1, 10, 50, 100, 200, 500]
            elif self.args.dataset == 'MNIST':
                # list for MNIST
                sigma_list = [0.05, 0.1, 0.2]
                burn_in_list = [1, 10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000]
            _ = self.search_burnin(sigma_list, burn_in_list)
        elif self.args.search_burnin_newdata:
            if self.args.dataset == '2dgaussian':
                # list for 2d gaussian
                num_remove_list = [100, 1000, 3000, 4900]
                burn_in_list = [1, 10, 100, 200, 500]
            elif self.args.dataset == 'MNIST':
                # list for MNIST
                num_remove_list = [1, 5, 10, 50, 100, 200, 500, 1000]
                burn_in_list = [1, 10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000]
            _ = self.search_burnin_newdata(num_remove_list, burn_in_list)
        elif self.args.paint_utility_s:
            num_remove_list = [1, 5, 10, 50, 100, 200, 500, 1000]
            scratch_acc_list = []
            unlearn_acc_list = []
            avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
            scratch_acc_list.append(avg_accuracy_scratch_D)
            unlearn_acc_list.append(avg_accuracy_scratch_D)
            # calculate K
            epsilon_list = [1] # set epsilon = 1
            K_dict = self.search_finetune_step(epsilon_list, num_remove_list)
            K_list = []
            for num_remove in num_remove_list:
                X_train_removed, y_train_removed = self.get_removed_data(num_remove)
                avg_accuracy_scratch_Dnew, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, self.args.sigma, None)
                scratch_acc_list.append(avg_accuracy_scratch_Dnew)
                avg_accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove][1], self.args.sigma, w_list)
                unlearn_acc_list.append(avg_accuracy_finetune)
                K_list.append(K_dict[num_remove][1])
            x_list = [0] + num_remove_list
            plt.plot(x_list, scratch_acc_list, label='learn from scratch')
            plt.plot(x_list, unlearn_acc_list, label='unlearn')
            plt.legend()
            for i, k in enumerate(K_list):
                plt.text(x_list[i], K_list[i], f'K = {k}', fontsize=8)
            plt.xlabel('# removed data')
            plt.ylabel('test accuracy')
            plt.savefig('./paint_utility_s.pdf')
            plt.clf()
            import pdb; pdb.set_trace()
            

        else:
            # given a single burn-in, temperature, sample from scratch on D:
            avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
            print('the avg accuracy on original D from scratch is:'+str(avg_accuracy_scratch_D))
            print('the average time cost: '+str(mean_time))
            
            epsilon_list = [0.1, 0.5, 1, 2, 5]
            num_remove_list = [1, 2, 5, 10, 50, 100, 200, 500, 1000]
            K_dict = self.search_finetune_step(epsilon_list, num_remove_list) # K[num_remove][target_epsilon]
            
            import pdb; pdb.set_trace()
            if self.args.search_finetune:
                if self.args.dataset == '2dgaussian':
                    # list for 2d gaussian
                    finetune_list = [1, 5, 10, 50, 100, 200, 500]
                    num_remove_list = [100, 1000, 3000, 4900]
                elif self.args.dataset == 'MNIST':
                    # list for MNIST
                    finetune_list = [1, 5, 10, 50, 100, 200, 500, 1000, 2000]
                    num_remove_list = [100, 1000, 4000]
                self.search_finetune(w_list, finetune_list, num_remove_list)
            else:
                # given a single burn-in, temperature, sample from scratch on D':
                X_train_removed = self.X_train[:-self.args.num_removes,:]
                y_train_removed = self.y_train[:-self.args.num_removes]
                avg_accuracy_scratch_Dnew, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, self.args.temp, None)
                print('the avg accuracy on removed D from scratch is:'+str(avg_accuracy_scratch_Dnew))
                print('the average time cost: '+str(mean_time))
                # sample from the old w
                avg_accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.finetune_step, self.args.temp, w_list)
                print('the avg accuracy on removed D finetune is:'+str(avg_accuracy_finetune))
                print('the average time cost: '+str(mean_time))
        
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
        
    def epsilon_expression(self, K, sigma, eta, C_lsi, alpha, S, M, m, n, delta):
        #part_1 = math.exp(- (2 * float(K) * float(sigma) **2 * float(eta)) / (alpha * float(C_lsi)))
        part_1 = math.exp(- (float(K) * m * float(eta)) / (alpha))
        part_2 = (4 * alpha * float(S)**2 * float(M)**2) / (float(m) * float(sigma)**2 * float(n)**2)
        part_3 = (math.log(1 / float(delta))) / (alpha - 1)
        epsilon = part_1 * part_2 + part_3
        return epsilon
    def search_finetune_step(self, epsilon_list, num_remove_list):
        C_lsi = 2 * self.args.sigma**2 / self.m
        K_dict = {}
        for num_remove in num_remove_list:
            K_list = {}
            for target_epsilon in epsilon_list:
                K = 1
                epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, self.args.sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                while min_epsilon_with_k.fun > target_epsilon:
                    K = K + 10
                    epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, self.args.sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                    min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                K_list[target_epsilon] = K
                print('num remove:'+str(num_remove)+'target epsilon: '+str(target_epsilon)+'K: '+str(K)+'alpha: '+str(min_epsilon_with_k.x))
            K_dict[num_remove] = K_list
        return K_dict
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
    
    def search_finetune(self, w_list, finetune_list, num_remove_list, fig_path = '_search_finetune.jpg'):
        cmap = plt.cm.get_cmap('tab10') 
        num_colors = cmap.N 
        for i, num_remove in enumerate(num_remove_list):
            this_w_list = w_list
            color = cmap(i % num_colors)
            acc_list = []
            X_train_removed = self.X_train[:-num_remove,:]
            y_train_removed = self.y_train[:-num_remove]
            for idx in range(len(finetune_list)):
                if idx ==0:
                    finetune_step = finetune_list[idx]
                else:
                    finetune_step = finetune_list[idx] - finetune_list[idx - 1]
                acc, _, new_w_list = self.get_mean_performance(X_train_removed, y_train_removed, finetune_step, self.args.temp, this_w_list, return_w = True)
                this_w_list = new_w_list
                acc_list.append(acc)
            plt.plot(finetune_list, acc_list, label='num remove (finetune):'+str(num_remove), color =  color)
            for i in range(len(finetune_list)):
                plt.text(finetune_list[i], acc_list[i], f'{acc_list[i]:.3f}', ha='right', va='bottom')
            avg_accuracy_scratch_Dnew, _ = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, self.args.temp, None)
            plt.hlines(y=avg_accuracy_scratch_Dnew, xmin=min(finetune_list), xmax=max(finetune_list), label='num remove (scratch)'+str(num_remove), color = color)
        plt.legend()
        plt.title(str(self.args.dataset)+'search fine tune')
        plt.xlabel('finetune steps')
        plt.ylabel('accuracy')
        plt.savefig(str(self.args.dataset)+fig_path)
        plt.clf()
                
    def search_burnin(self, sigma_list, burn_in_list, fig_path = '_search_burnin.jpg'):
        acc_dict = {}
        for sigma in sigma_list:
            acc_list = []
            this_w_list = None
            for idx in range(len(burn_in_list)):
                if idx == 0:
                    step = burn_in_list[idx]
                else:
                    step = burn_in_list[idx] - burn_in_list[idx - 1]
                avg_accuracy, _, new_w_list = self.get_mean_performance(self.X_train, self.y_train, step, sigma, this_w_list, return_w = True)
                this_w_list = new_w_list
                acc_list.append(avg_accuracy)
                print(acc_list)
            plt.plot(burn_in_list, acc_list, label='sigma :'+str(sigma))
            acc_dict[sigma] = acc_list
            for i in range(len(burn_in_list)):
                plt.text(burn_in_list[i], acc_list[i], f'{acc_list[i]:.3f}', ha='right', va='bottom')
        plt.legend()
        plt.title(str(self.args.dataset)+'search burn in')
        plt.xlabel('burn in steps')
        plt.ylabel('accuracy')
        plt.savefig(str(self.args.dataset)+fig_path)
        plt.clf()
        return acc_dict
    
    def search_burnin_newdata(self, num_remove_list, burn_in_list, fig_path = '_search_burnin_on_Dnew.jpg'):
        acc_dict = {}
        for num_remove in num_remove_list:
            acc_list = []
            this_w_list = None
            X_train_removed, y_train_removed = self.get_removed_data(num_remove)
            for idx in range(len(burn_in_list)):
                if idx == 0:
                    step = burn_in_list[idx]
                else:
                    step = burn_in_list[idx] - burn_in_list[idx - 1]
                avg_accuracy, _, new_w_list = self.get_mean_performance(X_train_removed, y_train_removed, step, self.args.sigma, this_w_list, return_w = True)
                this_w_list = new_w_list
                acc_list.append(avg_accuracy)
                print(acc_list)
            plt.plot(burn_in_list, acc_list, label='num_remove: '+str(num_remove))
            acc_dict[num_remove] = acc_list
            for i in range(len(burn_in_list)):
                plt.text(burn_in_list[i], acc_list[i], f'{acc_list[i]:.3f}', ha='right', va='bottom')
        plt.legend()
        plt.title(str(self.args.dataset)+'search burn in on new D')
        plt.xlabel('burn in steps')
        plt.ylabel('accuracy')
        plt.savefig(str(self.args.dataset)+fig_path)
        plt.clf()
        return acc_dict 
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
    #parser.add_argument('--step_size', type = float, default = 0.1, help = 'the step size of LMC')
    parser.add_argument('--gaussian_dim', type = int, default = 10, help = 'dimension of gaussian task')
    parser.add_argument('--len_list', type = int, default = 10000, help = 'length of w to paint in 2D gaussian')
    parser.add_argument('--finetune_step', type = int, default = 50, help = 'steps to finetune on the new removed data')
    parser.add_argument('--search_burnin', type = int, default = 0, help = 'whether grid search to paint for burn-in')
    parser.add_argument('--search_finetune', type = int, default = 0, help = 'whether to grid search finetune')
    parser.add_argument('--search_burnin_newdata', type = int, default = 0, help = 'search burn in on new data')
    parser.add_argument('--paint_utility_s', type = int, default = 0, help = 'paint the utility - s figure')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()
    runner.train()


if __name__ == '__main__':
    main()