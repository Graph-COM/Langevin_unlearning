from hmac import new
import time
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
    def train(self):
        if self.args.search_burnin:
            # check the burn_in step required to converge
            if self.args.dataset == '2dgaussian':
                # list for 2dgaussian
                temp_list = [0.1, 1, 5]
                burn_in_list = [1, 10, 50, 100, 200, 500]
            elif self.args.dataset == 'MNIST':
                # list for MNIST
                temp_list = [1, 10, 100, 1000]
                burn_in_list = [1, 100, 200, 500, 1000, 2000, 3000]
            self.search_burnin(temp_list, burn_in_list)
        elif self.args.search_burnin_newdata:
            if self.args.dataset == '2dgaussian':
                # list for 2d gaussian
                num_remove_list = [100, 1000, 3000, 4900]
                burn_in_list = [1, 10, 100, 200, 500]
            elif self.args.dataset == 'MNIST':
                # list for MNIST
                num_remove_list = [100, 1000, 4000]
                burn_in_list = [1, 100, 200, 500, 1000, 2000, 3000]
            self.search_burnin_newdata(num_remove_list, burn_in_list)
        else:
            # given a single burn-in, temperature, sample from scratch on D:
            avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.temp, None, len_list = 1, return_w = True)
            print('the avg accuracy on original D from scratch is:'+str(avg_accuracy_scratch_D))
            print('the average time cost: '+str(mean_time))
            
            if self.args.search_finetune:
                if self.args.dataset == '2dgaussian':
                    # list for 2d gaussian
                    finetune_list = [1, 5, 10, 50, 100, 200, 500]
                    num_remove_list = [100, 1000, 3000, 4900]
                elif self.args.dataset == 'MNIST':
                    # list for MNIST
                    finetune_list = [1, 5, 10, 50, 100, 200, 500, 1000, 2000, 3000]
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
        
    def get_mean_performance(self, X, y, step, temp, w_list, len_list = 1, return_w = False, num_trial = 100):
        new_w_list = []
        trial_list = []
        time_list = []
        if w_list is None:
            for trial_idx in tqdm(range(num_trial)):
                w_init, time = self.run_unadjusted_langvin(None, X, y, step, temp, len_list)
                time_list.append(time)
                w_init = np.vstack(w_init)
                new_w_list.append(w_init)
                accuracy = self.test_accuracy(w_init)
                trial_list.append(accuracy)
        else:
            for trial_idx in tqdm(range(num_trial)):
                w = w_list[trial_idx].reshape(-1)
                w = torch.tensor(w)
                new_w, time = self.run_unadjusted_langvin(w, X, y, step, temp, len_list = 1)
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
                
    def search_burnin(self, temp_list, burn_in_list, fig_path = '_search_burnin.jpg'):
        for temp in temp_list:
            acc_list = []
            this_w_list = None
            for idx in range(len(burn_in_list)):
                if idx == 0:
                    step = burn_in_list[idx]
                else:
                    step = burn_in_list[idx] - burn_in_list[idx - 1]
                avg_accuracy, _, new_w_list = self.get_mean_performance(self.X_train, self.y_train, step, temp, this_w_list, return_w = True)
                this_w_list = new_w_list
                acc_list.append(avg_accuracy)
                print(acc_list)
            plt.plot(burn_in_list, acc_list, label='temp: '+str(temp))
            for i in range(len(burn_in_list)):
                plt.text(burn_in_list[i], acc_list[i], f'{acc_list[i]:.3f}', ha='right', va='bottom')
        plt.legend()
        plt.title(str(self.args.dataset)+'search burn in')
        plt.xlabel('burn in steps')
        plt.ylabel('accuracy')
        plt.savefig(str(self.args.dataset)+fig_path)
        plt.clf()
    
    def search_burnin_newdata(self, num_remove_list, burn_in_list, fig_path = '_search_burnin_on_Dnew.jpg'):
        for num_remove in num_remove_list:
            acc_list = []
            this_w_list = None
            X_train_removed = self.X_train[:-num_remove,:]
            y_train_removed = self.y_train[:-num_remove]
            for idx in range(len(burn_in_list)):
                if idx == 0:
                    step = burn_in_list[idx]
                else:
                    step = burn_in_list[idx] - burn_in_list[idx - 1]
                avg_accuracy, _, new_w_list = self.get_mean_performance(X_train_removed, y_train_removed, step, self.args.temp, this_w_list, return_w = True)
                this_w_list = new_w_list
                acc_list.append(avg_accuracy)
            plt.plot(burn_in_list, acc_list, label='num_remove: '+str(num_remove))
            for i in range(len(burn_in_list)):
                plt.text(burn_in_list[i], acc_list[i], f'{acc_list[i]:.3f}', ha='right', va='bottom')
        plt.legend()
        plt.title(str(self.args.dataset)+'search burn in on new D')
        plt.xlabel('burn in steps')
        plt.ylabel('accuracy')
        plt.savefig(str(self.args.dataset)+fig_path)
        plt.clf()    
    def test_accuracy(self, w_list):
        w = torch.tensor(w_list[0])
        # test accuracy (before removal)
        pred = self.X_test.mv(w)
        accuracy = pred.gt(0).eq(self.y_test.gt(0)).float().mean()
        return accuracy
    def run_unadjusted_langvin(self, init_point, X, y, burn_in, temp, len_list, Sf = 1):
        start_time = time.time()
        w_list = unadjusted_langevin_algorithm(init_point, self.dim_w, X, y, self.args.lam, temp = temp, device = self.device, potential = logistic_potential, burn_in = burn_in, len_list = len_list, step=self.args.step_size, Sf = Sf)
        end_time = time.time()
        return w_list, end_time - start_time

def main():
    parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--result-dir', type=str, default='./result', help='directory for saving results')
    parser.add_argument('--dataset', type=str, default='MNIST', help='[MNIST, 2dgaussian, kdgaussian]')
    parser.add_argument('--extractor', type=str, default='raw_feature', help='extractor type')
    parser.add_argument('--lam', type=float, default=1e-8, help='L2 regularization')
    parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
    parser.add_argument('--num-steps', type=int, default=10000, help='number of optimization steps')
    parser.add_argument('--train-mode', type=str, default='binary', help='train mode [ovr/binary]')

    parser.add_argument('--gpu', type = int, default = 6, help = 'gpu')
    parser.add_argument('--temp', type = float, default = 1000, help = 'the temperature tau_0')
    parser.add_argument('--burn_in', type = int, default = 500, help = 'burn in step number of LMC')
    parser.add_argument('--step_size', type = float, default = 0.1, help = 'the step size of LMC')
    parser.add_argument('--gaussian_dim', type = int, default = 10, help = 'dimension of gaussian task')
    parser.add_argument('--len_list', type = int, default = 10000, help = 'length of w to paint in 2D gaussian')
    parser.add_argument('--finetune_step', type = int, default = 50, help = 'steps to finetune on the new removed data')
    parser.add_argument('--search_burnin', type = int, default = 0, help = 'whether grid search to paint for burn-in')
    parser.add_argument('--search_finetune', type = int, default = 0, help = 'whether to grid search finetune')
    parser.add_argument('--search_burnin_newdata', type = int, default = 0, help = 'search burn in on new data')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.train()


if __name__ == '__main__':
    main()