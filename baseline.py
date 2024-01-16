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

    def sequential2(self):
        num_remove_list = [100]
        num_remove_per_itr_list = [5, 10, 20]
        target_epsilon = 1
        create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/sequential2/')
        baseline_step_size = 2 / (self.L + self.m)

        baseline_I_factor_list = [1, 1.5, 2]

        for baseline_I_factor in baseline_I_factor_list:
            # first run the baseline, sequentially delete
            baseline_learn_scratch_acc, mean_time, baseline_w_list = self.get_mean_baseline(self.X_train, self.y_train, baseline_step_size, self.args.burn_in, None, len_list = 1, return_w = True)
            np.save('./result/LMC/'+str(self.args.dataset)+'/sequential2/baseline_acc_scratch'+str(baseline_I_factor)+'.npy', baseline_learn_scratch_acc)
            print('baseline learn scratch acc: ' + str(np.mean(baseline_learn_scratch_acc)))
            print('baseline learn scratch acc std: ' + str(np.std(baseline_learn_scratch_acc)))

            min_I = self.get_baseline_min_I(target_epsilon)
            print('min I:'+str(min_I))

            term1 = int(min_I * baseline_I_factor)
            baseline_k_list = []
            for baseline_step in range(1, num_remove_list[0]+1):
                X_train_removed, y_train_removed = self.get_removed_data(int((baseline_step)))
                term2 = self.get_baseline_term2(baseline_step)
                baseline_k = term1 + term2
                baseline_k_list.append(baseline_k)
                print('step:'+str(baseline_step)+'k:'+str(baseline_k))
                
                _, mean_time, baseline_w_list = self.get_mean_baseline(X_train_removed, y_train_removed, baseline_step_size, baseline_k, baseline_w_list, len_list = 1, return_w = True)
                baseline_sigma = self.calculate_baseline_sigma2(baseline_k, epsilon = target_epsilon)
                print('baseline sigma: ' + str(baseline_sigma))
                random_noise = np.random.normal(0, baseline_sigma, (100, 1, self.dim_w))
                baseline_w_list = torch.tensor(baseline_w_list + random_noise).float()
                baseline_unlearn_finetune_acc_list = []
                for i in range(100):
                    accuracy = self.test_accuracy(baseline_w_list[i])
                    baseline_unlearn_finetune_acc_list.append(accuracy)
                print('baseline step:'+str(baseline_step)+'acc:'+str(np.mean(baseline_unlearn_finetune_acc_list)))
                np.save('./result/LMC/'+str(self.args.dataset)+'/sequential2/baseline_acc_Ifac_'+str(baseline_I_factor)+'_step_'+str(baseline_step)+'.npy', baseline_unlearn_finetune_acc_list)
            np.save('./result/LMC/'+str(self.args.dataset)+'/sequential2/baseline_k_Ifac_'+str(baseline_I_factor)+'.npy', baseline_k_list)
            print('baseline k:'+str(baseline_k_list))
        
        this_sigma = 0.03
        for num_remove_per_itr in num_remove_per_itr_list:
            lmc_learn_scratch_acc, mean_time, lmc_w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, this_sigma, None, len_list = 1, return_w = True)
            print('LMc learn scratch acc: ' + str(np.mean(lmc_learn_scratch_acc)))
            print('LMc learn scratch acc std: ' + str(np.std(lmc_learn_scratch_acc)))
            np.save('./result/LMC/'+str(self.args.dataset)+'/sequential2/lmc_acc_scratch'+str(num_remove_per_itr)+'.npy', lmc_learn_scratch_acc)
            num_step = int(num_remove_list[0]/num_remove_per_itr)
            alpha_list = []
            self.k_list = np.zeros(num_step+1).astype(int)
            # first get k for step 1 as warm start
            k_1 = 1
            epsilon_of_s1 = lambda alpha: self.epsilon_s1(alpha, k_1, num_remove_per_itr, this_sigma) + (math.log(1 / float(self.delta))) / (alpha - 1)
            min_epsilon_s1_k1 = minimize_scalar(epsilon_of_s1, bounds=(1, 100000), method='bounded')
            while min_epsilon_s1_k1.fun > target_epsilon:
                k_1 = k_1 + 1
                epsilon_of_s1 = lambda alpha: self.epsilon_s1(alpha, k_1, num_remove_per_itr, this_sigma) + (math.log(1 / float(self.delta))) / (alpha - 1)
                min_epsilon_s1_k1 = minimize_scalar(epsilon_of_s1, bounds=(1, 100000), method='bounded')
            # set k_1 in the list
            self.k_list[1] = k_1
            alpha_list.append(min_epsilon_s1_k1.x)

            if num_step > 1:
                for step in tqdm(range(2,num_step+1)):
                    # here step start from 1. k_list[0] = 0 always
                    self.k_list[step] = 1
                    epsilon_of_sstep = lambda alpha: self.epsilon_s_with_alpha(alpha, num_remove_per_itr, this_sigma, step) + (math.log(1 / float(self.delta))) / (alpha - 1)
                    min_epsilon_sstep_kstep = minimize_scalar(epsilon_of_sstep, bounds=(1, 100000), method='bounded')
                    while min_epsilon_sstep_kstep.fun > target_epsilon:
                        self.k_list[step] = self.k_list[step] + 1
                        epsilon_of_sstep = lambda alpha: self.epsilon_s_with_alpha(alpha, num_remove_per_itr, this_sigma, step) + (math.log(1 / float(self.delta))) / (alpha - 1)
                        min_epsilon_sstep_kstep = minimize_scalar(epsilon_of_sstep, bounds=(1, 100000), method='bounded')
            np.save('./result/LMC/'+str(self.args.dataset)+'/sequential2/'+'k_list_nr'+str(num_remove_per_itr)+'.npy', self.k_list)
            print('lmc nr:'+str(num_remove_per_itr)+'k_list:'+str(self.k_list))
            # accumulate k
            accumulate_k = np.cumsum(self.k_list)
            print(accumulate_k)
            for lmc_step, lmc_k in enumerate(self.k_list):
                X_train_removed, y_train_removed = self.get_removed_data(int((lmc_step+1)*num_remove_per_itr))
                lmc_unlearn_finetune_acc, mean_time, lmc_w_list = self.get_mean_performance(X_train_removed, y_train_removed, lmc_k, this_sigma, lmc_w_list, len_list = 1, return_w = True)
                print('LMc unlearn finetune acc: ' + str(np.mean(lmc_unlearn_finetune_acc)))
                print('LMc unlearn finetune acc std: ' + str(np.std(lmc_unlearn_finetune_acc)))
                np.save('./result/LMC/'+str(self.args.dataset)+'/sequential2/lmc_acc_finetune_nr'+str(num_remove_per_itr)+'_step'+str(lmc_step)+'.npy', lmc_unlearn_finetune_acc)


    def get_baseline_term2(self, i):
        gamma = (self.L - self.m) / (self.L + self.m)
        term2 = math.log(math.log((4 * self.dim_w * i) / self.delta)) / (math.log(1 / gamma))
        return math.ceil(term2)
    def get_baseline_min_I(self, target_epsilon):
        gamma = (self.L - self.m) / (self.L + self.m)
        part_1 = math.sqrt(2 * self.dim_w) * (1-gamma)**(-1)
        part_2 = math.sqrt(2 * math.log(2 / self.delta) + target_epsilon) - math.sqrt(2 * math.log(2 / self.delta))
        numerator = math.log(part_1 / part_2)
        dominator = math.log( 1 / gamma)
        return math.ceil(numerator / dominator)
    def calculate_baseline_sigma2(self, I, epsilon):
        gamma = (self.L - self.m) / (self.L + self.m)
        numerator = 8 * self.M * gamma**I
        part_1 = self.m * self.n * (1 - gamma**I)
        part_2 = math.sqrt(2 * math.log(2 / self.delta) + 3 * epsilon) - math.sqrt(2 * math.log(2 / self.delta) + 2 * epsilon)
        dominator = part_1 * part_2
        return numerator / dominator

    def sequential(self):
        num_remove_list = [1000]
        num_remove_per_itr_list = [1000]
        target_epsilon = 1
        create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/sequential/')

        baseline_step_size = 2 / (self.L + self.m)

        baseline_k_list = [1, 4, 5]
        for baseline_k in baseline_k_list:
            # first run the baseline, sequentially delete
            baseline_learn_scratch_acc, mean_time, baseline_w_list = self.get_mean_baseline(self.X_train, self.y_train, baseline_step_size, self.args.burn_in, None, len_list = 1, return_w = True)
            np.save('./result/LMC/'+str(self.args.dataset)+'/sequential/baseline_acc_scratch'+str(baseline_k)+'.npy', baseline_learn_scratch_acc)
            print('baseline learn scratch acc: ' + str(np.mean(baseline_learn_scratch_acc)))
            print('baseline learn scratch acc std: ' + str(np.std(baseline_learn_scratch_acc)))

            for baseline_step in range(num_remove_list[0]):
                X_train_removed, y_train_removed = self.get_removed_data(int((baseline_step+1)))
                _, mean_time, baseline_w_list = self.get_mean_baseline(X_train_removed, y_train_removed, baseline_step_size, baseline_k, baseline_w_list, len_list = 1, return_w = True)
                baseline_sigma = self.calculate_baseline_sigma(baseline_k, epsilon = target_epsilon)
                print('baseline sigma: ' + str(baseline_sigma))
                random_noise = np.random.normal(0, baseline_sigma, (100, 1, self.dim_w))
                w_list_new_totest = torch.tensor(baseline_w_list + random_noise).float()
                baseline_unlearn_finetune_acc_list = []
                for i in range(100):
                    accuracy = self.test_accuracy(w_list_new_totest[i])
                    baseline_unlearn_finetune_acc_list.append(accuracy)
                np.save('./result/LMC/'+str(self.args.dataset)+'/sequential/baseline_acc_k_'+str(baseline_k)+'_step_'+str(baseline_step)+'.npy', baseline_unlearn_finetune_acc_list)

        for num_remove_per_itr in num_remove_per_itr_list:
            lmc_learn_scratch_acc, mean_time, lmc_w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
            print('LMc learn scratch acc: ' + str(np.mean(lmc_learn_scratch_acc)))
            print('LMc learn scratch acc std: ' + str(np.std(lmc_learn_scratch_acc)))
            np.save('./result/LMC/'+str(self.args.dataset)+'/sequential/lmc_acc_scratch'+str(num_remove_per_itr)+'.npy', lmc_learn_scratch_acc)
            num_step = int(num_remove_list[0]/num_remove_per_itr)
            alpha_list = []
            self.k_list = np.zeros(num_step+1).astype(int)
            # first get k for step 1 as warm start
            k_1 = 1
            epsilon_of_s1 = lambda alpha: self.epsilon_s1(alpha, k_1, num_remove_per_itr, self.args.sigma)
            min_epsilon_s1_k1 = minimize_scalar(epsilon_of_s1, bounds=(1, 100000), method='bounded')
            while min_epsilon_s1_k1.fun > target_epsilon:
                k_1 = k_1 + 1
                epsilon_of_s1 = lambda alpha: self.epsilon_s1(alpha, k_1, num_remove_per_itr, self.args.sigma)
                min_epsilon_s1_k1 = minimize_scalar(epsilon_of_s1, bounds=(1, 100000), method='bounded')
            # set k_1 in the list
            self.k_list[1] = k_1
            alpha_list.append(min_epsilon_s1_k1.x)

            if num_step > 1:
                for step in tqdm(range(2,num_step+1)):
                    # here step start from 1. k_list[0] = 0 always
                    self.k_list[step] = 1
                    epsilon_of_sstep = lambda alpha: self.epsilon_s_with_alpha(alpha, num_remove_per_itr, self.args.sigma, step)
                    min_epsilon_sstep_kstep = minimize_scalar(epsilon_of_sstep, bounds=(1, 100000), method='bounded')
                    while min_epsilon_sstep_kstep.fun > target_epsilon:
                        self.k_list[step] = self.k_list[step] + 1
                        epsilon_of_sstep = lambda alpha: self.epsilon_s_with_alpha(alpha, num_remove_per_itr, self.args.sigma, step)
                        min_epsilon_sstep_kstep = minimize_scalar(epsilon_of_sstep, bounds=(1, 100000), method='bounded')
            np.save('./result/LMC/'+str(self.args.dataset)+'/sequential/'+'k_list_nr'+str(num_remove_per_itr)+'.npy', self.k_list)
            # accumulate k
            accumulate_k = np.cumsum(self.k_list)
            print(accumulate_k)
            for lmc_step, lmc_k in enumerate(self.k_list):
                X_train_removed, y_train_removed = self.get_removed_data(int((lmc_step+1)*num_remove_per_itr))
                lmc_unlearn_finetune_acc, mean_time, lmc_w_list = self.get_mean_performance(X_train_removed, y_train_removed, lmc_k, self.args.sigma, lmc_w_list, len_list = 1, return_w = True)
                print('LMc unlearn finetune acc: ' + str(np.mean(lmc_unlearn_finetune_acc)))
                print('LMc unlearn finetune acc std: ' + str(np.std(lmc_unlearn_finetune_acc)))
                np.save('./result/LMC/'+str(self.args.dataset)+'/sequential/lmc_acc_finetune_nr'+str(num_remove_per_itr)+'_step'+str(lmc_step)+'.npy', lmc_unlearn_finetune_acc)
            
        #sns.lineplot(x=np.arange(num_remove_per_itr, num_remove_list[0]+1, num_remove_per_itr), y=accumulate_k[1:], label='num remove per itr: '+str(num_remove_per_itr), marker = 'o')
        #plt.legend()
        #plt.savefig('./cum_k.pdf')
    def epsilon0_(self, alpha, S, sigma):
        epsilon0 = (4 * alpha * float(S)**2 * float(self.M)**2) / (float(self.m) * float(sigma)**2 * float(self.n)**2)
        return epsilon0
    def epsilon_s1(self, alpha, k, S, sigma):
        epsilon0 = (4 * alpha * S**2 * self.M**2) / (self.m * sigma**2 * self.n**2)
        return math.exp(- (1/alpha) * self.eta * self.m * k) * (self.epsilon0_(alpha, S, sigma))

    def epsilon_s_with_alpha(self, alpha, S, sigma, step):
        # every time call this function, the k_list[step - 1], step > 1 must be greater than 0
        if step == 1:
            # the first step
            epsilon0 = (4 * alpha * S**2 * self.M**2) / (self.m * sigma**2 * self.n**2)
            return math.exp(- (1/alpha) * self.eta * self.m * self.k_list[1]) * (self.epsilon0_(alpha, S, sigma))
        else:
            part1 = math.exp(- (1/alpha) * self.eta * self.m * self.k_list[step])
            step = step - 1
            #part2 = 2 * ((4 * alpha * S**2 * self.M**2) / (self.m * sigma**2 * self.n**2)) * (alpha - 0.5) + (alpha - 0.5) / (alpha - 1) * self.epsilon_s_with_alpha(2*alpha, S, sigma, step)
            part2 = self.epsilon0_(2*alpha - 1, S, sigma) + (alpha - 0.5) / (alpha - 1) * self.epsilon_s_with_alpha(2*alpha, S, sigma, step)
            return part1 * part2

    def train(self):
        if self.args.run_baseline:
            epsilon_list = [0.05, 0.1, 0.5, 1, 2, 5]
            target_k_list = [1, 2, 5]

            baseline_step_size = 2 / (self.L + self.m)
            
            X_train_removed, y_train_removed = self.get_removed_data(1)
            baseline_learn_scratch_acc, mean_time, baseline_w_list = self.get_mean_baseline(self.X_train, self.y_train, baseline_step_size, self.args.burn_in, None, len_list = 1, return_w = True)
            np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/baseline_acc_scratch.npy', baseline_learn_scratch_acc)
            print('baseline learn scratch acc: ' + str(np.mean(baseline_learn_scratch_acc)))
            print('baseline learn scratch acc std: ' + str(np.std(baseline_learn_scratch_acc)))
            baseline_unlearn_scratch_acc, mean_time = self.get_mean_baseline(X_train_removed, y_train_removed, baseline_step_size, self.args.burn_in, None, len_list = 1)
            np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/baseline_acc_unlearn_scratch.npy', baseline_learn_scratch_acc)
            print('baseline unlearn scratch acc: ' + str(np.mean(baseline_unlearn_scratch_acc)))
            print('baseline unlearn scratch acc std: ' + str(np.std(baseline_unlearn_scratch_acc)))
            import pdb; pdb.set_trace()
            for target_k in target_k_list:
                print('working on target k:'+str(target_k))
                # first run algorithm #1 to learn and get parameters
                _, mean_time, w_list_new = self.get_mean_baseline(X_train_removed, y_train_removed, baseline_step_size, target_k, baseline_w_list, len_list = 1, return_w = True)
                create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/')
                for epsilon in epsilon_list:
                    baseline_sigma = self.calculate_baseline_sigma(target_k, epsilon = epsilon)
                    print('baseline sigma: ' + str(baseline_sigma))
                    random_noise = np.random.normal(0, baseline_sigma, (100, 1, self.dim_w))
                    w_list_new_totest = torch.tensor(w_list_new + random_noise).float()
                    baseline_unlearn_finetune_acc_list = []
                    for i in range(100):
                        accuracy = self.test_accuracy(w_list_new_totest[i])
                        baseline_unlearn_finetune_acc_list.append(accuracy)
                    print('baseline unlearn finetune acc: ' + str(np.mean(baseline_unlearn_finetune_acc_list)))
                    print('baseline unlearn finetune acc std: ' + str(np.std(baseline_unlearn_finetune_acc_list)))
                    np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/baseline_acc_unlearn_finetune'+str(epsilon)+'.npy', baseline_unlearn_finetune_acc_list)

                # run langevin unlearning
                num_remove_list = [1]
                if target_k == 1:
                    if self.args.dataset == 'MNIST':
                        sigma_list = [0.1872, 0.094, 0.00019, 0.0096, 0.0049, 0.0021] # sigma list for MNIST
                    elif self.args.dataset == 'CIFAR10':
                        sigma_list = [0.2431, 0.122, 0.025, 0.000125, 0.0064, 0.0028] # sigma list for CIFAR10
                elif target_k == 2:
                    if self.args.dataset == 'MNIST':
                        sigma_list = [0.18714501953125, 0.09368591346136476, 0.00018914795795776367, 0.00956726167840576, 0.004888916983032227, 0.0020690927830810547]
                    elif self.args.dataset == 'CIFAR10':
                        sigma_list = [0.24309802246093754, 0.12169189471997069, 0.02457275474243164, 0.00012432862245239257, 0.006353760723266601, 0.002700806646057129]
                elif target_k == 5:
                    if self.args.dataset == 'MNIST':
                        sigma_list = [0.18709939575195314, 0.09364013709448243, 0.00018869019428894046, 0.009521485311523437, 0.004852295889526367, 0.002032471689575195]
                    elif self.args.dataset == 'CIFAR10':
                        sigma_list = [0.24304327392578123, 0.1216369630797119, 0.024517823102172855, 0.00012377930604980467, 0.006317139629760741, 0.002655030279174805]
                else:
                    # for unseen target k, search it here
                    sigma_list = []
                    for epsilon in epsilon_list:
                        sigma_list.append(self.search_k(target_k, epsilon))
                    print(sigma_list)

                for epsilon, sigma in zip(epsilon_list, sigma_list):
                    print('epsilon: ' + str(epsilon))
                    self.args.sigma = sigma
                    K_dict, _ = self.search_finetune_step(epsilon_list, num_remove_list, self.args.sigma)
                    lmc_learn_scratch_acc, mean_time, lmc_w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
                    print('LMc learn scratch acc: ' + str(np.mean(lmc_learn_scratch_acc)))
                    print('LMc learn scratch acc std: ' + str(np.std(lmc_learn_scratch_acc)))
                    np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/lmc_acc_learn_scratch'+str(epsilon)+'.npy', lmc_learn_scratch_acc)
                    lmc_unlearn_scratch_acc, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, self.args.sigma, None, len_list = 1)
                    print('LMc unlearn scratch acc: ' + str(np.mean(lmc_unlearn_scratch_acc)))
                    print('LMc unlearn scratch acc std: ' + str(np.std(lmc_unlearn_scratch_acc)))
                    np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/lmc_acc_unlearn_scratch'+str(epsilon)+'.npy', lmc_unlearn_scratch_acc)
                    lmc_unlearn_finetune_acc, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, int(K_dict[num_remove_list[0]][epsilon]), self.args.sigma, lmc_w_list, len_list = 1)
                    print('LMc unlearn finetune acc: ' + str(np.mean(lmc_unlearn_finetune_acc)))
                    print('LMc unlearn finetune acc std: ' + str(np.std(lmc_unlearn_finetune_acc)))
                    np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/lmc_acc_unlearn_finetune'+str(epsilon)+'.npy', lmc_unlearn_finetune_acc)
            import pdb; pdb.set_trace()

    def search_k(self, target, epsilon, lower = 1e-3, upper = 0.30):
        if self.calculate_k_with_sigma(epsilon, lower) < target or self.calculate_k_with_sigma(epsilon, upper) > target:
            print('not good upper lowers')
            return
        while lower <= upper:
            mid = (lower + upper) / 2
            k = self.calculate_k_with_sigma(epsilon, mid)
            if k == target:
                print('find sigma for '+str(epsilon)+'k '+str(target))
                print(mid)
                return mid
            elif k < target:
                upper = mid
            else:
                lower = mid
            '''print(lower)
            print(upper)
            print(k)'''
            

    def calculate_k_with_sigma(self, epsilon, sigma):
        num_remove_list = [1]
        epsilon_list = [epsilon]
        K_dict, _ = self.search_finetune_step(epsilon_list, num_remove_list, sigma)
        return(K_dict[1][epsilon])
    
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
        #avg_accuracy = np.mean(trial_list)
        mean_time = np.mean(time_list)

        if return_w:
            new_w_list = np.stack(new_w_list, axis=0)
            return trial_list, mean_time, new_w_list
        else:
            return trial_list, mean_time
    
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
        #avg_accuracy = np.mean(trial_list)
        mean_time = np.mean(time_list)

        if return_w:
            new_w_list = np.stack(new_w_list, axis=0)
            return trial_list, mean_time, new_w_list
        else:
            return trial_list, mean_time
    
    def run_gradient_descent(self, init_point, X, y, baseline_step_size, burn_in, len_list):
        start_time = time.time()
        w_list = self.gradient_descent_algorithm(init_point, self.dim_w, X, y, self.args.lam*self.n, device = self.device, potential = logistic_potential, burn_in = burn_in, len_list = len_list, step=baseline_step_size, M = self.M)
        end_time = time.time()
        return w_list, end_time - start_time

    def gradient_descent_algorithm(self, init_point, dim_w, X, y, lam, device, potential, burn_in = 10000, len_list = 1, step=0.1, M = 1):
        # randomly sample from N(0, I)
        if init_point == None:
            #w0 = torch.randn(dim_w).to(device)
            w0 = torch.normal(mean=1000, std=1, size=(dim_w,)).reshape(-1).to(device)
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
        
    
    def calculate_baseline_sigma(self, I, epsilon = 1):
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
                    K = K + 1
                    epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                    min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                K_list[target_epsilon] = K
                alpha_list[target_epsilon] = min_epsilon_with_k.x
                #print('num remove:'+str(num_remove)+'target epsilon: '+str(target_epsilon)+'K: '+str(K)+'alpha: '+str(min_epsilon_with_k.x))
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
        w_list = unadjusted_langevin_algorithm(init_point, self.dim_w, X, y, self.args.lam*self.n, sigma = sigma, 
                                               device = self.device, potential = logistic_potential, burn_in = burn_in, 
                                               len_list = len_list, step=self.eta, M = self.M, m = self.m)
        end_time = time.time()
        return w_list, end_time - start_time
    
    def try__(self):
        self.k_list = np.zeros(21).astype(int)
        num_remove_per_itr = 5
        target_epsilon = 1
        # first get k for step 1 as warm start
        k_1 = 1
        epsilon_of_s1 = lambda alpha: self.epsilon_s1(alpha, k_1, num_remove_per_itr, 0.03) + (math.log(1 / float(self.delta))) / (alpha - 1)
        min_epsilon_s1_k1 = minimize_scalar(epsilon_of_s1, bounds=(1, 100000), method='bounded')
        while min_epsilon_s1_k1.fun > target_epsilon:
            k_1 = k_1 + 1
            epsilon_of_s1 = lambda alpha: self.epsilon_s1(alpha, k_1, num_remove_per_itr, 0.03) + (math.log(1 / float(self.delta))) / (alpha - 1)
            min_epsilon_s1_k1 = minimize_scalar(epsilon_of_s1, bounds=(1, 100000), method='bounded')
        # set k_1 in the list
        self.k_list[1] = k_1
        for step in range(2, 21):
            self.k_list[step] = 1
            epsilon_of_sstep = lambda alpha: self.epsilon_s_with_alpha(alpha, num_remove_per_itr, 0.03, step) + (math.log(1 / float(self.delta))) / (alpha - 1)
            min_epsilon_sstep_kstep = minimize_scalar(epsilon_of_sstep, bounds=(1, 100000), method='bounded')
            while min_epsilon_sstep_kstep.fun > target_epsilon:
                self.k_list[step] = self.k_list[step] + 1
                epsilon_of_sstep = lambda alpha: self.epsilon_s_with_alpha(alpha, num_remove_per_itr, 0.03, step) + (math.log(1 / float(self.delta))) / (alpha - 1)
                min_epsilon_sstep_kstep = minimize_scalar(epsilon_of_sstep, bounds=(1, 100000), method='bounded')
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
    parser.add_argument('--sequential2', type = int, default = 0, help = 'new baseline bound')
    parser.add_argument('--find_k', type = int, default = 0, help = 'find the k')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()
    #runner.try__()
    #import pdb; pdb.set_trace()
    

    # here requires to find sigma by hand
    #runner.find_sigma()
    if args.sequential == 1:
        runner.sequential()
    elif args.find_k == 1:
        #import pdb; pdb.set_trace()
        target_k_list = [1, 2, 5]
        epsilon_list = [0.05, 0.1, 0.5, 1, 2, 5]
        for target_k in target_k_list:
            result_list = []
            for epsilon in epsilon_list:
                result_sigma = runner.search_k(target_k, epsilon)
                result_list.append(result_sigma)
            print('target k:'+str(target_k))
            print(result_list)
    elif args.sequential2 == 1:
        runner.sequential2()

    else:
        runner.train()
    


if __name__ == '__main__':
    main()