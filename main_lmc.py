from hmac import new
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy.optimize import minimize_scalar

import torch

from utils import load_features, generate_gaussian, create_nested_folder
from langevin import unadjusted_langevin_algorithm


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
        elif args.dataset == 'ADULT':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.dim_w = 6
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
        self.L = 1 / 4 + self.args.lam * self.n
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
        self.eta = 1 / self.L
        print('step size eta:'+str(self.eta))
        # calculate RDP delta
        self.delta = 1 / self.n
        print('RDP constant delta:'+str(self.delta))
        
    def train(self):
        if self.args.paint_utility_s:
            # this is to paint the utilisy - s figure
            num_remove_list = [1, 10, 50, 100, 500, 1000] # the number of data to remove
            create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/')
            accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/learn_scratch_w.npy', w_list)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/acc_scratch_D.npy', accuracy_scratch_D)
            # calculate K
            epsilon_list = [0.5, 1, 2]
            K_dict, _ = self.search_finetune_step(self.args.sigma, epsilon_list, num_remove_list)
            for epsilon_idx, epsilon in enumerate(epsilon_list):
                K_list = []
                for num_remove in num_remove_list:
                    create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/'+str(epsilon)+'/')
                    X_train_removed, y_train_removed = self.get_removed_data(num_remove)
                    accuracy_scratch_Dnew, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, self.args.sigma, None)
                    np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/'+str(epsilon)+'/acc_scratch_Dnew_remove'+str(num_remove)+'.npy', accuracy_scratch_Dnew)
                    accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove][epsilon], self.args.sigma, w_list)
                    np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/'+str(epsilon)+'/acc_finetune_remove'+str(num_remove)+'.npy', accuracy_finetune)
                    K_list.append(K_dict[num_remove][epsilon])
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/'+str(epsilon)+'/K_list.npy', K_list)
        elif self.args.paint_utility_epsilon:
            epsilon_list = [0.1, 0.5, 1, 2, 5]
            num_remove_list = [1, 50, 100]
            accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/w_from_scratch.npy', w_list)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/acc_scratch_D.npy', accuracy_scratch_D)
            # calculate K
            K_dict, _ = self.search_finetune_step(self.args.sigma, epsilon_list, num_remove_list)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/K_list.npy', K_dict)
            for remove_idx, num_remove in enumerate(num_remove_list):
                K_list = []
                for epsilon in epsilon_list:
                    X_train_removed, y_train_removed = self.get_removed_data(num_remove_list[remove_idx])
                    accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[remove_idx]][epsilon], self.args.sigma, w_list)
                    create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/'+str(num_remove)+'/')
                    np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/'+str(num_remove)+'/acc_finetune_epsilon'+str(epsilon)+'.npy', accuracy_finetune)
                    K_list.append(K_dict[num_remove_list[0]][epsilon])
        elif self.args.paint_unlearning_sigma:
            num_remove_list = [100]
            epsilon_list = [1]
            
            #sigma_list = [0.05, 0.1, 0.2, 0.5, 1]
            sigma_list =[0.01]
            scratch_acc_list = []
            scratch_unlearn_list = []
            finetune_unlearn_list = []
            epsilon0_list = []
            X_train_removed, y_train_removed = self.get_removed_data(num_remove_list[0])
            for sigma in sigma_list:
                K_dict, alpha_dict = self.search_finetune_step(sigma, epsilon_list, num_remove_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/K_dict'+str(sigma)+'.npy', K_dict)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/alpha_dict'+str(sigma)+'.npy', alpha_dict)
                alpha = alpha_dict[num_remove_list[0]][epsilon_list[0]]
                DP_epsilon0_expression = lambda alpha_: self.calculate_epsilon0(alpha_, num_remove_list[0], sigma) + math.log(float(1/self.delta)) / (alpha_ - 1)
                DP_epsilon0 = minimize_scalar(DP_epsilon0_expression, bounds=(1, 10000), method='bounded')
                #epsilon0 = self.calculate_epsilon0(alpha, num_remove_list[0], sigma)
                epsilon0_list.append(DP_epsilon0.fun)
                accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, sigma, None, len_list = 1, return_w = True)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_learn_scratch_w.npy', w_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_scratch_D.npy', accuracy_scratch_D)
                accuracy_scratch_Dnew, mean_time, unlearn_w_list = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, sigma, None, return_w=True)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_unlearn_scratch_w.npy', unlearn_w_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_scratch_Dnew.npy', accuracy_scratch_Dnew)
                accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[0]][1], sigma, w_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_finetune.npy', accuracy_finetune)
            #np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/epsilon0.npy', epsilon0_list)
            print(epsilon0_list)
        elif self.args.retrain_noiseless == 1:
            num_remove_list = [1, 10, 50, 100, 500, 1000] # the number of data to remove
            for num_remove in num_remove_list:
                create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/retrain_noiseless/')
                X_train_removed, y_train_removed = self.get_removed_data(num_remove)
                accuracy_scratch_Dnew, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, 0, None)
                np.save('./result/LMC/'+str(self.args.dataset)+'/retrain_noiseless/retrain_noiseless'+str(num_remove)+'.npy', accuracy_scratch_Dnew)
        else:
            print('check!')

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
        part_1 = math.exp(- (float(K) * m * float(eta)) / (alpha))
        part_2 = (4 * alpha * float(S)**2 * float(M)**2) / (float(m) * float(sigma)**2 * float(n)**2)
        part_3 = (math.log(1 / float(delta))) / (alpha - 1)
        epsilon = part_1 * part_2 + part_3
        return epsilon
    
    def search_finetune_step(self, sigma, epsilon_list, num_remove_list):
        C_lsi = 2 * self.args.sigma**2 / self.m
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
    def calculate_epsilon0(self, alpha, S, sigma):
        return (4 * alpha * float(S)**2 * float(self.M)**2) / (float(self.m) * float(sigma)**2 * float(self.n)**2)

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
        mean_time = np.mean(time_list)

        if return_w:
            new_w_list = np.stack(new_w_list, axis=0)
            return trial_list, mean_time, new_w_list
        else:
            return trial_list, mean_time
                
    def test_accuracy(self, w_list):
        w = torch.tensor(w_list[0])
        # test accuracy (before removal)
        pred = self.X_test.mv(w)
        accuracy = pred.gt(0).eq(self.y_test.gt(0)).float().mean()
        return accuracy
    def run_unadjusted_langvin(self, init_point, X, y, burn_in, sigma, len_list, projection = 0, batch_size = 0):
        start_time = time.time()
        w_list = unadjusted_langevin_algorithm(init_point, self.dim_w, X, y, self.args.lam*self.n, sigma = sigma, 
                                               device = self.device, burn_in = burn_in, 
                                               len_list = len_list, step=self.eta, M = self.M, m = self.m)
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
    parser.add_argument('--sigma', type = float, default = 0.03, help = 'the parameter sigma')
    parser.add_argument('--burn_in', type = int, default = 10000, help = 'burn in step number of LMC')
    parser.add_argument('--gaussian_dim', type = int, default = 10, help = 'dimension of gaussian task')
    parser.add_argument('--len_list', type = int, default = 10000, help = 'length of w to paint in 2D gaussian')
    parser.add_argument('--finetune_step', type = int, default = 50, help = 'steps to finetune on the new removed data')

    parser.add_argument('--paint_utility_s', type = int, default = 0, help = 'paint the utility - s figure')
    parser.add_argument('--paint_utility_epsilon', type = int, default = 0, help = 'paint utility - epsilon figure')
    parser.add_argument('--paint_unlearning_sigma', type = int, default = 0, help = 'paint unlearning utility - sigma figure')
    parser.add_argument('--retrain_noiseless', type = int, default = 0, help = 'get the noiseless retrain result')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()

    runner.train()

if __name__ == '__main__':
    main()