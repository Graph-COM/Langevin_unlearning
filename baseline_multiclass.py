from hmac import new
import time
import numpy as np
import argparse
from tqdm import tqdm
import math
from scipy.optimize import minimize_scalar

import torch
import torch.nn.functional as F

from utils import load_features, create_nested_folder, transform_array
from langevin import unadjusted_langevin_algorithm_multiclass


class Runner():
    def __init__(self, args):
        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.args = args
        if args.dataset == 'CIFAR10_MULTI':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.dim_w = 512
            self.num_class = 10
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
        self.L = 1 + self.args.lam * self.n
        self.L = self.L
        print('L smooth constant'+str(self.L))
        # m-strongly convex constant
        self.m = self.args.lam * self.n
        self.m = self.m
        print('m strongly convex:'+str(self.m))
        # M-Lipschitz constant
        self.M = 2
        print('M lipschitz constant:'+str(self.M))
        # calculate step size
        self.eta = 1 / self.L
        print('step size eta:'+str(self.eta))
        # calculate RDP delta
        self.delta = 1 / self.n
        print('RDP constant delta:'+str(self.delta))


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
            
    def epsilon0_(self, alpha, S, sigma):
        epsilon0 = (4 * alpha * float(S)**2 * float(self.M)**2) / (float(self.m) * float(sigma)**2 * float(self.n)**2)
        return epsilon0
    def epsilon_s1(self, alpha, k, S, sigma):
        return math.exp(- (1/alpha) * self.eta * self.m * k) * (self.epsilon0_(alpha, S, sigma))

    def epsilon_s_with_alpha(self, alpha, S, sigma, step):
        # every time call this function, the k_list[step - 1], step > 1 must be greater than 0
        if step == 1:
            # the first step
            return math.exp(- (1/alpha) * self.eta * self.m * self.k_list[1]) * (self.epsilon0_(alpha, S, sigma))
        else:
            part1 = math.exp(- (1/alpha) * self.eta * self.m * self.k_list[step])
            step = step - 1
            part2 = (alpha - 0.5) / (alpha - 1) * (self.epsilon0_(2*alpha, S, sigma) + self.epsilon_s_with_alpha(2*alpha, S, sigma, step))
            return part1 * part2

    def train(self):
        if self.args.run_baseline:
            epsilon_list = [0.05, 0.1, 0.5, 1, 2, 5]
            target_k_list = [1, 2, 5]

            baseline_step_size = 2 / (self.L + self.m)
            
            X_train_removed, y_train_removed = self.get_removed_data(1)
            create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/baseline/')
            baseline_learn_scratch_acc, mean_time, baseline_w_list = self.get_mean_baseline(self.X_train, self.y_train, baseline_step_size, self.args.burn_in, None, len_list = 1, return_w = True)
            np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/baseline_acc_scratch.npy', baseline_learn_scratch_acc)
            print('baseline learn scratch acc: ' + str(np.mean(baseline_learn_scratch_acc)))
            print('baseline learn scratch acc std: ' + str(np.std(baseline_learn_scratch_acc)))
            baseline_unlearn_scratch_acc, mean_time = self.get_mean_baseline(X_train_removed, y_train_removed, baseline_step_size, self.args.burn_in, None, len_list = 1)
            np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/baseline_acc_unlearn_scratch.npy', baseline_learn_scratch_acc)
            print('baseline unlearn scratch acc: ' + str(np.mean(baseline_unlearn_scratch_acc)))
            print('baseline unlearn scratch acc std: ' + str(np.std(baseline_unlearn_scratch_acc)))
            for target_k in target_k_list:
                print('working on target k:'+str(target_k))
                # first run algorithm #1 to learn and get parameters
                _, mean_time, w_list_new = self.get_mean_baseline(X_train_removed, y_train_removed, baseline_step_size, target_k, baseline_w_list, len_list = 1, return_w = True)
                create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/')
                for epsilon in epsilon_list:
                    baseline_sigma = self.calculate_baseline_sigma(target_k, epsilon = epsilon)
                    print('baseline sigma: ' + str(baseline_sigma))
                    random_noise = np.random.normal(0, baseline_sigma, (100, self.dim_w, self.num_class))
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
                    if self.args.dataset == 'CIFAR10_MULTI':
                        sigma_list = [0.0473, 0.0238, 0.0049, 0.0025, 0.00125, 0.00052]
                elif target_k == 2:
                    if self.args.dataset == 'CIFAR10_MULTI':
                        sigma_list = [0.04712375541687011, 0.023586332015991213, 0.004756851043701172, 0.0024040242004394526, 0.0012253220367431641, 0.0005158119964599609]
                elif target_k == 5:
                    if self.args.dataset == 'CIFAR10_MULTI':
                        sigma_list = [0.047116889190673826, 0.023579465789794925, 0.004749984817504882, 0.002394869232177734, 0.001218455810546875, 0.0005112345123291015]
                else:
                    # for unseen target k, search it here
                    sigma_list = []
                    for epsilon in epsilon_list:
                        sigma_list.append(self.search_k(target_k, epsilon))
                    print(sigma_list)
                
                lmc_learn_noiseless_acc, _ = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, 0, None, len_list = 1, return_w = False)
                print('LMc learn noiseless acc: ' + str(np.mean(lmc_learn_noiseless_acc)))
                print('LMc learn noiseless acc std: ' + str(np.std(lmc_learn_noiseless_acc)))
                np.save('./result/LMC/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/lmc_acc_noiseless.npy', lmc_learn_noiseless_acc)
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

    def search_k(self, target, epsilon, lower = 1e-5, upper = 0.30):
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
                w = w_list[trial_idx]
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
                w_init, time = self.run_unadjusted_langvin_multiclass(None, X, y, step, sigma, len_list)
                time_list.append(time)
                w_init = np.vstack(w_init)
                new_w_list.append(w_init)
                accuracy = self.test_accuracy(w_init)
                trial_list.append(accuracy)
        else:
            for trial_idx in tqdm(range(num_trial)):
                w = w_list[trial_idx]
                w = torch.tensor(w)
                new_w, time = self.run_unadjusted_langvin_multiclass(w, X, y, step, sigma, len_list = 1)
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
        w_list = self.gradient_descent_algorithm(init_point, self.dim_w, X, y, self.args.lam*self.n, device = self.device, burn_in = burn_in, len_list = len_list, step=baseline_step_size, M = self.M)
        end_time = time.time()
        return w_list, end_time - start_time

    def gradient_descent_algorithm(self, init_point, dim_w, X, y, lam, device, burn_in = 10000, len_list = 1, step=0.1, M = 1):
        # randomly sample from N(0, I)
        if init_point == None:
            #w0 = torch.randn(dim_w).to(device)
            w0 = torch.normal(mean=1000, std=1, size=(dim_w, self.num_class)).to(device)
        else:
            w0 = init_point.to(device)
        wi = w0
        samples = []
        for i in range(len_list + burn_in):
            pre_log_softmax = torch.matmul(X, wi)
            pred_log = F.softmax(pre_log_softmax, dim = -1)
            per_sample_grad=torch.bmm(X.unsqueeze(-1), (pred_log - y).unsqueeze(1))
            row_norms = torch.norm(per_sample_grad,dim=(1, 2))
            clipped_grad = (per_sample_grad / row_norms.unsqueeze(-1).unsqueeze(-1)) * M
            clipped_grad[row_norms <= M] = per_sample_grad[row_norms <= M]
            grad_1 = clipped_grad.mean(0)
            grad_2 = lam * wi
            wi = wi.detach() - step * (grad_1 + grad_2)
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
        new_y_train = torch.randint(0, 10, (1, num_remove)).reshape(-1)
        new_y_train = transform_array(new_y_train)
        new_y_train = new_y_train.to(self.device)
        X_train_removed = torch.cat((X_train_removed, new_X_train), 0)
        y_train_removed = torch.cat((y_train_removed, new_y_train), 0)
        return X_train_removed, y_train_removed
        
    
    def calculate_baseline_sigma(self, I, epsilon = 1):
        # calculate the noise for descent to delete
        gamma = (self.L - self.m) / (self.L + self.m)
        numerator = 4 * math.sqrt(2) * self.M * gamma**I
        dominator = self.m * self.n * (1 - gamma**I) * (math.sqrt(math.log(1 / self.delta) + epsilon) - math.sqrt(math.log(1 / self.delta)))
        return numerator / dominator
    
    def epsilon_expression(self, K, sigma, eta, C_lsi, alpha, S, M, m, n, delta):
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
                min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(2, 10000), method='bounded')
                while min_epsilon_with_k.fun > target_epsilon:
                    K = K + 1
                    epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                    min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(2, 10000), method='bounded')
                K_list[target_epsilon] = K
                alpha_list[target_epsilon] = min_epsilon_with_k.x
            K_dict[num_remove] = K_list
            alpha_dict[num_remove] = alpha_list
        return K_dict, alpha_dict

    def test_accuracy(self, w_list):
        w = torch.tensor(w_list)
        # test accuracy (before removal)
        pred = self.X_test.matmul(w)
        score = F.softmax(pred, dim = -1)
        _, pred_class = score.max(dim = 1)
        _, label = self.y_test.max(dim = 1)
        correct_predictions = (pred_class == label).sum().item()
        accuracy = correct_predictions / score.shape[0]
        return accuracy
    def run_unadjusted_langvin_multiclass(self, init_point, X, y, burn_in, sigma, len_list):
        start_time = time.time()
        w_list = unadjusted_langevin_algorithm_multiclass(init_point, self.dim_w, X, y, self.args.lam*self.n, sigma = sigma, num_class = self.num_class,
                                               device = self.device, burn_in = burn_in, 
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
        min_epsilon_s1_k1 = minimize_scalar(epsilon_of_s1, bounds=(2, 100000), method='bounded')
        while min_epsilon_s1_k1.fun > target_epsilon:
            k_1 = k_1 + 1
            epsilon_of_s1 = lambda alpha: self.epsilon_s1(alpha, k_1, num_remove_per_itr, 0.03) + (math.log(1 / float(self.delta))) / (alpha - 1)
            min_epsilon_s1_k1 = minimize_scalar(epsilon_of_s1, bounds=(2, 100000), method='bounded')
        # set k_1 in the list
        self.k_list[1] = k_1
        for step in range(2, 21):
            self.k_list[step] = 1
            epsilon_of_sstep = lambda alpha: self.epsilon_s_with_alpha(alpha, num_remove_per_itr, 0.03, step) + (math.log(1 / float(self.delta))) / (alpha - 1)
            min_epsilon_sstep_kstep = minimize_scalar(epsilon_of_sstep, bounds=(2, 100000), method='bounded')
            while min_epsilon_sstep_kstep.fun > target_epsilon:
                self.k_list[step] = self.k_list[step] + 1
                epsilon_of_sstep = lambda alpha: self.epsilon_s_with_alpha(alpha, num_remove_per_itr, 0.03, step) + (math.log(1 / float(self.delta))) / (alpha - 1)
                min_epsilon_sstep_kstep = minimize_scalar(epsilon_of_sstep, bounds=(2, 100000), method='bounded')
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
    parser.add_argument('--sequential2', type = int, default = 0, help = 'whether test sequential unlearning')
    parser.add_argument('--find_k', type = int, default = 0, help = 'find the k')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()
    if args.find_k == 1:
        # here may require to find sigma by hand
        # please use runner.try__() to manually search for better sigma when k = 1
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