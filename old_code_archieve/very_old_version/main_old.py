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


def get_accuracy(init_point, dim_w, X_train, y_train, X_test, y_test, device, args):
    if init_point == None:
        steps_to_sample = args.burn_in
    else:
        steps_to_sample = args.finetune_step
    start_time = time.time()
    w_list = unadjusted_langevin_algorithm(init_point, dim_w, X_train, y_train, args.lam, temp = args.temp, device = device, potential = logistic_potential, burn_in = steps_to_sample, len_list = 1, step=args.step_size, Sf = 1)
    end_time = time.time()
    w = torch.tensor(w_list[0])
    # test accuracy (before removal)
    pred = X_test.mv(w)
    accuracy = pred.gt(0).eq(y_test.gt(0)).float().mean()
    print('the accuracy before removal is: '+str(accuracy.item()))
    print('sampling from scratch time cost:'+str(end_time - start_time))

    if args.dataset == '2dgaussian':
        # try visualize the distribution and sample list:
        w_list = unadjusted_langevin_algorithm(init_point, dim_w, X_train, y_train, args.lam, temp = args.temp, device = device, potential = logistic_potential, burn_in = steps_to_sample, len_list = args.len_list, step=args.step_size, Sf = 1)
        if init_point == None:
            plot_2dgaussian(logistic_density, X_train, y_train, args, '2sgaussian_init_gt')
            plot_w_2dgaussian(w_list, '2dgaussian_init_w')
        else:
            plot_2dgaussian(logistic_density, X_train, y_train, args, '2dgaussian_removed_gt')
            plot_w_2dgaussian(w_list, '2dgaussian_removed_w')

    return accuracy, w





def main():
    parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--result-dir', type=str, default='./result', help='directory for saving results')
    parser.add_argument('--extractor', type=str, default='raw_feature', help='extractor type')
    parser.add_argument('--dataset', type=str, default='MNIST', help='[MNIST, 2dgaussian, kdgaussian]')
    parser.add_argument('--lam', type=float, default=1e-8, help='L2 regularization')
    parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
    #parser.add_argument('--train-splits', type=int, default=1, help='number of training data splits')
    #parser.add_argument('--subsample-ratio', type=float, default=1.0, help='negative example subsample ratio')
    parser.add_argument('--num-steps', type=int, default=10000, help='number of optimization steps')
    parser.add_argument('--train-mode', type=str, default='binary', help='train mode [ovr/binary]')
    #parser.add_argument('--train-sep', action='store_true', default=False, help='train binary classifiers separately')
    #parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')

    parser.add_argument('--gpu', type = int, default = 6, help = 'gpu')
    parser.add_argument('--temp', type = float, default = 1000, help = 'the temperature tau_0')
    parser.add_argument('--burn_in', type = int, default = 500, help = 'burn in step number of LMC')
    parser.add_argument('--step_size', type = float, default = 0.1, help = 'the step size of LMC')
    parser.add_argument('--gaussian_dim', type = int, default = 10, help = 'dimension of gaussian task')
    parser.add_argument('--len_list', type = int, default = 10000, help = 'length of w to paint in 2D gaussian')
    parser.add_argument('--finetune_step', type = int, default = 50, help = 'steps to finetune on the new removed data')
    parser.add_argument('--search_burnin', type = int, default = 0, help = 'whether grid search to paint for burn-in')
    parser.add_argument('--search_finetune', type = int, default = 0, help = 'whether to grid search finetune')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'MNIST':
        X_train, X_test, y_train, y_train_onehot, y_test = load_features(args)
    elif args.dataset == '2dgaussian':
        mean_1 = torch.tensor([-2, -2])
        mean_2 = torch.tensor([2, 2])
        std = torch.tensor([1, 1])
        X_train, y_train = generate_gaussian(2, 10000, mean_1, mean_2, std)
        X_test, y_test = generate_gaussian(2, 1000, mean_1, mean_2, std)
    elif args.dataset == 'kdgaussian':
        mean_1 = torch.ones(args.gaussian_dim) * -2
        mean_2 = torch.ones(args.gaussian_dim) * 2
        std = torch.ones(args.gaussian_dim)
        X_train, y_train = generate_gaussian(args.gaussian_dim, 10000, mean_1, mean_2, std)
        X_test, y_test = generate_gaussian(args.gaussian_dim, 1000, mean_1, mean_2, std)
        
    # first sample a set of parameter w as initialization
    if args.train_mode == 'binary':
        if args.dataset == 'MNIST':
            dim_w = 784
        elif args.dataset == '2dgaussian':
            dim_w = 2
        elif args.dataset == 'kdgaussian':
            dim_w = args.gaussian_dim

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        if args.search_burnin:
            temp_list = [0.1, 1, 10, 100, 1000]
            burn_in_list = [1, 10, 100, 200, 500, 1000]
            for temp in temp_list:
                acc_list = []
                for burn_in in burn_in_list:
                    for trial_id in range(100):
                        trial_list = []
                        args.temp = temp
                        args.burn_in = burn_in
                        accuracy_from_scratch, w_init = get_accuracy(None, dim_w, X_train, y_train, X_test, y_test, device, args)
                        trial_list.append(accuracy_from_scratch)
                        if trial_id == 9:
                            acc_list.append(np.mean(trial_list))
                plt.plot(burn_in_list, acc_list, label='temp: '+str(temp))
            plt.legend()
            plt.title(str(args.dataset)+'grid search burn in')
            plt.xlabel('burn in steps')
            plt.ylabel('accuracy')
            plt.savefig(str(args.dataset)+'_grid_burnin.jpg')
        
        if args.search_finetune:
            # train the weights (before removal)
            accuracy_from_scratch, w_init = get_accuracy(None, dim_w, X_train, y_train, X_test, y_test, device, args)
            finetune_list = [0, 1, 10, 50, 100]
            remove_num_list = [500, 1000, 2000, 3000, 5000]
            for num_remove in remove_num_list:
                acc_list = []
                args.num_removes = num_remove
                # remove data
                X_train_removed = X_train[:-args.num_removes,:]
                y_train_removed = y_train[:-args.num_removes]
                for finetune_step in finetune_list:
                    if finetune_step == 0:
                        acc_list.append(accuracy_from_scratch)
                    else:
                        args.finetune_step = finetune_step
                        for trial_id in range(30):
                            trial_list = []
                            accuracy_removed, w_removed = get_accuracy(w_init, dim_w, X_train_removed, y_train_removed, X_test, y_test, device, args)
                            trial_list.append(accuracy_removed)
                            if trial_id == 29:
                                acc_list.append(np.mean(trial_list))
                plt.plot(finetune_list, acc_list, label='num_removed: '+str(num_remove))
            plt.legend()
            plt.title(str(args.dataset)+' grid search finetune')
            plt.xlabel('finetune steps')
            plt.ylabel('accuracy')
            plt.savefig(str(args.dataset)+'_grid_finetune.jpg')

        else:
            # train the weights (before removal)
            accuracy_from_scratch, w_init = get_accuracy(None, dim_w, X_train, y_train, X_test, y_test, device, args)
            # remove data
            X_train_removed = X_train[:-args.num_removes,:]
            y_train_removed = y_train[:-args.num_removes]
            # fine-tune on new data
            accuracy_removed, w_removed = get_accuracy(w_init, dim_w, X_train_removed, y_train_removed, X_test, y_test, device, args)
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()