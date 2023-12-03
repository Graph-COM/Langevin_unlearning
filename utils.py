import math
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torchvision import datasets, transforms



# constructs one-hot representations of labels
def onehot(y):
    y_onehot = -torch.ones(y.size(0), y.max() + 1).float()
    y_onehot.scatter_(1, y.long().unsqueeze(1), 1)
    return y_onehot


def load_features(args):
    ckpt_file = '%s/%s_%s_extracted.pth' % (args.data_dir, args.extractor, args.dataset)
    if os.path.exists(ckpt_file):
        checkpoint = torch.load(ckpt_file)
        X_train = checkpoint['X_train'].cpu()
        y_train = checkpoint['y_train'].cpu()
        X_test = checkpoint['X_test'].cpu()
        y_test = checkpoint['y_test'].cpu()
    else:
        print('Extracted features not found, loading raw features.')
        if args.dataset == 'MNIST':
            trainset = datasets.MNIST(args.data_dir, train=True, download = True, transform=transforms.ToTensor())
            testset = datasets.MNIST(args.data_dir, train=False, download = True, transform=transforms.ToTensor())
            X_train = torch.zeros(len(trainset), 784)
            y_train = torch.zeros(len(trainset))
            X_test = torch.zeros(len(testset), 784)
            y_test = torch.zeros(len(testset))
            for i in range(len(trainset)):
                x, y = trainset[i]
                X_train[i] = x.view(784) - 0.5
                y_train[i] = y
            for i in range(len(testset)):
                x, y = testset[i]
                X_test[i] = x.view(784) - 0.5
                y_test[i] = y
            # load classes 3 and 8
            X_train_3 = X_train[torch.where(y_train.eq(3))]
            y_train_3 = y_train[torch.where(y_train.eq(3))]
            X_train_8 = X_train[torch.where(y_train.eq(8))]
            y_train_8 = y_train[torch.where(y_train.eq(8))]
            X_train = torch.cat((X_train_3, X_train_8), 0)
            y_train = torch.cat((y_train_3, y_train_8), 0).eq(3).float()
            #train_indices = (y_train.eq(3) + y_train.eq(8)).gt(0)
            test_indices = (y_test.eq(3) + y_test.eq(8)).gt(0)
            #X_train = X_train[train_indices]
            #y_train = y_train[train_indices].eq(3).float()
            X_test = X_test[test_indices]
            y_test = y_test[test_indices].eq(3).float()
        else:
            print("Error: Unknown dataset %s. Aborting." % args.dataset) 
            sys.exit(1)

    # L2 normalize features
    X_train /= X_train.norm(2, 1).unsqueeze(1)
    X_test /= X_test.norm(2, 1).unsqueeze(1)
    # convert labels to +/-1 or one-hot vectors
    if args.train_mode == 'binary':
        y_train_onehot = y_train
        y_train = (2 * y_train - 1)
    else:
        y_train_onehot = onehot(y_train)
    if len(y_train_onehot.size()) == 1:
        y_train_onehot = y_train_onehot.unsqueeze(1)
        
    return X_train, X_test, y_train, y_train_onehot, y_test


# generate binary classification of gaussian distribution
def generate_gaussian(dim, num, mean_1, mean_2, std):
    num_per_class = int(num / 2)
    std = std.expand(num_per_class, -1).float()
    mean_1 = mean_1.expand(num_per_class, -1).float()
    mean_2 = mean_2.expand(num_per_class, -1).float()
    sample1 = torch.normal(mean_1, std)
    sample2 = torch.normal(mean_2, std)
    y_1 = torch.ones(num_per_class)
    y_2 = torch.ones(num_per_class) - 2
    samples = torch.cat((sample1, sample2), 0)
    labels = torch.cat((y_1, y_2), 0)

    #shuffle_idx = list(range(num))
    #random.shuffle(shuffle_idx)
    #X = samples[shuffle_idx]
    #y = labels[shuffle_idx]
    return samples, labels

#plot 2D gaussian logistic weight picture
def plot_2dgaussian(logistic_density, X_train, y_train, args, title):
    r = np.linspace(-10, 10, 100)
    x, y = np.meshgrid(r, r)
    z = np.vstack([x.flatten(), y.flatten()]).T

    q0 = []
    for i in tqdm(range(z.shape[0])):
        this_q = logistic_density(torch.tensor(z[i]).float(), X_train.cpu(), y_train.cpu(), args.lam, args.temp)
        q0.append(this_q.item())
    q0 = torch.tensor(q0)
    plt.pcolormesh(x, y, q0.reshape(x.shape),
                cmap='viridis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title(title)
    plt.savefig(title + '.jpg')
    plt.clf()

def plot_w_2dgaussian(w_init_list, title):
    w_init_list = np.array(w_init_list)
    plt.hist2d(w_init_list[:,0], w_init_list[:,1], cmap='viridis', range = [[-10, 10],[-10, 10]],  rasterized=False, bins=200, density=True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title(title)
    plt.savefig(title+'.jpg')
    plt.clf()


    
    
