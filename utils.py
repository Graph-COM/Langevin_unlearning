import math
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import nltk
import pandas
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import torch

from datasets import load_dataset
from torchvision import datasets, transforms, models
from transformers import AutoTokenizer, AutoModel


def transform_array(arr):
    n = len(arr)
    
    result = torch.full((n, 10), 0)

    for i, num in enumerate(arr):
        result[int(i), int(num.item())] = 1

    return result

# constructs one-hot representations of labels
def onehot(y):
    y_onehot = -torch.ones(y.size(0), y.max() + 1).float()
    y_onehot.scatter_(1, y.long().unsqueeze(1), 1)
    return y_onehot

#cifar10_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])

cifar10_transform = transforms.Compose([
    transforms.Resize(224),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(ResNetFeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        return x

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
        elif args.dataset == 'CIFAR10':
            if not os.path.exists('./data/CIFAR10/train.pt'):
                trainset = datasets.CIFAR10(args.data_dir, train=True, download = True, transform=cifar10_transform)
                testset = datasets.CIFAR10(args.data_dir, train=False, download = True, transform=cifar10_transform)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
                testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
                pretrained_resnet = models.resnet18(pretrained=True)
                feature_extractor = ResNetFeatureExtractor(pretrained_resnet)
                feature_extractor.eval()
                train_feature = []
                test_feature = []
                with torch.no_grad():
                    for inputs, labels in tqdm(trainloader):
                        output = feature_extractor(inputs)
                        train_feature.append([output.reshape(-1), labels])
                    for inputs, labels in tqdm(testloader):
                        output = feature_extractor(inputs)
                        test_feature.append([output.reshape(-1), labels])
                torch.save(train_feature, './data/CIFAR10/train.pt')
                torch.save(test_feature, './data/CIFAR10/test.pt')
                trainset = train_feature
                testset = test_feature
            else:
                trainset = torch.load('./data/CIFAR10/train.pt')
                testset = torch.load('./data/CIFAR10/test.pt')
            X_train = torch.zeros(len(trainset), 512)
            y_train = torch.zeros(len(trainset))
            X_test = torch.zeros(len(testset),512)
            y_test = torch.zeros(len(testset))
            for i in range(len(trainset)):
                x, y = trainset[i]
                X_train[i] = x.view(512)
                y_train[i] = y.item()
            for i in range(len(testset)):
                x, y = testset[i]
                X_test[i] = x.view(512)
                y_test[i] = y.item()
            X_train_3 = X_train[torch.where(y_train.eq(3))]
            y_train_3 = y_train[torch.where(y_train.eq(3))]
            X_train_8 = X_train[torch.where(y_train.eq(8))]
            y_train_8 = y_train[torch.where(y_train.eq(8))]
            X_train = torch.cat((X_train_3, X_train_8), 0)
            y_train = torch.cat((y_train_3, y_train_8), 0).eq(3).float()
            test_indices = (y_test.eq(3) + y_test.eq(8)).gt(0)
            X_test = X_test[test_indices]
            y_test = y_test[test_indices].eq(3).float()
        elif args.dataset == 'CIFAR10_MULTI':
            if not os.path.exists('./data/CIFAR10/train.pt'):
                trainset = datasets.CIFAR10(args.data_dir, train=True, download = True, transform=cifar10_transform)
                testset = datasets.CIFAR10(args.data_dir, train=False, download = True, transform=cifar10_transform)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
                testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
                pretrained_resnet = models.resnet18(pretrained=True)
                feature_extractor = ResNetFeatureExtractor(pretrained_resnet)
                feature_extractor.eval()
                train_feature = []
                test_feature = []
                with torch.no_grad():
                    for inputs, labels in tqdm(trainloader):
                        output = feature_extractor(inputs)
                        train_feature.append([output.reshape(-1), labels])
                    for inputs, labels in tqdm(testloader):
                        output = feature_extractor(inputs)
                        test_feature.append([output.reshape(-1), labels])
                torch.save(train_feature, './data/CIFAR10/train.pt')
                torch.save(test_feature, './data/CIFAR10/test.pt')
                trainset = train_feature
                testset = test_feature
            else:
                trainset = torch.load('./data/CIFAR10/train.pt')
                testset = torch.load('./data/CIFAR10/test.pt')
            X_train = torch.zeros(len(trainset), 512)
            y_train = torch.zeros(len(trainset))
            X_test = torch.zeros(len(testset),512)
            y_test = torch.zeros(len(testset))
            for i in range(len(trainset)):
                x, y = trainset[i]
                X_train[i] = x.view(512)
                y_train[i] = y.item()
            for i in range(len(testset)):
                x, y = testset[i]
                X_test[i] = x.view(512)
                y_test[i] = y.item()
            y_train = transform_array(y_train).float()
            y_test = transform_array(y_test).float()
        elif args.dataset == 'ADULT':
            train_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            column_names = [
                "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"
            ]
            numeric_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
            #categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
            categorical_features = ["sex"]
            # Load the training data
            data = pandas.read_csv(train_data_url, names=column_names, sep=',\s', na_values="?", engine='python')
            # Load the testing data
            data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
            X = data.drop('income', axis=1)
            y = data['income']
            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])
            #preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])
            preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
            X_preprocessed = preprocessor.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.1, random_state = 123)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_train = torch.tensor(y_train.values, dtype=torch.float32)
            y_test = torch.tensor(y_test.values, dtype=torch.float32)
            #import pdb; pdb.set_trace()
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
        y_train_onehot = y_train
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


def create_nested_folder(path):
    os.makedirs(path, exist_ok=True)


    
    
