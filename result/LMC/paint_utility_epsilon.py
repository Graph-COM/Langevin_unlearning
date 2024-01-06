import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text


sns.set_style("whitegrid")

#dataset = 'MNIST'
dataset = 'CIFAR10'

K_dict = np.load('./'+str(dataset)+'/paint_utility_epsilon/K_list.npy', allow_pickle = True)
if dataset == 'MNIST':
    K_list = [17271, 2461, 1031, 421, 111]
elif dataset == 'CIFAR10':
    K_list = [20061, 2931, 1251, 521, 151]
epsilon_list = [0.1, 0.5, 1, 2, 5]


sns.lineplot(x = epsilon_list, y = K_list, marker='o', label = 'unlearn D\' from D')
texts = []
for i, (xi, yi) in enumerate(zip(epsilon_list, K_list)):
    unlearn_acc_list = np.load('./'+str(dataset)+'/paint_utility_epsilon/acc_finetune_epsilon'+str(xi)+'.npy', allow_pickle = True)
    texts.append(plt.text(xi, yi, f'accuracy={np.mean(unlearn_acc_list):.3f}', fontsize=10))
adjust_text(texts)
#plt.axhline(y=learn_acc, color='r', linestyle='--', label = 'learn D\' from scratch')
plt.legend(fontsize=15)
plt.xlabel(r'$\epsilon$', fontsize=15)
plt.ylabel('K', fontsize=15)

plt.tight_layout()
plt.savefig('./'+str(dataset)+'_paint_utiliy_epsilon.pdf')