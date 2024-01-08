import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text


sns.set_style("whitegrid")

dataset = 'MNIST'
#dataset = 'CIFAR10'

num_remove_list = [10, 300, 1000]
epsilon_list = [0.1, 0.5, 1, 2, 5]
K_dict = np.load('./'+str(dataset)+'/paint_utility_epsilon/K_list.npy', allow_pickle = True)

for remove_idx, num_remove in enumerate(num_remove_list):
    
    K_list = []
    for epsilon in epsilon_list:
        K_list.append(K_dict[num_remove][epsilon])

    sns.lineplot(x = epsilon_list, y = K_list, marker='o', label = 'unlearn D\' from D, S='+str(num_remove))
    texts = []
    for i, (xi, yi) in enumerate(zip(epsilon_list, K_list)):
        unlearn_acc_list = np.load('./'+str(dataset)+'/paint_utility_epsilon/'+str(num_remove)+'/acc_finetune_epsilon'+str(xi)+'.npy', allow_pickle = True)
        texts.append(plt.text(xi, yi, f'accuracy={np.mean(unlearn_acc_list):.3f}', fontsize=10))
    adjust_text(texts)
    #plt.axhline(y=learn_acc, color='r', linestyle='--', label = 'learn D\' from scratch')
plt.legend(fontsize=15)
plt.xlabel(r'$\epsilon$', fontsize=15)
plt.ylabel('K', fontsize=15)

plt.tight_layout()
plt.savefig('./'+str(dataset)+'_paint_utiliy_epsilon.pdf')