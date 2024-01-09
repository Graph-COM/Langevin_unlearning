import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

sns.set_style("whitegrid")

#dataset = 'MNIST'
dataset = 'CIFAR10'


num_remove_list = [10, 50, 100, 500, 1000]
learn_scratch_list = np.load('./'+str(dataset)+'/paint_utility_s/acc_scratch_D.npy', allow_pickle = True)

epsilon_list = [0.5, 1, 2]
num_remove_list = [10, 50, 100, 500, 1000]

colors = sns.color_palette()

for epsilon_idx, epsilon in enumerate(epsilon_list):
    K_list = np.load('./'+str(dataset)+'/paint_utility_s/'+str(epsilon)+'/K_list.npy', allow_pickle = True)
    unlearn_scratch_acc = []
    unlearn_scratch_std = []
    unlearn_finetune_acc = []
    unlearn_finetune_std = []
    for num_remove in num_remove_list:
        unlearn_scratch_list = np.load('./'+str(dataset)+'/paint_utility_s/'+str(epsilon)+'/acc_scratch_Dnew_remove'+str(num_remove)+'.npy', allow_pickle = True)
        acc = np.mean(unlearn_scratch_list)
        acc_std = np.std(unlearn_scratch_list)
        unlearn_scratch_acc.append(acc)
        unlearn_scratch_std.append(acc_std)
        unlearn_finetune_list = np.load('./'+str(dataset)+'/paint_utility_s/'+str(epsilon)+'/acc_finetune_remove'+str(num_remove)+'.npy', allow_pickle = True)
        acc = np.mean(unlearn_finetune_list)
        acc_std = np.std(unlearn_finetune_list)
        unlearn_finetune_acc.append(acc)
        unlearn_finetune_std.append(acc_std)
        
    unlearn_scratch_acc = np.array(unlearn_scratch_acc)
    unlearn_scratch_std = np.array(unlearn_scratch_std)
    unlearn_finetune_acc = np.array(unlearn_finetune_acc)
    unlearn_finetune_std = np.array(unlearn_finetune_std)

    sns.lineplot(x = num_remove_list, y = unlearn_scratch_acc, color = colors[epsilon_idx%len(colors)], marker='o', label = r"D' from scratch, $\epsilon$="+str(epsilon))
    sns.lineplot(x = num_remove_list, y = unlearn_finetune_acc, color = colors[epsilon_idx%len(colors)], marker='o', label = r"D' from D, $\epsilon$="+str(epsilon), linestyle = '--')
    #plt.fill_between(num_remove_list, unlearn_scratch_acc - unlearn_scratch_std, unlearn_scratch_acc + unlearn_scratch_std, alpha=0.2)
    #plt.fill_between(num_remove_list, unlearn_finetune_acc - unlearn_finetune_std, unlearn_finetune_acc + unlearn_finetune_std, alpha = 0.2)
    texts = []
    for i, (xi, yi) in enumerate(zip(num_remove_list, unlearn_finetune_acc)):
        texts.append(plt.text(xi, yi, f'K={K_list[i]}', fontsize=8))
    adjust_text(texts)
    #plt.axhline(y=learn_acc, color='r', linestyle='--', label = 'learn D from scratch')

plt.legend(fontsize=8)
plt.xlabel(r'number of removed data S', fontsize=15)
plt.ylabel('test accuracy', fontsize=15)

plt.tight_layout()
plt.savefig('./'+str(dataset)+'_paint_utiliy_s.pdf')