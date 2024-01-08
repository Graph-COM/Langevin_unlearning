import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

sns.set_style("whitegrid")

dataset = 'MNIST'
# dataset = 'CIFAR10'


sigma_list = [0.05, 0.1, 0.2, 0.5, 1]
epsilon0_list = np.load('./'+str(dataset)+'/paint_unlearning_sigma/epsilon0.npy')

D_acc = []
D_std = []
Dnew_acc = []
Dnew_std = []
Dnew_ft_acc = []
Dnew_ft_std = []



for sigma in sigma_list:
    scratch_acc_list = np.load('./'+str(dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_scratch_D.npy')
    scratch_unlearn_list = np.load('./'+str(dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_scratch_Dnew.npy')
    finetune_unlearn_list = np.load('./'+str(dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_finetune.npy')
    D_acc.append(np.mean(scratch_acc_list))
    D_std.append(np.std(scratch_acc_list))
    Dnew_acc.append(np.mean(scratch_unlearn_list))
    Dnew_std.append(np.std(scratch_unlearn_list))
    Dnew_ft_acc.append(np.mean(finetune_unlearn_list))
    Dnew_ft_std.append(np.std(finetune_unlearn_list))

D_acc = np.array(D_acc)
D_std = np.array(D_std)
Dnew_acc = np.array(Dnew_acc)
Dnew_std = np.array(Dnew_std)
Dnew_ft_acc = np.array(Dnew_ft_acc)
Dnew_ft_std = np.array(Dnew_ft_std)

fig, ax1 = plt.subplots()

ax1.plot(sigma_list, epsilon0_list, label = r'$\epsilon_0$', marker='^', linestyle='--')  
ax1.set_xlabel(r'$\sigma$', fontsize = 15)
ax1.set_ylabel(r'$\epsilon_0$', fontsize = 15)  
ax1.tick_params(axis='y')  
ax2 = ax1.twinx() 
ax2.plot(sigma_list, D_acc, label = 'learn D', marker = 'o')
ax2.plot(sigma_list, Dnew_acc, label = 'learn D\'', marker = 'o')
ax2.plot(sigma_list, Dnew_ft_acc, label = 'unlearn D\'', marker = 'o')

#ax2.fill_between(sigma_list, D_acc - D_std, D_acc + D_std, alpha=0.2)
#ax2.fill_between(sigma_list, Dnew_acc - Dnew_std, Dnew_acc + Dnew_std, alpha = 0.2)
#ax2.fill_between(sigma_list, Dnew_ft_acc - Dnew_ft_std, Dnew_ft_acc + Dnew_ft_std, alpha = 0.2)

ax2.set_ylabel('test accuracy', fontsize = 15)
ax2.tick_params(axis='y')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles = handles1 + handles2
labels = labels1 + labels2

ax1.legend(handles, labels)

plt.tight_layout()
plt.savefig('./'+str(dataset)+'_paint_utility_sigma.pdf')
plt.clf()