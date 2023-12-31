import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

sns.set_style("whitegrid")

scratch_acc_list = np.load('./learn_scratch_acc.npy')
scratch_unlearn_list = np.load('./unlearn_scratch_acc.npy')
finetune_unlearn_list = np.load('./unlearn_finetune_acc.npy')
epsilon0_list = np.load('./epsilon0.npy')
sigma_list = [0.05, 0.1, 0.2, 0.5, 1]
fig, ax1 = plt.subplots()

ax1.plot(sigma_list, epsilon0_list, label = r'$\epsilon_0$', marker='^', linestyle='--')  
ax1.set_xlabel(r'$\sigma$', fontsize = 15)
ax1.set_ylabel(r'$\epsilon_0$', fontsize = 15)  
ax1.tick_params(axis='y')  
ax2 = ax1.twinx() 
ax2.plot(sigma_list, scratch_acc_list, label = 'learn D', marker = 'o')
ax2.plot(sigma_list, scratch_unlearn_list, label = 'learn D\'', marker = 'o')
ax2.plot(sigma_list, finetune_unlearn_list, label = 'unlearn D\'', marker = 'o')

ax2.set_ylabel('test accuracy', fontsize = 15)
ax2.tick_params(axis='y')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles = handles1 + handles2
labels = labels1 + labels2

ax1.legend(handles, labels)

plt.tight_layout()
plt.savefig('paint_utility_sigma.pdf')
plt.clf()