import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text


sns.set_style("whitegrid")

K_dict = np.load('./K_list.npy', allow_pickle = True)
K_list = [9171, 1311, 551, 221, 61]
epsilon_list = [0.1, 0.5, 1, 2, 5]
unlearn_acc_list = np.load('./unlearn_acc_list.npy', allow_pickle = True)
learn_acc = unlearn_acc_list[0]
unlearn_acc_list = unlearn_acc_list[1:]

sns.lineplot(x = epsilon_list, y = K_list, marker='o', label = 'unlearn D\' from D')
texts = []
for i, (xi, yi) in enumerate(zip(epsilon_list, K_list)):
    texts.append(plt.text(xi, yi, f'accuracy={unlearn_acc_list[i]:.3f}', fontsize=10))
adjust_text(texts)
#plt.axhline(y=learn_acc, color='r', linestyle='--', label = 'learn D\' from scratch')
plt.legend(fontsize=15)
plt.xlabel(r'$\epsilon$', fontsize=15)
plt.ylabel('K', fontsize=15)

plt.tight_layout()
plt.savefig('./paint_utiliy_epsilon.pdf')