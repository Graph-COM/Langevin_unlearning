import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

sns.set_style("whitegrid")

K_list = np.load('./k_list.npy', allow_pickle = True)
num_remove_list = [1, 10, 50, 100, 500, 1000]
learn_scratch_list = np.load('./learn_scratch_acc.npy', allow_pickle = True)
unlearn_list = np.load('./unlearn_scratch_acc.npy', allow_pickle = True)
num_remove_list = [10, 50, 100, 500, 1000]
learn_acc = learn_scratch_list[0]
learn_scratch_list = learn_scratch_list[2:]
unlearn_list = unlearn_list[2:]
K_list = K_list[1:]
sns.lineplot(x = num_remove_list, y = learn_scratch_list, marker='o', label = 'learn D\' from scratch')
sns.lineplot(x = num_remove_list, y = unlearn_list, marker='o', label = 'learn D\' from D')
texts = []
for i, (xi, yi) in enumerate(zip(num_remove_list, unlearn_list)):
    texts.append(plt.text(xi, yi, f'K={K_list[i]}', fontsize=10))
adjust_text(texts)
#plt.axhline(y=learn_acc, color='r', linestyle='--', label = 'learn D from scratch')
plt.legend(fontsize=12)
plt.xlabel(r'number of removed data S', fontsize=15)
plt.ylabel('test accuracy', fontsize=15)

plt.tight_layout()
plt.savefig('./paint_utiliy_s.pdf')