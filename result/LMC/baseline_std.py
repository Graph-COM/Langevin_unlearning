import numpy as np

datas = ['MNIST', 'CIFAR10']

epsilon_list = [0.1, 0.5, 1, 2, 5]

for data in datas:
    print(data)
    for epsilon in epsilon_list:
        path = './'+str(data)+'/baseline/lmc_acc_unlearn_finetune'+str(epsilon)+'.npy'
        _data = np.load(path, allow_pickle = True)
        print(epsilon)
        print(np.mean(_data))
        print(np.std(_data))