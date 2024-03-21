# Langevin_unlearning

This is the official implementation of the paper Langevin Unlearning: A New Perspective of Noisy Gradient Descent for Machine Unlearning.

## Environment requirements

The code is runnable under the following enveironment:

````
matplotlib                      3.7.2
notebook                        7.0.7
numpy                           1.24.4
pandas                          2.0.3
scikit-learn                    1.3.0
scipy                           1.10.1
seaborn                         0.13.0
torch                           2.0.0+cu117
torchvision                     0.15.1+cu117
tqdm                            4.65.0
````

## To implement and re-produce the result in Figure 3.a, run

````
python baseline.py --lam 1e-6 --dataset [MNIST/CIFAR10]
````

## To implement and re-produce the result in Figure 3.b, run

````
python baseline.py --lam 1e-6 --dataset [MNIST/CIFAR10] --sequential2 1
````

## To implement and re-produce the result in Figure 3.c, run

````
python main_lmc.py --lam 1e-6 --sigma 0.03 --dataset [MNIST/CIFAR10] --paint_unlearning_sigma 1
````

## To implement and re-produce the result in Figure 4, run

````
python main_lmc.py --lam 1e-6 --sigma 0.03 --dataset [MNIST/CIFAR10] --paint_utility_epsilon 1
````

## To implement and reproduce the result in Figure 5 (Appendix), run

````
python main_lmc.py --lam 1e-6 --sigma 0.03 --dataset [MNIST/CIFAR10] --paint_utility_s 1
````

## To visualize the figures in the paper

please refer to ./result/LMC/Plotplace.ipynb

* Notes: *use --gpu to allocate to a GPU device*

  
