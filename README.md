# Langevin_unlearning

This is the official implementation of the **Neurips 2024 Spotlight** paper [Langevin Unlearning: A New Perspective of Noisy Gradient Descent for Machine Unlearning.](https://arxiv.org/abs/2401.10371)

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

* **Notes:**
We have already calculated the lowest $\sigma$ w.r.t. target $\epsilon= \[0.05, 0.1, 0.5, 1, 2, 5\], k =\[1,2,5\]$ for MNIST and CIFAR10 (see line 285-298 in baseline.py).
To calculate the lowest $\sigma$ w.r.t. an arbitrary $\epsilon, k$, one may run
````
python baseline.py --lam 1e-6 --dataset [MNIST/CIFAR10] --find_k 1
````
The command above could produce satisfying value of $\sigma$ when $k>1$, for $k=1$, one may further select more concise $\sigma$ value via further manual search via the try__() function (line499)

## To implement and re-produce the result in Figure 3.b, run

````
python baseline.py --lam 1e-6 --dataset [MNIST/CIFAR10] --sequential2 1
python baseline_multiclass.py --lam 1e-6 --dataset [CIFAR10_MULTI] --sequential2 1
````
Note that CIFAR10_MULTI refers to multiple class classification.





## To implement and re-produce the result in Figure 3.c, run

````
python main_lmc.py --lam 1e-6 --sigma 0.03 --dataset [MNIST/CIFAR10] --paint_unlearning_sigma 1
python main_lmc_multiclass.py --lam 1e-6 --sigma 0.03 --dataset CIFAR10_MULTI --paint_unlearning_sigma 1
````

## To implement and re-produce the result in Figure 4, run

````
python main_lmc.py --lam 1e-6 --sigma 0.03 --dataset [MNIST] --paint_utility_epsilon 1
````

## To implement and reproduce the result in Figure 5 (Appendix), run

````
python main_lmc.py --lam 1e-6 --sigma 0.03 --dataset [MNIST/CIFAR10] --paint_utility_s 1
python main_lmc_multiclass.py --lam 1e-6 --sigma 0.015 --dataset CIFAR10_MULTI --paint_utility_s 1
````

## To visualize the figures in the paper

please refer to ./result/LMC/Plotplace.ipynb

* Notes: *use --gpu to allocate to a GPU device*

## Citation 

If you find our work helpful, please cite us:
```
@article{chien2024langevin,
  title={Langevin Unlearning: A New Perspective of Noisy Gradient Descent for Machine Unlearning},
  author={Chien, Eli and Wang, Haoyu and Chen, Ziang and Li, Pan},
  journal={arXiv preprint arXiv:2401.10371},
  year={2024}
}
```


  
