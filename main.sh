# The script below are for LMC painting

# paint utility - s figure
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset MNIST --paint_utility_s 1 --gpu 1 >./MNIST_LMC_paint_utility_s.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset CIFAR10 --paint_utility_s 1 --gpu 6 >./CIFAR10_LMC_paint_utility_s.log 2>&1 </dev/null &

# paint utility - epsilon figure
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset MNIST --paint_utility_epsilon 1 --gpu 1 >./MNIST_LMC_paint_utility_epsilon.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03  --dataset CIFAR10 --paint_utility_epsilon 1 --gpu 6 >./CIFAR10_LMC_paint_utility_epsilon.log 2>&1 </dev/null &

# paint unlearning utility - sigma figure
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset MNIST --paint_unlearning_sigma 1 --gpu 6 >./MNIST_LMC_paint_unlearning_sigma.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset CIFAR10 --paint_unlearning_sigma 1 --gpu 1 >./CIFAR10_LMC_paint_unlearning_sigma.log 2>&1 </dev/null &

