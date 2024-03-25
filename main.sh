# The script below are for LMC painting

# paint utility - s figure
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset MNIST --paint_utility_s 1 --gpu 1 >./MNIST_LMC_paint_utility_s.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset CIFAR10 --paint_utility_s 1 --gpu 6 >./CIFAR10_LMC_paint_utility_s.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset ADULT --paint_utility_s 1 --gpu 1 >./ADULT_LMC_paint_utility_s.log 2>&1 </dev/null &
#nohup python -u main_lmc_multiclass.py --lam 1e-6 --sigma 0.015 --dataset CIFAR10_MULTI --paint_utility_s 1 --gpu 2 >./CIFAR10_MULTI_LMC_paint_utility_s.log 2>&1 </dev/null &

# paint utility - epsilon figure
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset MNIST --paint_utility_epsilon 1 --gpu 1 >./MNIST_LMC_paint_utility_epsilon.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03  --dataset CIFAR10 --paint_utility_epsilon 1 --gpu 6 >./CIFAR10_LMC_paint_utility_epsilon.log 2>&1 </dev/null &

# paint unlearning utility - sigma figure
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset MNIST --paint_unlearning_sigma 1 --gpu 6 >./MNIST_LMC_paint_unlearning_sigma.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset CIFAR10 --paint_unlearning_sigma 1 --gpu 1 >./CIFAR10_LMC_paint_unlearning_sigma.log 2>&1 </dev/null &

# get the utility of noiseless retrain
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset MNIST --retrain_noiseless 1 --gpu 2 >./MNIST_LMC_retrain_noiseless.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset CIFAR10 --retrain_noiseless 1 --gpu 0 >./CIFAR10_LMC_retrain_noiseless.log 2>&1 </dev/null &

#nohup python -u main_lmc_old.py --lam 1e-6 --dataset ADULT --search_burnin 1 --gpu 1 >./ADULT_LMC_search_burnin_lam1e6.log 2>&1 </dev/null &


#nohup python -u main_lmc_multiclass.py --lam 1e-6 --dataset CIFAR10_MULTI --search_burnin 1 --gpu 1 >./CIFAR10_MULTI_search_burnin_lam1e6.log 2>&1 </dev/null &