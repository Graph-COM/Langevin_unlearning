# The script below are for LMC painting

# paint utility - s figure
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --paint_utility_s 1 --gpu 5 >./MNIST_LMC_paint_utility_s.log 2>&1 </dev/null &

# paint utility - epsilon figure
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --paint_utility_epsilon 1 --gpu 6 >./MNIST_LMC_paint_utility_epsilon.log 2>&1 </dev/null &

# paint utility - sigma figure
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --paint_utility_sigma 1 --gpu 6 >./MNIST_LMC_paint_utility_sigma.log 2>&1 </dev/null &

# paint unlearning utility - sigma figure
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --paint_unlearning_sigma 1 --gpu 6 >./MNIST_LMC_paint_unlearning_sigma_2.log 2>&1 </dev/null &

# calculate unlearning step between our bound and the baseline bound
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --compare_k 1 --gpu 2 >./MNIST_LMC_compare_k.log 2>&1 </dev/null &

# find the best batch size b per gradient for sgd
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --find_best_batch 1 --gpu 6 >./MNIST_LMC_find_best_batch.log 2>&1 </dev/null &

