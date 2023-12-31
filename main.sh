# search burn in
#nohup python -u main.py --step_size 0.1 --search_burnin 1 --lam 1e-6 --dataset 2dgaussian >./search_burnin_2dgaussian.log 2>&1 </dev/null &
#nohup python -u main.py --search_burnin 1 --lam 1e-6 --dataset MNIST --gpu 5 >./search_burnin_MNIST.log 2>&1 </dev/null &

# search burn in on new data
#nohup python -u main.py --step_size 0.1 --search_burnin_newdata 1 --lam 1e-6 --dataset 2dgaussian --temp 5 >./search_burnin_newdata_2dgaussian.log 2>&1 </dev/null &
#nohup python -u main.py --search_burnin_newdata 1 --lam 1e-6 --dataset MNIST --sigma 0.1 --gpu 7 >./search_burnin_newdata_MNIST.log 2>&1 </dev/null &

# search fine-tune
#nohup python -u main.py --step_size 0.1 --search_finetune 1 --lam 1e-6 --dataset 2dgaussian --burn_in 500 --temp 5 >./search_finetune_2dgaussian.log 2>&1 </dev/null &
#nohup python -u main.py --step_size 0.1 --search_finetune 1 --lam 1e-7 --dataset MNIST --burn_in 5000 --sigma 0.1 >./search_finetune_MNIST.log 2>&1 </dev/null &

# after searching, run the code to get results
#nohup python -u main.py --step_size 0.1 --lam 1e-6 --dataset 2dgaussian --burn_in 500 --temp 5 --finetune_step 200 --num-removes 3000 --gpu 7 >./2dgaussian.log 2>&1 </dev/null &
#nohup python -u main.py --step_size 0.1 --lam 1e-8 --dataset MNIST --burn_in 3000 --temp 100 --finetune_step 200 --num-removes 4000 >./MNIST.log 2>&1 </dev/null &

# paint utility - s figure
#nohup python -u main.py --lam 1e-6 --dataset MNIST --paint_utility_s 1 --gpu 5 >./MNIST_paint_utility_s.log 2>&1 </dev/null &

# paint utility - epsilon figure
#nohup python -u main.py --lam 1e-6 --dataset MNIST --paint_utility_epsilon 1 --gpu 6 >./MNIST_paint_utility_epsilon.log 2>&1 </dev/null &

# paint utility - sigma figure
#nohup python -u main.py --lam 1e-6 --dataset MNIST --paint_utility_sigma 1 --gpu 6 >./MNIST_paint_utility_sigma.log 2>&1 </dev/null &

# paint unlearning utility - sigma figure
#nohup python -u main.py --lam 1e-6 --dataset MNIST --paint_unlearning_sigma 1 --gpu 6 >./MNIST_paint_unlearning_sigma_2.log 2>&1 </dev/null &

# calculate unlearning step between our bound and the baseline bound
#nohup python -u main.py --lam 1e-6 --dataset MNIST --compare_k 1 --gpu 2 >./MNIST_compare_k.log 2>&1 </dev/null &

# find the best batch size b per gradient for sgd
nohup python -u main.py --lam 1e-6 --dataset MNIST --find_best_batch 1 --gpu 6 >./MNIST_find_best_batch.log 2>&1 </dev/null &
