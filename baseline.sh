#nohup python -u baseline.py --lam 1e-6 --dataset MNIST --gpu 1 >./MNIST_baseline.log 2>&1 </dev/null &
#nohup python -u baseline.py --lam 1e-6 --dataset CIFAR10 --gpu 6 >./CIFAR10_baseline.log 2>&1 </dev/null &
#nohup python -u baseline.py --lam 1e-6 --dataset ADULT --gpu 2 >./ADULT_baseline.log 2>&1 </dev/null &
#nohup python -u baseline_multiclass.py --lam 1e-6 --dataset CIFAR10_MULTI --gpu 4 >./CIFAR10_MULTI_baseline.log 2>&1 </dev/null &

#nohup python -u baseline.py --lam 1e-6 --dataset MNIST --sequential2 1 --gpu 6 >./MNIST_sequential2.log 2>&1 </dev/null &
#nohup python -u baseline.py --lam 1e-6 --dataset CIFAR10 --sequential2 1 --gpu 7 >./CIFAR10_sequential2.log 2>&1 </dev/null &