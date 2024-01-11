#nohup python -u baseline.py --lam 1e-6 --dataset MNIST --gpu 0 >./MNIST_baseline.log 2>&1 </dev/null &
#nohup python -u baseline.py --lam 1e-6 --dataset CIFAR10 --gpu 7 >./CIFAR10_baseline.log 2>&1 </dev/null &

#nohup python -u baseline.py --lam 1e-6 --dataset MNIST --sequential 1 --gpu 7 >./MNIST_sequential.log 2>&1 </dev/null &
nohup python -u baseline.py --lam 1e-6 --dataset CIFAR10 --sequential 1 --gpu 7 >./CIFAR10_sequential.log 2>&1 </dev/null &