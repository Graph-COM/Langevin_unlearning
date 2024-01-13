#nohup python -u baseline.py --lam 1e-6 --dataset MNIST --gpu 1 >./MNIST_baseline.log 2>&1 </dev/null &
#nohup python -u baseline.py --lam 1e-6 --dataset CIFAR10 --gpu 6 >./CIFAR10_baseline.log 2>&1 </dev/null &

#nohup python -u baseline.py --lam 1e-6 --dataset MNIST --sequential 1 --gpu 1 >./MNIST_sequential.log 2>&1 </dev/null &
#nohup python -u baseline.py --lam 1e-6 --dataset CIFAR10 --sequential 1 --gpu 6 >./CIFAR10_sequential.log 2>&1 </dev/null &

nohup python -u baseline.py --lam 1e-6 --dataset MNIST --sequential2 1 --gpu 6 >./MNIST_sequential2.log 2>&1 </dev/null &
nohup python -u baseline.py --lam 1e-6 --dataset CIFAR10 --sequential2 1 --gpu 1 >./CIFAR10_sequential2.log 2>&1 </dev/null &