CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.001 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.01 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset MNIST --lr 0.1 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.001 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.01 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR10 --lr 0.1 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.001 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.01 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim SGD --momentum 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim SGD --momentum 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim SGD --momentum 0.8 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim SGD --momentum 0.8;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim SGD --momentum 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim SGD --momentum 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim Adagrad & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim Adagrad;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0.9 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0.9;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim RMSprop --alpha 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim RMSprop --alpha 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.999;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim Adam --beta1 0.9 --beta2 0.99 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim Adam --beta1 0.9 --beta2 0.99;
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 128 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 64 --optim Adam --beta1 0.5 --beta2 0.999 & \
CUDA_VISIBLE_DEVICES=1 python main.py --cuda --dataset CIFAR100 --lr 0.1 --batch_size 32 --optim Adam --beta1 0.5 --beta2 0.999;
