if __name__ == '__main__':
    with open('work.sh', 'w') as file:
        device = 1
        datasets = ['MNIST', 'CIFAR10', 'CIFAR100']
        lrs = [.001, .01, .1]
        batch_sizes = [128, 64, 32]
        optims = ['SGD', 'Adagrad', 'RMSprop', 'Adam']
        momentums = [0, .8, .9]
        alphas = [0, .9, .99]
        betas = [(.9, .999), (.9, .99), (.5, .999)]

        cmds = []

        for dataset in datasets:
            for lr in lrs:
                for optim in optims:
                    if optim == 'SGD':
                        for momentum in momentums:
                            for batch_size in batch_sizes:
                                cmds.append(f'CUDA_VISIBLE_DEVICES={device} python main.py --cuda '
                                            f'--dataset {dataset} --lr {lr} --batch_size {batch_size} --optim {optim} '
                                            f'--momentum {momentum}')
                    elif optim == 'Adagrad':
                        for batch_size in batch_sizes:
                            cmds.append(f'CUDA_VISIBLE_DEVICES={device} python main.py --cuda '
                                        f'--dataset {dataset} --lr {lr} --batch_size {batch_size} --optim {optim}')
                    elif optim == 'RMSprop':
                        for alpha in alphas:
                            for batch_size in batch_sizes:
                                cmds.append(f'CUDA_VISIBLE_DEVICES={device} python main.py --cuda '
                                            f'--dataset {dataset} --lr {lr} --batch_size {batch_size} --optim {optim} '
                                            f'--alpha {alpha}')
                    elif optim == 'Adam':
                        for beta in betas:
                            for batch_size in batch_sizes:
                                cmds.append(f'CUDA_VISIBLE_DEVICES={device} python main.py --cuda '
                                            f'--dataset {dataset} --lr {lr} --batch_size {batch_size} --optim {optim} '
                                            f'--beta1 {beta[0]} --beta2 {beta[1]}')

        for i, cmd in enumerate(cmds):
            file.write(f'{cmd}')
            if (i + 1) % 3 == 0:
                file.write(';\n')
            else:
                file.write(' & \\\n')
