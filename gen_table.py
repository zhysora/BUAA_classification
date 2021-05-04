from subprocess import getoutput

if __name__ == '__main__':
    with open('table.csv', 'w') as f:
        dataset = 'MNIST'   # ['MNIST', 'CIFAR10', 'CIFAR100']
        lrs = [1e-4, 1e-3, 1e-2]
        batch_sizes = [32, 64, 128]
        optims = ['SGD', 'Adagrad', 'RMSprop', 'Adam']
        momentums = [0, .9]
        alphas = [.99]
        betas = [(.9, .999)]

        cmds = []

        for lr in lrs:
            for batch_size in batch_sizes:
                for optim in optims:
                    if optim == 'SGD':
                        for momentum in momentums:
                            outputs = getoutput(f'grep -r "best test" '
                                                f'out/{dataset}*b{batch_size}*lr{lr}*{optim}*hp{momentum}*').split('\n')

                            if len(outputs) != 3:
                                print('bad logs!!!')

                            m1 = outputs[0].split()[10]
                            m2 = outputs[1].split()[10]
                            m3 = outputs[2].split()[10]
                            epoch = outputs[2].split()[13]
                            f.write(f'{dataset}, {batch_size}, {lr}, {optim}, {momentum}, , , , '
                                    f'{m1}, {m2}, {m3}, {epoch}\n')
                    elif optim == 'Adagrad':
                        outputs = getoutput(f'grep -r "best test" '
                                            f'out/{dataset}*b{batch_size}*lr{lr}*{optim}*').split('\n')

                        if len(outputs) != 3:
                            print('bad logs!!!')

                        m1 = outputs[0].split()[10]
                        m2 = outputs[1].split()[10]
                        m3 = outputs[2].split()[10]
                        epoch = outputs[2].split()[13]
                        f.write(f'{dataset}, {batch_size}, {lr}, {optim}, , , , , '
                                f'{m1}, {m2}, {m3}, {epoch}\n')
                    elif optim == 'RMSprop':
                        for alpha in alphas:
                            outputs = getoutput(f'grep -r "best test" '
                                                f'out/{dataset}*b{batch_size}*lr{lr}*{optim}*hp{alpha}*')
                            outputs = outputs.split('\n')
                            if len(outputs) != 3:
                                print('bad logs!!!')
                            m1 = outputs[0].split()[10]
                            m2 = outputs[1].split()[10]
                            m3 = outputs[2].split()[10]
                            epoch = outputs[2].split()[13]
                            f.write(f'{dataset}, {batch_size}, {lr}, {optim}, , {alpha}, , , '
                                    f'{m1}, {m2}, {m3}, {epoch}\n')
                    elif optim == 'Adam':
                        for beta in betas:
                            outputs = getoutput(f'grep -r "best test" '
                                                f'out/{dataset}*b{batch_size}*lr{lr}*{optim}*hp{beta}*')
                            outputs = outputs.split('\n')
                            if len(outputs) != 3:
                                print('bad logs!!!')
                            m1 = outputs[0].split()[10]
                            m2 = outputs[1].split()[10]
                            m3 = outputs[2].split()[10]
                            epoch = outputs[2].split()[13]
                            f.write(f'{dataset}, {batch_size}, {lr}, {optim}, , , {beta[0]}, {beta[1]}, '
                                    f'{m1}, {m2}, {m3}, {epoch}\n')

