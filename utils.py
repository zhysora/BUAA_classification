import os
import argparse
import logging
import torch
from torch.nn.functional import softmax
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    # dataset config
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10', 'CIFAR100'], default='MNIST')
    parser.add_argument('--batch_size', type=int, default=32)
    # optim config
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optim', choices=['SGD', 'Adagrad', 'RMSprop', 'Adam'], default='SGD',
                        help='type of optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='hyper-param for SGD')
    parser.add_argument('--alpha', type=float, default=.99, help='hyper-param for RMSprop')
    parser.add_argument('--beta1', type=float, default=.9, help='hyper-param for Adam')
    parser.add_argument('--beta2', type=float, default=.999, help='hyper-param for Adam')

    parser.add_argument('--name', default='demo', help='output dir')

    args = parser.parse_args()
    args.name = args.dataset
    args.name += f'_b{args.batch_size}'
    args.name += f'_lr{args.lr}'
    args.name += f'_{args.optim}'
    if args.optim in ['SGD']:
        args.name += f'_hp{args.momentum}'
    if args.optim in ['RMSprop']:
        args.name += f'_hp{args.alpha}'
    if args.optim in ['Adam']:
        args.name += f'_hp({args.beta1},{args.beta2})'

    return args


def get_root_logger(args):
    if not os.path.exists(f'out/{args.name}'):
        os.mkdir(f'out/{args.name}')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'out/{args.name}/log.txt', mode='w')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def evaluate(model, loader):
    def pred(x):
        _, id = torch.max(softmax(x, dim=-1), dim=1)
        return id.cpu().numpy()

    true_ys = np.array([], dtype=np.int)
    pred_ys = np.array([], dtype=np.int)
    for x, y in loader:
        x = x.to(model.device())
        y = y.to(model.device())

        y_hat = model(x)
        true_ys = np.concatenate((true_ys, y.squeeze(-1).cpu().numpy()))
        pred_ys = np.concatenate((pred_ys, pred(y_hat)))

    return precision_score(true_ys, pred_ys, average='macro'), \
           recall_score(true_ys, pred_ys, average='macro'), \
           f1_score(true_ys, pred_ys, average='macro')
