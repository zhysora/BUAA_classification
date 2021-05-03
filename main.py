import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD, Adagrad, RMSprop, Adam
from torch.nn import CrossEntropyLoss as CELoss
import matplotlib.pyplot as plt

from dataset import get_dataset_info
from models import Classifier
from utils import parse_args, get_root_logger, evaluate


if __name__ == '__main__':
    if not os.path.exists('out'):
        os.mkdir('out')

    args = parse_args()
    logger = get_root_logger(args)
    logger.info('===> Current Config')
    for k, v in args._get_kwargs():
        logger.info(f'{k} = {v}')

    logger.info('===> Loading Dataset')
    # load dataset
    train_set, n_class, in_chans, img_size = get_dataset_info(args.dataset, False)
    test_set, _, _, _ = get_dataset_info(args.dataset, True)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    logger.info(f'size of training set: {train_set.__len__()}')
    logger.info(f'size of testing set: {test_set.__len__()}')

    logger.info('===> Building Model')
    model = Classifier(in_chans, n_class, img_size)

    if args.cuda:
        logger.info('===> Setting CUDA')
        model = model.cuda()

    logger.info('===> Setting Optimizer & Loss')
    optim_dict = dict(
        SGD=SGD(model.parameters(), args.lr, momentum=args.momentum),
        Adagrad=Adagrad(model.parameters(), args.lr),
        RMSprop=RMSprop(model.parameters(), args.lr, alpha=args.alpha),
        Adam=Adam(model.parameters(), args.lr, betas=(args.beta1, args.beta2)),
    )
    optim = optim_dict[args.optim]

    loss_func = CELoss()

    train_loss_epoch = []
    # train_m1_epoch = []
    # train_m2_epoch = []
    # train_m3_epoch = []
    test_m1_epoch = []
    test_m2_epoch = []
    test_m3_epoch = []
    m1, m2, m3 = 'acc', 'rec', 'f1'

    for epoch_no in range(args.epoch):
        logger.info(f'===> Training Epoch[{epoch_no+1}/{args.epoch}]')
        model.train()
        loss_arr = []
        for batch_no, input_batch in enumerate(train_loader):
            x, y = input_batch
            x = x.to(model.device())
            y = y.to(model.device())

            y_hat = model(x)
            loss = loss_func(y_hat, y.squeeze(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()
            if batch_no % 20 == 0:
                logger.info(f'Batch-{batch_no} loss:{loss.item()}')

            loss_arr.append(loss.item())

        train_loss_epoch.append(np.mean(loss_arr))

        model.eval()
        # logger.info(f'===> Eval on Train set at Epoch-{epoch_no+1}')
        # acc, rec, f1 = evaluate(model, train_loader)
        # train_m1_epoch.append(acc)
        # train_m2_epoch.append(rec)
        # train_m3_epoch.append(f1)
        # logger.info(f'acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}')
        logger.info(f'===> Eval on Test set at Epoch-{epoch_no+1}')
        acc, rec, f1 = evaluate(model, test_loader)
        test_m1_epoch.append(acc)
        test_m2_epoch.append(rec)
        test_m3_epoch.append(f1)
        logger.info(f'acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}')

    logger.info('===> Summary')
    logger.info(f'train_loss_epoch: {train_loss_epoch}')
    # logger.info(f'train_{m1}_epoch: {train_m1_epoch}')
    # logger.info(f'train_{m2}_epoch: {train_m2_epoch}')
    # logger.info(f'train_{m3}_epoch: {train_m3_epoch}')
    logger.info(f'test_{m1}_epoch: {test_m1_epoch}')
    logger.info(f'test_{m2}_epoch: {test_m2_epoch}')
    logger.info(f'test_{m3}_epoch: {test_m3_epoch}')

    logger.info('===> Final Report')
    logger.info(f'best test {m1}: {np.max(test_m1_epoch):.6f} at epoch {np.argmax(test_m1_epoch)+1}')
    logger.info(f'best test {m2}: {np.max(test_m2_epoch):.6f} at epoch {np.argmax(test_m2_epoch)+1}')
    logger.info(f'best test {m3}: {np.max(test_m3_epoch):.6f} at epoch {np.argmax(test_m3_epoch)+1}')

    # painting
    x_axis = [i + 1 for i in range(args.epoch)]

    plt.figure()
    plt.plot(x_axis, train_loss_epoch, marker='.', color='y', label='loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Train Loss Curve')
    plt.savefig(f'out/{args.name}/train_loss_curve.png')
    plt.show()
    plt.cla()

    # plt.figure()
    # plt.plot(x_axis, train_m1_epoch, marker='o', color='r', label=m1)
    # plt.plot(x_axis, train_m2_epoch, marker='s', color='b', label=m2)
    # plt.plot(x_axis, train_m3_epoch, marker='^', color='g', label=m3)
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('value')
    # plt.title('Train Metric Curve')
    # plt.savefig(f'out/{args.name}/train_metric_curve.png')
    # plt.show()
    # plt.cla()

    plt.figure()
    plt.plot(x_axis, test_m1_epoch, marker='o', color='r', label=m1)
    plt.plot(x_axis, test_m2_epoch, marker='s', color='b', label=m2)
    plt.plot(x_axis, test_m3_epoch, marker='^', color='g', label=m3)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Test Metric Curve')
    plt.savefig(f'out/{args.name}/test_metric_curve.png')
    plt.show()
    plt.cla()
