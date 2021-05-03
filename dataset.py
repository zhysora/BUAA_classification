from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from os import mkdir
from os.path import exists
from torch.utils.data import Dataset


def get_dataset_info(name='MNIST', test=False):
    r"""
        MNIST: train 60000, test 10000, shape (1, 28, 28), labels: 0-9
        CIFAR10: train 50000, test 10000, shape (3, 32, 32), labels: 0-9
        CIFAR100: train 50000, test 10000, shape (3, 32, 32), labels: 0-99

    Args:
        name (str): 数据集名称, Choice in ['MNIST', 'CIFAR10', 'CIFAR100']
        test (bool): 是否为测试集, Default: False
    Returns:
        (Dataset, int, int, int): 数据集, 类别数, channel, img_size
    """
    pre_tf = Compose([ToTensor(), Normalize([0.5], [0.5])])
    assert name in ['MNIST', 'CIFAR10', 'CIFAR100'], 'Unknown Dataset !!'
    if not exists('data'):
        mkdir('data')
    if name == 'MNIST':
        return MNIST('data', train=not test, transform=pre_tf, download=True), 10, 1, 28
    if name == 'CIFAR10':
        return CIFAR10('data', train=not test, transform=pre_tf, download=True), 10, 3, 32
    if name == 'CIFAR100':
        return CIFAR100('data', train=not test, transform=pre_tf, download=True), 100, 3, 32


if __name__ == '__main__':
    dataset, _, _, _ = get_dataset_info('CIFAR100', True)
    max_label = 0
    for i in range(dataset.__len__()):
        _, label = dataset.__getitem__(i)
        max_label = max(label, max_label)
    print(max_label)
