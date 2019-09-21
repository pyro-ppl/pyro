import os
import sys

import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from pyro.distributions.torch_patch import patch_dependency


@patch_dependency('torchvision.datasets.MNIST', torchvision)
class _MNIST(getattr(MNIST, '_pyro_unpatched', MNIST)):
    urls = [
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/train-images-idx3-ubyte.gz",
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/train-labels-idx1-ubyte.gz",
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/t10k-images-idx3-ubyte.gz",
        "https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/t10k-labels-idx1-ubyte.gz",
    ]


def get_data_loader(dataset_name,
                    data_dir,
                    batch_size=1,
                    dataset_transforms=None,
                    is_training_set=True,
                    shuffle=True):
    if not dataset_transforms:
        dataset_transforms = []
    trans = transforms.Compose([transforms.ToTensor()] + dataset_transforms)
    dataset = getattr(datasets, dataset_name)
    print("downloading data")
    dset = dataset(root=data_dir,
                   train=is_training_set,
                   transform=trans,
                   download=True)
    print("download complete.")
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def print_and_log(logger, msg):
    # print and log a message (if a logger is present)
    print(msg)
    sys.stdout.flush()
    if logger is not None:
        logger.write("{}\n".format(msg))
        logger.flush()


def get_data_directory(filepath=None):
    if 'CI' in os.environ:
        return os.path.expanduser('~/.data')
    return os.path.abspath(os.path.join(os.path.dirname(filepath),
                                        '.data'))
