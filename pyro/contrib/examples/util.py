from __future__ import absolute_import, division, print_function

import os
import sys

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms


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
    return DataLoader(
        dataset(root=data_dir,
                train=is_training_set,
                transform=trans,
                download=True),
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
