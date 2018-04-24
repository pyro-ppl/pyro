import sys

import numpy as np
import torch
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


def set_seed(seed, use_cuda):
    """
    setting the seed for controlling randomness in this example
    :param seed: seed value (int)
    :param use_cuda: set the random seed for torch.cuda or not
    :return: None
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
