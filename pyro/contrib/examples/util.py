import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import torch
import numpy as np
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXAMPLES_DIR, 'data')
RESULTS_DIR = os.path.join(EXAMPLES_DIR, 'results')


def get_data_loader(dataset_name,
                    batch_size=1,
                    dataset_transforms=None,
                    is_training_set=True,
                    shuffle=True):
    if not dataset_transforms:
        dataset_transforms = []
    trans = transforms.Compose([transforms.ToTensor()] + dataset_transforms)
    dataset = getattr(datasets, dataset_name)
    return DataLoader(
        dataset(root=DATA_DIR,
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
