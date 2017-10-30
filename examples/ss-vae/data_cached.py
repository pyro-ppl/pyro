import torch
from torchvision.datasets import MNIST
import numpy as np
from functools import reduce
from torch.utils.data import DataLoader


# transformations for MNIST data
def fn_x_mnist(x, use_cuda):
    # normalize pixel values of the image to be in [0,1] instead of [0,255]
    xp = x * (1. / 255)

    # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
    xp_1d_size = reduce(lambda a, b: a * b, xp.size()[1:])
    xp = xp.view(-1, xp_1d_size)

    # send the data to GPU(s)
    if use_cuda:
        xp = xp.cuda()

    return xp


def fn_y_mnist(y, use_cuda):
    yp = torch.zeros(y.size(0), 10)

    # send the data to GPU(s)
    if use_cuda:
        yp = yp.cuda()
        y = y.cuda()

    # transform the label y (integer between 0 and 9) to a one-hot
    yp = yp.scatter_(1, y.view(-1, 1), 1.0)
    return yp


def split_sup_unsup(X, y, sup_perc):
    """
        helper function for splitting the data into supervised and un-supervised part
    :param X: images
    :param y: labels (digits)
    :param sup_perc: what percentage of data is supervised
    :return: splits of data by sup_perc percentage
    """
    # number of examples
    n = X.size()[0]

    # number of supervised examples
    sup_n = int(n * sup_perc / 100.0)

    return X[0:sup_n], y[0:sup_n], X[sup_n:n], y[sup_n:n],


class MNISTCached(MNIST):
    """
        a wrapper around MNIST to load and cache the transformed data
        once at the beginning of the inference
    """
    def __init__(self, train="sup", sup_perc=5.0, use_cuda=True, *args, **kwargs):
        super(MNISTCached, self).__init__(train=train in ["sup", "unsup"], *args, **kwargs)

        # transformations on MNIST data (normalization and one-hot conversion for labels)
        def transform(x):
            return fn_x_mnist(x, use_cuda)

        def target_transform(y):
            return fn_y_mnist(y, use_cuda)

        assert train in ["sup", "unsup", "test"], "invalid train/test option values"

        if train in ["sup", "unsup"]:

            # transform the training data if transformations are provided
            if transform is not None:
                self.train_data = (transform(self.train_data.float()))
            if target_transform is not None:
                self.train_labels = (target_transform(self.train_labels))

            train_data_sup, train_labels_sup, train_data_unsup, train_labels_unsup = \
                split_sup_unsup(self.train_data, self.train_labels, sup_perc)
            if train == "sup":
                self.train_data, self.train_labels = train_data_sup, train_labels_sup
            else:
                self.train_data = train_data_unsup

                # making sure that the unsupervised labels are not available to inference
                self.train_labels = (torch.Tensor(train_labels_unsup.shape[0]).view(-1, 1)) * np.nan

        else:
            # transform the testing data if transformations are provided
            if transform is not None:
                self.test_data = (transform(self.test_data.float()))
            if target_transform is not None:
                self.test_labels = (target_transform(self.test_labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index or slice object

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target


def setup_data_loaders(dataset, use_cuda, batch_size, sup_perc, root='./data', download=True, **kwargs):
    """
        helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param dataset: the data to use
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param sup_perc: percentage of supervised data
    :param root: where on the filesystem should the dataset be
    :param download: download the dataset (if it doesn't exist already)
    :param kwargs: other params for the pytorch data loader
    :return: three data loaders: (supervised data for training, un-supervised data for training,
                                  supervised data for testing)
    """
    # instantiate the dataset as training/testing sets
    train_set_sup = dataset(root=root, train="sup", download=download,
                            sup_perc=sup_perc, use_cuda=use_cuda)

    train_set_unsup = dataset(root=root, train="unsup", download=download,
                              sup_perc=sup_perc, use_cuda=use_cuda)

    test_set = dataset(root=root, train="test", sup_perc=sup_perc,
                       use_cuda=use_cuda)

    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 0, 'pin_memory': False}

    # setup the data loaders
    train_loader_sup = DataLoader(train_set_sup, batch_size=batch_size, shuffle=True, **kwargs)
    train_loader_unsup = DataLoader(train_set_unsup, batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader_sup, train_loader_unsup, test_loader