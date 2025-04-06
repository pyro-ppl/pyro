# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms


class MNIST(datasets.MNIST):
    mirrors = ["https://github.com/pyro-ppl/datasets/blob/master/mnist/"]

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            errors = []
            for mirror in self.mirrors:
                url = f"{mirror}{filename}?raw=true"
                try:
                    datasets.utils.download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except datasets.URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.mirrors, errors):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)


def get_data_loader(
    dataset_name,
    data_dir,
    batch_size=1,
    dataset_transforms=None,
    is_training_set=True,
    shuffle=True,
):
    if not dataset_transforms:
        dataset_transforms = []
    trans = transforms.Compose([transforms.ToTensor()] + dataset_transforms)
    if dataset_name == "MNIST":
        dataset = MNIST
    else:
        dataset = getattr(datasets, dataset_name)
    print("downloading data")
    dset = dataset(root=data_dir, train=is_training_set, transform=trans, download=True)
    print("download complete.")
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle)


def print_and_log(logger, msg):
    # print and log a message (if a logger is present)
    print(msg)
    sys.stdout.flush()
    if logger is not None:
        logger.write("{}\n".format(msg))
        logger.flush()


def get_data_directory(filepath=None):
    if "CI" in os.environ:
        return os.path.expanduser("~/.data")
    return os.path.abspath(os.path.join(os.path.dirname(filepath), ".data"))


def _mkdir_p(dirname):
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except FileExistsError:
            pass
