# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, functional


class CVAEMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.original = MNIST(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.original)

    def __getitem__(self, item):
        image, digit = self.original[item]
        sample = {'original': image, 'digit': digit}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        sample['original'] = functional.to_tensor(sample['original'])
        sample['digit'] = torch.as_tensor(np.asarray(sample['digit']),
                                          dtype=torch.int64)
        return sample


class MaskImages:
    def __init__(self, num_quadrant_inputs, mask_with=-1):
        if num_quadrant_inputs <= 0 or num_quadrant_inputs >= 4:
            raise ValueError('Number of quadrants as inputs must be 1, 2 or 3')
        self.num = num_quadrant_inputs
        self.mask_with = mask_with

    def __call__(self, sample):
        tensor = sample['original'].squeeze()
        out = tensor.detach().clone()
        h, w = tensor.shape

        out[h // 2:, :w // 2] = self.mask_with
        if self.num == 2:
            out[:, :w // 2] = self.mask_with
        if self.num == 3:
            out[:h // 2, :] = self.mask_with

        inp = tensor.clone()
        inp[out != -1] = self.mask_with

        sample['input'] = inp
        sample['output'] = out
        return sample


def get_data(num_quadrant_inputs, batch_size):
    transforms = Compose([
        ToTensor(),
        MaskImages(num_quadrant_inputs=num_quadrant_inputs)
    ])
    datasets, dataloaders, dataset_sizes = {}, {}, {}
    for mode in ['train', 'val']:
        datasets[mode] = CVAEMNIST(
            '../data',
            download=True,
            transform=transforms,
            train=mode == 'train'
        )
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=batch_size,
            shuffle=mode == 'train',
            num_workers=0
        )
        dataset_sizes[mode] = len(datasets[mode])

    return datasets, dataloaders, dataset_sizes
