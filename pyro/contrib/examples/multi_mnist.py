# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This script generates a dataset similar to the Multi-MNIST dataset
described in [1].

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

import os

import numpy as np
from PIL import Image

from pyro.contrib.examples.util import get_data_loader


def imresize(arr, size):
    return np.array(Image.fromarray(arr).resize(size))


def sample_one(canvas_size, mnist):
    i = np.random.randint(mnist['digits'].shape[0])
    digit = mnist['digits'][i]
    label = mnist['labels'][i].item()
    scale = 0.1 * np.random.randn() + 1.3
    new_size = tuple(int(s / scale) for s in digit.shape)
    resized = imresize(digit, new_size)
    w = resized.shape[0]
    assert w == resized.shape[1]
    padding = canvas_size - w
    pad_l = np.random.randint(0, padding)
    pad_r = np.random.randint(0, padding)
    pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
    positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    return positioned, label


def sample_multi(num_digits, canvas_size, mnist):
    canvas = np.zeros((canvas_size, canvas_size))
    labels = []
    for _ in range(num_digits):
        positioned_digit, label = sample_one(canvas_size, mnist)
        canvas += positioned_digit
        labels.append(label)
    # Crude check for overlapping digits.
    if np.max(canvas) > 255:
        return sample_multi(num_digits, canvas_size, mnist)
    else:
        return canvas, labels


def mk_dataset(n, mnist, max_digits, canvas_size):
    x = []
    y = []
    for _ in range(n):
        num_digits = np.random.randint(max_digits + 1)
        canvas, labels = sample_multi(num_digits, canvas_size, mnist)
        x.append(canvas)
        y.append(labels)
    return np.array(x, dtype=np.uint8), y


def load_mnist(root_path):
    loader = get_data_loader('MNIST', root_path)
    return {
        'digits': loader.dataset.data.cpu().numpy(),
        'labels': loader.dataset.targets
    }


def load(root_path):
    file_path = os.path.join(root_path, 'multi_mnist_uint8.npz')
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
        return data['x'], data['y']
    else:
        # Set RNG to known state.
        rng_state = np.random.get_state()
        np.random.seed(681307)
        mnist = load_mnist(root_path)
        print('Generating multi-MNIST dataset...')
        x, y = mk_dataset(60000, mnist, 2, 50)
        # Revert RNG state.
        np.random.set_state(rng_state)
        # Crude checksum.
        # assert x.sum() == 883114919, 'Did not generate the expected data.'
        with open(file_path, 'wb') as f:
            np.savez_compressed(f, x=x, y=y)
        print('Done!')
        return x, y
