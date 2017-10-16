"""
This script generates a dataset similar to the Multi-MNIST dataset
described in [1].

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

import os
import numpy as np
import torch
import torchvision.datasets as dset
from scipy.misc import imresize


def sample_one(canvas_size, mnist):
    i = np.random.randint(mnist['digits'].shape[0])
    digit = mnist['digits'][i]
    label = mnist['labels'][i]
    scale = 0.1 * np.random.randn() + 1.3
    resized = imresize(digit, 1. / scale)
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


def load_mnist():
    loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root='./data',
            train=True,
            download=True))
    return {
        'digits': loader.dataset.train_data.numpy(),
        'labels': loader.dataset.train_labels
    }


# Generate the training set and dump it to disk. (Note, this will
# always generate the same data, else error out.)
def main():
    outfile = './data/multi_mnist_train_uint8.npz'
    if os.path.exists(outfile):
        print('Output file "{}" already exists. Quiting...'.format(outfile))
        return
    np.random.seed(681307)
    mnist = load_mnist()
    x, y = mk_dataset(60000, mnist, 2, 50)
    assert x.sum() == 884438093, 'Did not generate expected data.'
    with open(outfile, 'wb') as f:
        np.savez_compressed(f, x=x, y=y)


if __name__ == "__main__":
    main()
