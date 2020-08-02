# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
An example to use Pyro Gaussian Process module to classify MNIST and binary MNIST.

Follow the idea from reference [1], we will combine a convolutional neural network
(CNN) with a RBF kernel to create a "deep" kernel:

    >>> deep_kernel = gp.kernels.Warping(rbf, iwarping_fn=cnn)

SparseVariationalGP model allows us train the data in mini-batch (time complexity
scales linearly to the number of data points).

Note that the implementation here is different from [1]. There the authors
use CNN as a feature extraction layer, then add a Gaussian Process layer on the
top of CNN. Hence, their inducing points lie in the space of extracted features.
Here we join CNN module and RBF kernel together to make it a deep kernel.
Hence, our inducing points lie in the space of original images.

After 16 epochs with default hyperparameters, the accuaracy of 10-class MNIST
is 98.45% and the accuaracy of binary MNIST is 99.41%.

Reference:

[1] Stochastic Variational Deep Kernel Learning
    Andrew G. Wilson, Zhiting Hu, Ruslan R. Salakhutdinov, Eric P. Xing
"""

# Code adapted from https://github.com/pytorch/examples/tree/master/mnist
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer
from pyro.contrib.examples.util import get_data_loader, get_data_directory


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(args, train_loader, gpmodule, optimizer, loss_fn, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.binary:
            target = (target % 2).float()  # convert numbers 0->9 to 0 or 1

        gpmodule.set_data(data, target)
        optimizer.zero_grad()
        loss = loss_fn(gpmodule.model, gpmodule.guide)
        loss.backward()
        optimizer.step()
        batch_idx = batch_idx + 1
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {:2d} [{:5d}/{} ({:2.0f}%)]\tLoss: {:.6f}"
                  .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss))


def test(args, test_loader, gpmodule):
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.binary:
            target = (target % 2).float()  # convert numbers 0->9 to 0 or 1

        # get prediction of GP model on new data
        f_loc, f_var = gpmodule(data)
        # use its likelihood to give prediction class
        pred = gpmodule.likelihood(f_loc, f_var)
        # compare prediction and target to count accuracy
        correct += pred.eq(target).long().cpu().sum().item()

    print("\nTest set: Accuracy: {}/{} ({:.2f}%)\n"
          .format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main(args):
    data_dir = args.data_dir if args.data_dir is not None else get_data_directory(__file__)
    train_loader = get_data_loader(dataset_name='MNIST',
                                   data_dir=data_dir,
                                   batch_size=args.batch_size,
                                   dataset_transforms=[transforms.Normalize((0.1307,), (0.3081,))],
                                   is_training_set=True,
                                   shuffle=True)
    test_loader = get_data_loader(dataset_name='MNIST',
                                  data_dir=data_dir,
                                  batch_size=args.test_batch_size,
                                  dataset_transforms=[transforms.Normalize((0.1307,), (0.3081,))],
                                  is_training_set=False,
                                  shuffle=False)
    if args.cuda:
        train_loader.num_workers = 1
        test_loader.num_workers = 1

    cnn = CNN()

    # Create deep kernel by warping RBF with CNN.
    # CNN will transform a high dimension image into a low dimension 2D tensors for RBF kernel.
    # This kernel accepts inputs are inputs of CNN and gives outputs are covariance matrix of RBF
    # on outputs of CNN.
    rbf = gp.kernels.RBF(input_dim=10, lengthscale=torch.ones(10))
    deep_kernel = gp.kernels.Warping(rbf, iwarping_fn=cnn)

    # init inducing points (taken randomly from dataset)
    batches = []
    for i, (data, _) in enumerate(train_loader):
        batches.append(data)
        if i >= ((args.num_inducing - 1) // args.batch_size):
            break
    Xu = torch.cat(batches)[:args.num_inducing].clone()

    if args.binary:
        likelihood = gp.likelihoods.Binary()
        latent_shape = torch.Size([])
    else:
        # use MultiClass likelihood for 10-class classification problem
        likelihood = gp.likelihoods.MultiClass(num_classes=10)
        # Because we use Categorical distribution in MultiClass likelihood, we need GP model
        # returns a list of probabilities of each class. Hence it is required to use
        # latent_shape = 10.
        latent_shape = torch.Size([10])

    # Turns on "whiten" flag will help optimization for variational models.
    gpmodule = gp.models.VariationalSparseGP(X=Xu, y=None, kernel=deep_kernel, Xu=Xu,
                                             likelihood=likelihood, latent_shape=latent_shape,
                                             num_data=60000, whiten=True, jitter=2e-6)
    if args.cuda:
        gpmodule.cuda()

    optimizer = torch.optim.Adam(gpmodule.parameters(), lr=args.lr)

    elbo = infer.JitTraceMeanField_ELBO() if args.jit else infer.TraceMeanField_ELBO()
    loss_fn = elbo.differentiable_loss

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, train_loader, gpmodule, optimizer, loss_fn, epoch)
        with torch.no_grad():
            test(args, test_loader, gpmodule)
        print("Amount of time spent for epoch {}: {}s\n"
              .format(epoch, int(time.time() - start_time)))


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description='Pyro GP MNIST Example')
    parser.add_argument('--data-dir', type=str, default=None, metavar='PATH',
                        help='default directory to cache MNIST data')
    parser.add_argument('--num-inducing', type=int, default=70, metavar='N',
                        help='number of inducing input (default: 70)')
    parser.add_argument('--binary', action='store_true', default=False,
                        help='do binary classification')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='enables PyTorch jit')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    pyro.set_rng_seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True

    main(args)
