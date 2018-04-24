"""
An example to use Gaussian Process (GP) module to classify MNIST. Follow the idea
from reference [1], we will combine a convolutional neural network with a RBF kernel
to create a "deep" kernel. Then we train a SparseVariationalGP model using SVI. Note
that the model is trained end-to-end in mini-batch.

With default arguments (trained on CPU), the accuracy is 98.59%.

Reference:

[1] Stochastic Variational Deep Kernel Learning
    Andrew G. Wilson, Zhiting Hu, Ruslan R. Salakhutdinov, Eric P. Xing
"""

# Code adapted from https://github.com/pytorch/examples/tree/master/mnist
from __future__ import absolute_import, division, print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer
import pyro.optim as optim
from pyro.contrib.examples.util import get_data_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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


def train(args, train_loader, gpmodel, svi, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        gpmodel.set_data(data, target)
        loss = svi.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {:2d} [{:5d}/{} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test(args, test_loader, gpmodel):
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # get prediction of GP model on new data
        f_loc, f_var = gpmodel(data)
        # use its likelihood to give prediction class
        pred = gpmodel.likelihood(f_loc, f_var)
        # compare prediction and target to count accuaracy
        correct += pred.eq(target).long().cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(args):
    train_loader = get_data_loader(dataset_name='MNIST',
                                   data_dir=args.data_dir,
                                   batch_size=args.batch_size,
                                   dataset_transforms=[transforms.Normalize((0.1307,), (0.3081,))],
                                   is_training_set=True,
                                   shuffle=True)
    test_loader = get_data_loader(dataset_name='MNIST',
                                  data_dir=args.data_dir,
                                  batch_size=args.batch_size,
                                  dataset_transforms=[transforms.Normalize((0.1307,), (0.3081,))],
                                  is_training_set=False,
                                  shuffle=True)

    cnn = CNN().cuda() if args.cuda else CNN()

    # optimizer in SVI just works with params which are active inside its model/guide scope;
    # so we need this helper to mark cnn's parameters active for each `svi.step()` call.
    def cnn_fn(x):
        return pyro.module("CNN", cnn)(x)
    # Create deep kernel by warping RBF with CNN.
    # CNN will transform a high dimension image into a low dimension 2D tensors for RBF kernel.
    # This kernel accepts inputs are inputs of CNN and gives outputs are covariance matrix of RBF on
    # outputs of CNN.
    kernel = gp.kernels.RBF(input_dim=10, lengthscale=torch.ones(10)).warp(iwarping_fn=cnn_fn)

    # init inducing points (taken randomly from dataset)
    Xu = next(iter(train_loader))[0][:args.num_inducing]
    # use MultiClass likelihood for 10-class classification problem
    likelihood = gp.likelihoods.MultiClass(num_classes=10)
    # Because we use Categorical distribution in MultiClass likelihood, we need GP model returns a list
    # of probabilities of each class. Hence it is required to use latent_shape = 10.
    # Turns on "whiten" flag will help optimization for variational models.
    gpmodel = gp.models.VariationalSparseGP(X=Xu, y=None, kernel=kernel, Xu=Xu,
                                            likelihood=likelihood, latent_shape=torch.Size([10]),
                                            num_data=60000, whiten=True)
    if args.cuda:
        gpmodel.cuda()

    optimizer = optim.Adam({"lr": args.lr})

    svi = infer.SVI(gpmodel.model, gpmodel.guide, optimizer, infer.Trace_ELBO())

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, train_loader, gpmodel, svi, epoch)
        with torch.no_grad():
            test(args, test_loader, gpmodel)
        print("Amount of time spent for epoch {}: {}s\n".format(epoch, int(time.time() - start_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pyro GP MNIST Example')
    parser.add_argument('--data-dir', type=str, default='../data', metavar='PATH',
                        help='default directory to cache MNIST data')
    parser.add_argument('--num-inducing', type=int, default=70, metavar='N',
                        help='number of inducing input (default: 70)')
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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    pyro.set_rng_seed(args.seed)

    main(args)
