"""
An example to use Gaussian Process (GP) module to classify MNIST. Follow the idea
from reference [1], we will combine a convolutional neural network with a RBF kernel
to create a "deep" kernel. Then we train a SparseVariationalGP model using SVI. Note
that the model is trained end-to-end in mini-batch.

With default arguments, the accuracy is 98.46%.

Reference:

[1] Stochastic Variational Deep Kernel Learning
    Andrew G. Wilson, Zhiting Hu, Ruslan R. Salakhutdinov, Eric P. Xing
"""

# Code adapted from https://github.com/pytorch/examples/tree/master/mnist
from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer
import pyro.optim as optim
import pyro.poutine as poutine
from examples import util


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


def train(args, train_loader, gpmodel, svi, cnn, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        gpmodel.set_data(data, target)
        # mark params of cnn active for svi's optimizer
        pyro.get_param_store().mark_params_active(cnn.parameters())
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
        f_loc, f_var = gpmodel(data)
        pred = gpmodel.likelihood(f_loc, f_var)
        correct += pred.eq(target).long().cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(args):
    train_loader = util.get_data_loader(dataset_name='MNIST',
                                        batch_size=args.batch_size,
                                        dataset_transforms=[transforms.Normalize((0.1307,), (0.3081,))],
                                        is_training_set=True,
                                        shuffle=True)
    test_loader = util.get_data_loader(dataset_name='MNIST',
                                       batch_size=args.batch_size,
                                       dataset_transforms=[transforms.Normalize((0.1307,), (0.3081,))],
                                       is_training_set=False,
                                       shuffle=True)

    cnn = pyro.module("CNN", CNN().cuda() if args.cuda else CNN())
    # create deep kernel by warping RBF with cnn
    kernel = gp.kernels.RBF(input_dim=10, lengthscale=torch.ones(10)).warp(iwarping_fn=cnn)
    # init inducing points (taken randomly from dataset)
    Xu = next(iter(train_loader))[0][:args.num_inducing]
    likelihood = gp.likelihoods.MultiClass(num_classes=10)
    gpmodel = gp.models.SparseVariationalGP(X=Xu, y=None, kernel=kernel, Xu=Xu,
                                            likelihood=likelihood, latent_shape=torch.Size([10]),
                                            num_data=60000, whiten=True)
    if args.cuda:
        gpmodel.cuda()

    optimizer = optim.Adam({"lr": args.lr})

    # it is necessary to scale the loss by taking averaging over train data size
    svi = infer.SVI(poutine.scale(gpmodel.model, 1/60000), poutine.scale(gpmodel.guide, 1/60000),
                    optimizer, "ELBO")

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, gpmodel, svi, cnn, epoch)
        with torch.no_grad():
            test(args, test_loader, gpmodel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pyro GP MNIST Example')
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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    pyro.set_rng_seed(args.seed)

    main(args)
