from __future__ import print_function
from pyro.contrib.gp.models import GPRegression
import numpy as np
import random
import copy
import logging.config
import shutil
import os
import re
import yaml
from pyro.optim import Adam
import torch.optim
from pyro.infer import SVI
import torch
from scipy.optimize import fmin_l_bfgs_b
import logging as log

from sklearn.gaussian_process import GaussianProcessRegressor


class BO(GPRegression):
    """
    Bayesian Optimization module.

    :param torch.autograd.Variable X: A 1D or 2D tensor of inputs.
    :param torch.autograd.Variable y: A 1D tensor of output data for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor noise: An optional noise parameter.
    :param int batch_size: batch size for evaluating multiple point at once.
    """

    def __init__(self, X, y, kernel, noise=None):
        super().__init__(X, y, kernel, noise=noise)
        self.dim = self.X.shape[1]


    def fit_gp(self, num_steps=1000):
        """
        Fit the gaussian process to the data X,y with ELBO

        :param int num_steps: number of steps for ELBO to take.
        :return: losses of SVI at each step.
        :rtype: list(float)
        """
        
        assert self.X.shape[0] == self.y.shape[0], f'they have to be the same shape on axis 0, X : {self.X.shape}, y : {self.y.shape}'
        assert self.y.shape[1] == 1, 'Only 1D output is currently supported'
        self.X, self.y = self.X.cpu(), self.y.cpu()
        self._y_train_mean = self.y.mean(0) #normalize the input
        self.y -= self._y_train_mean
        optim = Adam({"lr": 0.001})
        svi = SVI(self.model, self.guide, optim, loss="ELBO")
        losses = []
        for _ in range(num_steps):
            losses.append(svi.step())
        self.y += self._y_train_mean
        return losses

    def predict(self, x, return_std=False):
        if return_std:
            mean, cov = self(x, full_cov=True, noiseless=True)
            std = cov.diag().sqrt()
            return mean+self._y_train_mean, std
        else:
            mean = self(x, noiseless=True)
            return mean

    def get_next(self, batch_size, opt_restarts, num_warmup=1000, num_iter=250):
        """
        Compute the next batch_size best data point X' to evaluate according to acqusition function
        :return: 
        :rtype: list(float)
        """

        x_tries = torch.rand([num_warmup, self.dim])
        ys = self.acquire(x_tries)
        _, y_max_idx = ys.max(0, keepdim=True)
        x_max = x_tries[y_max_idx, ...]

        X = []
        for i in range(batch_size):
            for j in range(opt_restarts):
                x = x_max.clone()
                x.requires_grad  = True
                optim = torch.optim.Adam([x])
                for k in range(num_iter):
                    def closure(): 
                        optim.zero_grad()
                        loss = self.acquire(x)
                        loss.backward()
                        return loss
                    optim.step(closure=closure)
                    X.append(x.clone())
        return X

    def acquire(self, x):
        return self.predict(x)