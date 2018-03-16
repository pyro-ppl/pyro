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
from pyro.contrib.gp.models.gpr import GPRegression

class BO:
    def __init__(self, gp, acq_func):
        self.gp = gp
        self.acq_func = acq_func
        self.num_dim = self.gp.X.shape[1]

    def get_next(self, batch_size, opt_restarts, num_warmup=1000, num_iter=250):
        """
        Compute the next batch_size best data point to evaluate according to acqusition function, 
        by random sampling to warmup and ADAM optimization.


        :param int batch_size:  description
        :param int opt_restarts:  description
        
        num_warmup {int} -- [description] (default: {1000})
        num_iter {int} -- [description] (default: {250})
        
        [type] -- [description]
        """
        
        X = torch.zeros([batch_size, self.num_dim])
        for i in range(batch_size):
            self.acq_func.update()
            x_min = 0
            y_min = 100000
            for j in range(opt_restarts):
                x = torch.randn([1, self.num_dim])
                x.requires_grad  = True
                optim = torch.optim.Adam([x])
                for k in range(num_iter):
                    def closure(): 
                        optim.zero_grad()
                        loss = self.acq_func(x)
                        loss.backward()
                        return loss
                    optim.step(closure=closure)
                with torch.no_grad():
                    y = self.acq_func(x)                    
                if y_min is None or y_min > y:
                    y_min = y
                    x_min = x.clone()
            X[i,:] = x[0,:]
        return X

    def fit_gp(self, normalize_y = True, num_steps=1000): 
        """ 
        Fit the gaussian process to the data X,y with ELBO 
 
        :param int num_steps: number of steps for ELBO to take. 
        :return: losses of SVI at each step. 
        :rtype: list(float) 
        """ 
         
        assert self.gp.X.shape[0] == self.gp.y.shape[0], f'they have to be the same shape on axis 0, X : {self.gp.X.shape}, y : {self.gp.y.shape}' 
        assert self.gp.y.shape[1] == 1, 'Only 1D output is currently supported' 
        if normalize_y:
            self._y_train_mean = self.gp.y.mean(0) #normalize the input 
            self.gp.y -= self._y_train_mean 
        optim = Adam({"lr": 0.01}) 
        svi = SVI(self.model, self.guide, optim, loss="ELBO") 
        losses = [] 
        for _ in range(num_steps): 
            losses.append(svi.step()) 
        self.y += self._y_train_mean 
        return losses 
 
    def predict(self, x, return_std=False): 
        if return_std: 
            mean, cov = self.gp(x, full_cov=True, noiseless=True) 
            std = cov.diag().sqrt() 
            return mean+self._y_train_mean, std 
        else: 
            mean = self.gp(x, noiseless=True) 
            return mean 