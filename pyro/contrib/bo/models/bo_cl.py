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
from pyro.contrib.bo.models.bo import BO
class BOCL(BO):
    """
    Bayesian Optimization with Constant Liar
    """

    def get_next(self, batch_size, opt_restarts, num_warmup=1000, num_iter=250, liar='mean'):
        if liar == 'max':    
            liar_constant = self.gp.y.max(0)
        elif liar == 'min':
            liar_constant = self.gp.y.min(0)
        else:
            liar_constant = self.gp.y.mean(0)

        X = torch.zeros([batch_size, self.num_dim])
        for i in range(batch_size):
            x = super().get_next(1, opt_restarts, num_warmup, num_iter)
            y = liar_constant
            self.gp.X = torch.cat((self.gp.X, x), 0)
            self.gp.y = torch.cat((self.gp.y, y), 0)
            self.fit_gp()
            X[i,:] = x
        return X
