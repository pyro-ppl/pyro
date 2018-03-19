#!/usr/bin/env python

import numpy as np
import torch 
from torch.autograd import Variable
from torch import nn
from copy import deepcopy
import sys


class GaussianProcess(nn.Module):
    
    def __init__(self, gp_config, verbose=False):

        super(GaussianProcess, self).__init__()

        self.verbose = verbose
        self.config = gp_config
                

    def forward(self, X, y):
        
        if type(X) == np.ndarray:
            X = Variable(torch.FloatTensor(X.copy()))
        if type(y) == np.ndarray:
            y = Variable(torch.FloatTensor(y.copy()))
        
        self.X = X
        self.y = y
        self.N = y.size(0)

        self.mu = self.config["mean_func"]["mu"]
        hypers_ = self.config["mean_func"]["hypers"]

        y_diff = y - self.mu(X, hypers_)

        self.K = self.config["cov_func"]["K"]
        hypers_ = self.config["cov_func"]["hypers"]
        self.Knn = self.K(X, X, hypers_)

        self.K_noise = self.config["noise_func"]["K"]
        hypers_ = self.config["noise_func"]["hypers"]
        self.Knn += self.K_noise(X, hypers_)
        
        self.Lnn = torch.potrf(self.Knn, False)

        log_det = 2. * self.Lnn.diag().log().sum()
        self.alpha = torch.mv(self.Lnn.t().inverse(),
                              torch.mv(self.Lnn.inverse(), y_diff))
        log_quad_term = y_diff.dot(self.alpha)
        
        loss = 0.5 * (log_det + log_quad_term) # == nlml upto constants we don't care about

        return loss
    

    def print_hypers(self):

        print_str = []
        for hn, hh in self.get_all_hypers().items():
            if "log_" in hn:
                print_str.append("%s: %s" % (hn.split("log_")[-1],
                                             " ".join(["%.5f" % np.exp(hh_i)
                                                       for hh_i in hh.data.numpy()])))
            else:
                print_str.append("%s: %s" % (hn, " ".join(["%.5f" % hh_i
                                                           for hh_i in hh.data.numpy()])))
        print_str = ", ".join(print_str)

        print print_str
        # for external use
        return print_str
            
            
    def get_all_hypers(self):
        
        # return dict of all GP hypers
        # shallow copy?
        all_hypers = dict([(k, v) for (k, v) in self.config["mean_func"]["hypers"].items()])

        if self.config["cov_func"]["K"].__name__ == "covSum":
            for summand_cov_func in self.config["cov_func"]["hypers"]:
                all_hypers.update(summand_cov_func["hypers"])
        else:
            all_hypers.update(self.config["cov_func"]["hypers"])

        all_hypers.update(self.config["noise_func"]["hypers"])

        return all_hypers
    
        
    def fit(self, X, y, method="adam", lr=1e-1, epochs=100, print_every=10):
        
        # get all trainable hypers
        hypers_ = self.get_all_hypers()
        hypers_ = [hh for hh in hypers_.values() if hh.requires_grad]
        # TODO figure out why self.parameters() is empty

        opt_dict = {"lbfgs": torch.optim.LBFGS(hypers_, lr=lr),
                    "adam": torch.optim.Adam(hypers_,
                                          lr=lr, betas=(0.5, 0.9))} 
        self.optimizer = opt_dict[method]
        self.losses = np.zeros(epochs)
        converged = False

        for i in range(epochs):

            if method == "lbfgs":

                loss = self.forward(X, y)
                self.losses[i] = loss.data.numpy()[0]
                if self.verbose and i % print_every == 0:
                    print 'loss @ %i: %.5f\r' % (i, self.losses[i]),
                    sys.stdout.flush()
                    #self.print_hypers()

                def closure():
                    self.optimizer.zero_grad()
                    loss = self.forward(X, y)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)

            elif method == "adam":
                self.optimizer.zero_grad()
                loss = self.forward(X, y)
                self.losses[i] = loss.data.numpy()[0]
                if self.verbose and i % print_every == 0:
                    print 'loss @ %i: %.5f\r' % (i, self.losses[i]),
                    sys.stdout.flush()
                loss.backward()
                self.optimizer.step()

            else:
                raise RuntimeError("Invalid optimization method %s" % method)

            if i > 1:
                # TODO should probably look at fluctuation in params
                error_fluct = np.mean(np.abs(np.diff(self.losses[max(0, i-10):i+1]) / self.losses[max(0, i-10):i]))
                if error_fluct < 1e-2:
                    print "Convergence attained after %i epochs" % i
                    converged = True
                    break
        if not converged:
            print "Warning: did not converge after %i epochs!" % epochs
                    

    def predict(self, Xt, predict_targets=False):
        
        if type(Xt) == np.ndarray:
            Xt = Variable(torch.FloatTensor(Xt.copy()))

        hypers_ = self.config["cov_func"]["hypers"]
        Kmn = self.K(Xt, self.X, hypers_)
        Kmm = self.K(Xt, Xt, hypers_)

        mu = torch.mv(Kmn, self.alpha)

        hypers_ = self.config["mean_func"]["hypers"]
        mu += self.mu(Xt, hypers_)

        v = torch.mm(self.Lnn.inverse(), Kmn.t()) # NxM
        sigma = Kmm - torch.mm(v.t(), v) # MxM

        # by default predict latent function
        if predict_targets:
            noise_hypers = self.config["noise_func"]["hypers"]
            Kstar_noise = self.K_noise(Xt, noise_hypers)
            sigma += Kstar_noise

        pred_dict = {"mean_func": self.mu(Xt, hypers_).data.numpy(),
                     "pred_mean": mu.data.numpy(),
                     "pred_var": sigma.data.numpy()}
        
        return pred_dict


if __name__ == "__main__":

    ### TEST
    from sklearn.datasets import make_regression
    from gp_priors import muConstant, covARD, covNoise
    from gp_priors import build_meanfunc, build_covfunc
    
    # 1. LOAD DATASET
    np.random.seed(1)

    X, y = make_regression(n_samples=100, n_features=1)

    y /= y.std()

    X = ((X - X.mean(axis=0)) / X.std(axis=0))

    N, D = X.shape

    # 2. LOAD MODEL
    gp_config = {"mean_func": build_meanfunc(muConstant, optimize_hypers=False, constant_mean=0.0),
                 "cov_func": build_covfunc(covARD, 
                                           optimize_hypers = True, 
                                           log_signal_var = 0.0,
                                           log_lengthscales = np.zeros(D)),
                 "noise_func": build_covfunc(covNoise, optimize_hypers=True, log_noise=-3.0)}

    gp = GaussianProcess(gp_config, verbose=True)
    #gp.fit(X, y, method="adam", lr=0.05, epochs=100, print_every=1)
    gp.fit(X, y, method="lbfgs", lr=0.01, epochs=100, print_every=1)
    print "\nFinal hypers"
    gp.print_hypers()

    pred_dict = gp.predict(X)

    print np.mean(y)
    print np.mean(np.abs(pred_dict['pred_mean'] - y))

