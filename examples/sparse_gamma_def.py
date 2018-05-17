from __future__ import absolute_import, division, print_function
from collections import defaultdict
import cloudpickle

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer.svi import SVI
from pyro.util import ng_ones, ng_zeros
from pyro.distributions.testing.fakes import NonreparameterizedGamma as NonRepGamma
from pyro.distributions.torch.gamma import Gamma as RepGamma
from pyro.distributions.testing.rejection_gamma import ShapeAugmentedGamma
import pyro.poutine as poutine
import time
import sys


data = Variable(torch.DoubleTensor(np.loadtxt('faces_training.csv',delimiter=',')))
print("Number of datapoints in training dataset: %d" % data.size(0))

class SparseGammaDEF(object):
    def __init__(self, gamma_dist=RepGamma, transformation="softplus", boost=1, seed=0):
        pyro.util.set_rng_seed(seed)
        self.top_width = 100
        self.mid_width = 40
        self.bottom_width = 15
        self.image_size = 64 * 64
        self.alpha_z = Variable(torch.Tensor([0.1]).type_as(data))
        self.alpha_w = Variable(torch.Tensor([0.1]).type_as(data))
        self.beta_w = Variable(torch.Tensor([0.3]).type_as(data))
        #self.softplus = nn.Softplus()
        if transformation=="softplus":
            self.forward = lambda x: torch.log(torch.exp(x)+1.0)
            self.backward = lambda x: torch.log(torch.exp(x)-1.0)
            self.alpha_init = 0.5   # => 0.97
            self.mean_init = 0.0    # =>  0.7
            self.sigma = 0.1
        else:
            self.forward = lambda x: torch.exp(x)
            self.backward = lambda x: torch.log(x)
            self.alpha_init = -0.03
            self.mean_init = -0.37
            self.sigma = 0.05
        self.transformation = transformation
        self.gamma_dist = gamma_dist
        self.kwargs = {'boost': boost} if gamma_dist == ShapeAugmentedGamma else {}
        self.scale_factor = 1.0
        n_wvp = 2 * (self.top_width * self.mid_width + self.mid_width *
                     self.bottom_width + self.bottom_width * self.image_size)
        n_lvp = 2 * 320 * (self.top_width + self.mid_width + self.bottom_width)
        #print("The total number of weight variational parameters is %d" % n_wvp)
        #print("The total number of local variational parameters is %d" % n_lvp)

    def model(self, x):
        x_size = x.size(0)
        with poutine.scale(None, self.scale_factor):
            z_top = pyro.sample("z_top",
                                self.gamma_dist( self.alpha_z.expand(x_size, self.top_width),
                                                 self.alpha_z.expand(x_size, self.top_width)))
            w_top = pyro.sample("w_top",
                self.gamma_dist( self.alpha_w.expand(self.top_width * self.mid_width),
                                 self.beta_w.expand(self.top_width * self.mid_width)))
            w_top = w_top.view(self.top_width, self.mid_width)
            mean_mid = torch.mm(z_top, w_top)

            z_mid = pyro.sample("z_mid",
                        self.gamma_dist(self.alpha_z.expand(x_size, self.mid_width),
                        self.alpha_z.expand(x_size, self.mid_width) / mean_mid))
            w_mid = pyro.sample("w_mid", self.gamma_dist(
                                self.alpha_w.expand(self.mid_width * self.bottom_width),
                                self.beta_w.expand(self.mid_width * self.bottom_width)))
            w_mid = w_mid.view(self.mid_width, self.bottom_width)
            mean_bottom = torch.mm(z_mid, w_mid)

            z_bottom = pyro.sample("z_bottom",
                            self.gamma_dist(self.alpha_z.expand(x_size, self.bottom_width),
                            self.alpha_z.expand(x_size, self.bottom_width) / mean_bottom))
            w_bottom = pyro.sample("w_bottom", self.gamma_dist(
                                   self.alpha_w.expand(self.bottom_width * self.image_size),
                                   self.beta_w.expand(self.bottom_width * self.image_size)))
            w_bottom = w_bottom.view(self.bottom_width, self.image_size)
            mean_obs = torch.mm(z_bottom, w_bottom)

            with pyro.iarange('observe_data'):
                pyro.observe('obs', dist.poisson, x, mean_obs)

    def guide(self, x):
        x_size = x.size(0)

        def sample_zs(name, width):
            alpha_z_q = pyro.param("log_alpha_z_q_%s" % name,
                            Variable((self.alpha_init * torch.ones(x_size, width) + \
                                 self.sigma * torch.randn(x_size, width)).type_as(data.data),
                                 requires_grad=True))
            mean_z_q = pyro.param("log_mean_z_q_%s" % name,
                              Variable((self.mean_init * torch.ones(x_size, width) + \
                                self.sigma * torch.randn(x_size, width)).type_as(data.data),
                                requires_grad=True))

            alpha_z_q, mean_z_q = self.forward(alpha_z_q), self.forward(mean_z_q)
            z = pyro.sample("z_%s" % name, self.gamma_dist(alpha_z_q, alpha_z_q/mean_z_q,
                            **self.kwargs))

        with poutine.scale(None, self.scale_factor):
            sample_zs("top", self.top_width)
            sample_zs("mid", self.mid_width)
            sample_zs("bottom", self.bottom_width)

        def sample_ws(name, width):
            alpha_w_q = pyro.param("log_alpha_w_q_%s" % name,
                                   Variable((self.alpha_init * torch.ones(width) + \
                                   self.sigma * torch.randn(width)).type_as(data.data),
                                   requires_grad=True))
            mean_w_q = pyro.param("log_mean_w_q_%s" % name,
                                  Variable((self.mean_init * torch.ones(width) + \
                                  self.sigma * torch.randn(width)).type_as(data.data),
                                  requires_grad=True))
            alpha_w_q, mean_w_q = self.forward(alpha_w_q), self.forward(mean_w_q)
            w = pyro.sample("w_%s" % name, self.gamma_dist(alpha_w_q, alpha_w_q/mean_w_q,
                            **self.kwargs))

        with poutine.scale(None, self.scale_factor):
            sample_ws("top", self.top_width * self.mid_width)
            sample_ws("mid", self.mid_width * self.bottom_width)
            sample_ws("bottom", self.bottom_width * self.image_size)

    def get_w_stats(self, name):
        alpha_w_q = pyro.param("log_alpha_w_q_%s" % name)
        mean_w_q = pyro.param("log_mean_w_q_%s" % name)
        return np.min(alpha_w_q.data.numpy()), np.max(alpha_w_q.data.numpy()),\
               np.min(mean_w_q.data.numpy()),  np.max(mean_w_q.data.numpy())

    def get_z_stats(self, name):
        alpha_z_q = pyro.param("log_alpha_z_q_%s" % name)
        mean_z_q = pyro.param("log_mean_z_q_%s" % name)
        return np.min(alpha_z_q.data.numpy()), np.max(alpha_z_q.data.numpy()),\
               np.min(mean_z_q.data.numpy()),  np.max(mean_z_q.data.numpy())

    def clip_params(self):
        for param, clip in zip( ("log_alpha", "log_mean"), (-2.25, -4.6) ):
        #for param, clip in zip( ("log_alpha", "log_mean"), (-3.0, -4.0) ):
            for layer in ["top", "mid", "bottom"]:
                for wz in ["_w_q_", "_z_q_"]:
                    pyro.param(param + wz + layer).data.clamp_(min=clip)

    def do_inference(self, optimizer="adam", eta=1.0, n_steps=21, continuous_eval=False, seed=0,
                     num_particles=10, print_stats=False, t=0.10):
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        t0 = time.time()

        if optimizer=="adam":
            opt = optim.Adam({"lr": 0.3, "betas": (0.90, 0.999)})
        else:
            opt = optim.AdagradRMSProp({"eta": eta, "t": t, "zero_start": True, "kappa": 1.0})
        svi = SVI(self.model, self.guide, opt, loss="ELBO", trace_graph=False, mean_field_analytic_entropy=True,
                  include_score_function=True)
        svi_eval = SVI(self.model, self.guide, opt, loss="ELBO", trace_graph=False, mean_field_analytic_entropy=True,
                  num_particles=num_particles)
        elbo_curve, losses = [], []

        for k in range(n_steps):
            loss = svi.step(data)
            self.clip_params()
            losses.append(loss)

            if continuous_eval:
                if (k+1) % 5 == 0 and k < 100:
                    elbo_curve.append( (k+1, -1.0e-7 * svi_eval.evaluate_loss(data)) )
                elif (k+1) % 10==0 and k < 200:
                    elbo_curve.append( (k+1, -1.0e-7 * svi_eval.evaluate_loss(data)) )
                elif (k+1) % 20==0:
                    elbo_curve.append( (k+1, -1.0e-7 * svi_eval.evaluate_loss(data)) )
            elif k==n_steps-1:
                elbo_curve.append( -1.0e-7 * svi_eval.evaluate_loss(data) )

            if k % 250 == 0 and k > 00:
                t_k = time.time()
                print("[epoch %05d] mean elbo: %.5f     elapsed time: %.4f" % (k,
                     -np.mean(losses[-10:]) * 1.0e-7 / self.scale_factor,
                      t_k - t0))
            if print_stats and k % 5 == 0:
                form = ("%.2f %.2f %.2f %.2f   " * 3)[:-3]
                print("[W] " + form % (self.get_w_stats("top") + self.get_w_stats("mid") +
                      self.get_w_stats("bottom") ))
                print("[Z] " + form % (self.get_z_stats("top") + self.get_z_stats("mid") +
                      self.get_z_stats("bottom") ))

        return elbo_curve

grid_search = False

if grid_search:
    n_trials = 3
    boost = 4
    etas = [4.5]
    #etas = [3.0, 4.0, 5.0, 6.0, 7.0]
    ts = [0.10]
    gamma_dists = ["rsvi", "rep"]
    for gamma_dist in gamma_dists:
        for eta in etas:
            for t in ts:
                elbos = []
                for seed in range(n_trials):
                    if gamma_dist=="rep":
                        sgdef = SparseGammaDEF(gamma_dist=RepGamma, boost=boost)
                    else:
                        sgdef = SparseGammaDEF(gamma_dist=ShapeAugmentedGamma, boost=boost)
                    elbos.append(sgdef.do_inference(optimizer="AR", eta=eta, n_steps=100, t=t,
                                                    seed=seed, num_particles=50)[-1])
                print("dist = %s, eta = %.2f, t=%.2f  ==>  mean elbo = %.5f" % (gamma_dist, eta, t, np.mean(elbos)))

# dist = rep, eta = 2.00, t=0.50  ==>  mean elbo = -1.36599  <===
# dist = rsvi, eta = 3.00, t=0.50  ==>  mean elbo = -1.16361  <==
else:
    boost = 4
    t = 0.10
    for gamma_dist in ["rsvi", "rep"]:
        if gamma_dist=="rep":
            sgdef = SparseGammaDEF(gamma_dist=RepGamma, boost=boost)
            eta = 4.5
        else:
            sgdef = SparseGammaDEF(gamma_dist=ShapeAugmentedGamma, boost=boost)
            eta = 4.5
        elbo_curve = sgdef.do_inference(optimizer="AR", eta=eta, n_steps=2001, t=t,
                                        seed=7, num_particles=100, continuous_eval=True)
        with open("elbo_curves/gamma.final.t1.eta45.%s.pkl" % gamma_dist, "wb") as output_file:
            output_file.write(cloudpickle.dumps(elbo_curve, protocol=2))

"""
assert(len(sys.argv)==4), "dist eta boost"
gamma_dist=sys.argv[1]
eta=float(sys.argv[2])
boost=int(sys.argv[3])
tag = "results.adagrad.clip45." + gamma_dist + (".eta%03d" % int(100*eta)) + (".b%d" % boost)
print("doing: %s %.3f %d" % (gamma_dist, eta, boost))

if gamma_dist=="rep":
    sgdef = SparseGammaDEF(gamma_dist=RepGamma, boost=boost)
else:
    sgdef = SparseGammaDEF(gamma_dist=ShapeAugmentedGamma, boost=boost)

elbo_curve = sgdef.do_inference(optimizer="AR", eta=eta)

with open("elbo_curves/%s.pkl" % tag, "wb") as output_file:
    output_file.write(cloudpickle.dumps(elbo_curve, protocol=2))
"""
