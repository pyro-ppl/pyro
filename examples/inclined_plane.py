from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import pyro
from pyro.distributions import Uniform, DiagNormal
from pyro.infer.kl_qp import KL_QP
import torch.optim as optim
import sys

"""
Samantha really likes physics---but she likes pyro even more. Instead of using
calculus to do her physics lab homework (which she could easily do), she's going
to use bayesian inference. The problem setup is as follows. In lab she observed
a little box slide down an inclined plane (length of 2 meters and with an incline of
30 degrees) 10 times. Each time she measured and recorded the descent time. The timing
device she used has a known measurement error of 20 milliseconds. Using the observed
data, she wants to infer the coefficient of friction mu between the box and the inclined
plane. She already has python code that can simulate the amount of time that it takes
the little box to slide down the inclined plane as a function of mu. Using pyro, she
can reverse the simulator and infer mu from the observed descent times.
"""

little_g = 9.8   # m/s/s
mu0 = 0.12  # actual coefficient of friction in the experiment
time_measurement_sigma = 0.02  # observation noise in seconds (known quantity)

# the forward simulator, which does numerical integration of the equations of motion
# in steps of size dx, and optionally includes measurement noise

def simulate(mu, length=2.0, phi=np.pi / 6.0, dx=0.01, noise_sigma=None):
    T = Variable(torch.zeros(1))
    velocity = Variable(torch.zeros(1))
    displacement = Variable(torch.zeros(1))
    acceleration = Variable(torch.Tensor([little_g * np.sin(phi)])) -\
                           (Variable(torch.Tensor([little_g * np.cos(phi)])) * mu)

    if acceleration.data[0] <= 0.0:         # the box doesn't slide if the friction is too large
        return Variable(torch.Tensor([np.inf]))

    while displacement.data[0] < length:  # otherwise slide to the end of the inclined plane
        velocity = torch.sqrt(velocity * velocity + 2.0 * dx * acceleration)
        displacement += dx
        T += dx / velocity

    if noise_sigma is None:
        return T.unsqueeze(0)
    else:
        return T + Variable(noise_sigma * torch.randn(1))

# analytic formula that the simulator above is computing via
# numerical integration (no measurement noise)

def analytic_T(mu, length=2.0, phi=np.pi / 6.0):
    numerator = 2.0 * length
    denominator = little_g * (np.sin(phi) - mu * np.cos(phi))
    return np.sqrt(numerator / denominator)

# generate N_obs observations using simulator and the true coefficient of friction mu0
print("generating simulated data using the true coefficient of friction %.3f" % mu0)
N_obs = 10
observed_data = [simulate(Variable(torch.Tensor([mu0])), noise_sigma=time_measurement_sigma)
                 for _ in range(N_obs)]
observed_mean = np.mean([T.data[0] for T in observed_data])

# define model with uniform prior on mu and gaussian noise on the descent time

def model(observed_data):
    mu_prior = Uniform(Variable(torch.zeros(1)), Variable(torch.ones(1)))
    mu = pyro.sample("mu", mu_prior)

    def observe_T(T_obs, obs_name):
        T_simulated = simulate(mu)
        T_obs_dist = DiagNormal(T_simulated, Variable(torch.Tensor([time_measurement_sigma])))
        T = pyro.observe(obs_name, T_obs_dist, T_obs)

    pyro.map_data("map", observed_data, lambda i, x: observe_T(x, "obs_%d" % i), batch_size=1)
    return mu

# define a gaussian variational approximation for the posterior over mu

def guide(observed_data):
    mean_mu = pyro.param("mean_mu", Variable(torch.Tensor([0.25]), requires_grad=True))
    log_sigma_mu = pyro.param("log_sigma_mu", Variable(torch.Tensor([-4.0]), requires_grad=True))
    sigma_mu = torch.exp(log_sigma_mu)
    mu = pyro.sample("mu", DiagNormal(mean_mu, sigma_mu))
    pyro.map_data("map", observed_data, lambda i, x: None, batch_size=1)
    return mu

# do variational inference using KL_QP
print("doing inference with simulated data")
verbose = False
n_steps = 3001
kl_optim = KL_QP(model, guide, pyro.optim(optim.Adam, {"lr": 0.003, "betas": (0.93, 0.993)}))
for step in range(n_steps):
    loss = kl_optim.step(observed_data)
    if step % 100 == 0:
        if verbose:
            print("[epoch %d] mean_mu: %.3f" % (step, pyro.param("mean_mu").data[0, 0]))
            print("[epoch %d] sigma_mu: %.3f" % (step,
                                                 torch.exp(pyro.param("log_sigma_mu")).data[0, 0]))
        else:
            print(".", end='')
        sys.stdout.flush()

# report results
inferred_mu = pyro.param("mean_mu").data[0]
inferred_mu_uncertainty = torch.exp(pyro.param("log_sigma_mu")).data[0]
print("\nthe coefficient of friction inferred by pyro is %.3f +- %.3f" %
      (inferred_mu, inferred_mu_uncertainty))

# note that, given the finite step size in the simulator, the simulated descent times will
# not precisely match the numbers from the analytic result.
# in particular the first two numbers reported below should match each other pretty closely
# but will be systematically off from the third number
print("the mean observed descent time in the dataset is: %.4f seconds" % observed_mean)
print("the (forward) simulated descent time for the inferred (mean) mu is: %.4f seconds" %
      simulate(pyro.param("mean_mu")).data[0])
print("elementary calulus gives the descent time for the inferred (mean) mu as: %.4f seconds" %
      analytic_T(pyro.param("mean_mu").data[0]))
