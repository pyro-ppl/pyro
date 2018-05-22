from __future__ import print_function

import argparse

import numpy as np
import torch

import pyro
from pyro.distributions import Normal, Uniform
from pyro.infer import EmpiricalMarginal, Importance

"""
Samantha really likes physics---but she likes Pyro even more. Instead of using
calculus to do her physics lab homework (which she could easily do), she's going
to use bayesian inference. The problem setup is as follows. In lab she observed
a little box slide down an inclined plane (length of 2 meters and with an incline of
30 degrees) 20 times. Each time she measured and recorded the descent time. The timing
device she used has a known measurement error of 20 milliseconds. Using the observed
data, she wants to infer the coefficient of friction mu between the box and the inclined
plane. She already has (deterministic) python code that can simulate the amount of time
that it takes the little box to slide down the inclined plane as a function of mu. Using
Pyro, she can reverse the simulator and infer mu from the observed descent times.
"""

little_g = 9.8  # m/s/s
mu0 = 0.12  # actual coefficient of friction in the experiment
time_measurement_sigma = 0.02  # observation noise in seconds (known quantity)


# the forward simulator, which does numerical integration of the equations of motion
# in steps of size dt, and optionally includes measurement noise

def simulate(mu, length=2.0, phi=np.pi / 6.0, dt=0.005, noise_sigma=None):
    T = torch.zeros(())
    velocity = torch.zeros(())
    displacement = torch.zeros(())
    acceleration = torch.tensor(little_g * np.sin(phi)) - \
        torch.tensor(little_g * np.cos(phi)) * mu

    if acceleration.numpy() <= 0.0:  # the box doesn't slide if the friction is too large
        return torch.tensor(1.0e5)   # return a very large time instead of infinity

    while displacement.numpy() < length:  # otherwise slide to the end of the inclined plane
        displacement += velocity * dt
        velocity += acceleration * dt
        T += dt

    if noise_sigma is None:
        return T
    else:
        return T + noise_sigma * torch.randn(())


# analytic formula that the simulator above is computing via
# numerical integration (no measurement noise)

def analytic_T(mu, length=2.0, phi=np.pi / 6.0):
    numerator = 2.0 * length
    denominator = little_g * (np.sin(phi) - mu * np.cos(phi))
    return np.sqrt(numerator / denominator)


# generate N_obs observations using simulator and the true coefficient of friction mu0
print("generating simulated data using the true coefficient of friction %.3f" % mu0)
N_obs = 20
torch.manual_seed(2)
observed_data = torch.tensor([simulate(torch.tensor(mu0), noise_sigma=time_measurement_sigma)
                              for _ in range(N_obs)])
observed_mean = np.mean([T.item() for T in observed_data])


# define model with uniform prior on mu and gaussian noise on the descent time
def model(observed_data):
    mu_prior = Uniform(0.0, 1.0)
    mu = pyro.sample("mu", mu_prior)

    def observe_T(T_obs, obs_name):
        T_simulated = simulate(mu)
        T_obs_dist = Normal(T_simulated, torch.tensor(time_measurement_sigma))
        pyro.sample(obs_name, T_obs_dist, obs=T_obs)

    for i, T_obs in enumerate(observed_data):
        observe_T(T_obs, "obs_%d" % i)

    return mu


def main(args):
    # create an importance sampler (the prior is used as the proposal distribution)
    importance = Importance(model, guide=None, num_samples=args.num_samples)
    # get posterior samples of mu (which is the return value of model)
    # from the raw execution traces provided by the importance sampler.
    print("doing importance sampling...")
    emp_marginal = EmpiricalMarginal(importance.run(observed_data))

    # calculate statistics over posterior samples
    posterior_mean = emp_marginal.mean
    posterior_std_dev = emp_marginal.variance.sqrt()

    # report results
    inferred_mu = posterior_mean.item()
    inferred_mu_uncertainty = posterior_std_dev.item()
    print("the coefficient of friction inferred by pyro is %.3f +- %.3f" %
          (inferred_mu, inferred_mu_uncertainty))

    # note that, given the finite step size in the simulator, the simulated descent times will
    # not precisely match the numbers from the analytic result.
    # in particular the first two numbers reported below should match each other pretty closely
    # but will be systematically off from the third number
    print("the mean observed descent time in the dataset is: %.4f seconds" % observed_mean)
    print("the (forward) simulated descent time for the inferred (mean) mu is: %.4f seconds" %
          simulate(posterior_mean).item())
    print(("disregarding measurement noise, elementary calculus gives the descent time\n" +
           "for the inferred (mean) mu as: %.4f seconds") % analytic_T(posterior_mean.item()))

    """
    ################## EXERCISE ###################
    # vectorize the computations in this example! #
    ###############################################
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=500, type=int)
    args = parser.parse_args()
    main(args)
