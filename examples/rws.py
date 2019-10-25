# Example from http://pyro.ai/examples/intro_part_ii.html
import matplotlib.pyplot as plt
import torch
import numpy as np
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


def scale(guess):
    weight = pyro.sample('weight', dist.Normal(guess, 1.))
    return pyro.sample('measurement', dist.Normal(weight, 1.), obs=9.5)


def scale_parametrized_guide(guess):
    loc = pyro.param('loc', torch.tensor(guess))
    log_scale = pyro.param('log_scale', torch.tensor(1.))
    return pyro.sample('weight', dist.Normal(loc, torch.exp(log_scale)))


if __name__ == '__main__':
    guess = 8.5
    num_particles = 7

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=scale,
                         guide=scale_parametrized_guide,
                         optim=pyro.optim.SGD({'lr': 0.001, 'momentum': 0.1}),
                         loss=pyro.infer.ReweightedWakeSleep(num_particles=num_particles))

    theta_losses, phi_losses, q_loc, q_log_scale = [], [], [], []
    num_steps = 2500
    for t in range(num_steps):
        theta_loss, phi_loss = svi.step(guess)
        theta_losses.append(theta_loss)
        phi_losses.append(phi_loss)
        q_loc.append(pyro.param('loc').item())
        q_log_scale.append(pyro.param('log_scale').item())

    fig, axs = plt.subplots(4, 1, dpi=200, sharex=True)
    axs[0].plot(theta_losses)
    axs[0].set_ylabel(r'$\theta$ loss')
    axs[1].plot(phi_losses)
    axs[1].set_ylabel(r'$\phi$ loss')
    axs[2].plot(q_loc)
    axs[2].set_ylabel('q loc')
    axs[2].axhline(9.14, color='black')
    axs[3].plot(np.exp(q_log_scale))
    axs[3].set_ylabel('q scale')
    axs[3].axhline(0.6, color='black')
    fig.tight_layout()
    plt.show()
