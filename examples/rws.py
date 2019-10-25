# Example from http://pyro.ai/examples/intro_part_ii.html
import matplotlib.pyplot as plt
import torch
import numpy as np
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


def scale(guess, guess_scale, obs_scale, obs):
    weight = pyro.sample('weight', dist.Normal(guess, guess_scale))
    return pyro.sample('measurement', dist.Normal(weight, obs_scale), obs=obs)


def scale_parametrized_guide(guess, guess_scale, obs_scale, obs):
    loc = pyro.param('loc', torch.tensor(0.))
    log_scale = pyro.param('log_scale', torch.tensor(0.))
    return pyro.sample('weight', dist.Normal(loc, torch.exp(log_scale)))


if __name__ == '__main__':
    guess = 8.5
    guess_scale = 1
    obs_scale = 0.75
    obs = 9.5
    true_q_scale = np.sqrt(1 / (1 / guess_scale**2 + 1 / obs_scale**2))
    true_q_loc = true_q_scale**2 * (guess / guess_scale**2 + obs / obs_scale**2)

    num_particles = 7

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=scale,
                         guide=scale_parametrized_guide,
                         optim=pyro.optim.SGD({'lr': 0.001, 'momentum': 0.1}),
                         loss=pyro.infer.ReweightedWakeSleep(num_particles=num_particles))

    theta_losses, phi_losses, q_loc, q_log_scale = [], [], [], []
    num_steps = 10000
    for t in range(num_steps):
        theta_loss, phi_loss = svi.step(guess, guess_scale, obs_scale, obs)
        theta_losses.append(theta_loss)
        phi_losses.append(phi_loss)
        q_loc.append(pyro.param('loc').item())
        q_log_scale.append(pyro.param('log_scale').item())
        if t % 100 == 0:
            print('Iteration {}: theta loss = {}, phi loss = {}'.format(
                t, theta_loss, phi_loss))

    fig, axs = plt.subplots(4, 1, dpi=200, sharex=True)
    axs[0].plot(theta_losses)
    axs[0].set_ylabel(r'$\theta$ loss')
    axs[1].plot(phi_losses)
    axs[1].set_ylabel(r'$\phi$ loss')
    axs[2].plot(q_loc)
    axs[2].set_ylabel('q loc')
    axs[2].axhline(true_q_loc, color='black')
    axs[3].plot(np.exp(q_log_scale))
    axs[3].set_ylabel('q scale')
    axs[3].axhline(true_q_scale, color='black')
    fig.tight_layout()
    plt.show()
