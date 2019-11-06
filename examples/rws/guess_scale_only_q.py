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
    guess_scale = 1.0
    obs_scale = 0.75
    obs = 9.5
    true_q_scale = np.sqrt(1 / (1 / guess_scale**2 + 1 / obs_scale**2))
    true_q_loc = true_q_scale**2 * (guess / guess_scale**2 + obs / obs_scale**2)

    num_particles = 10
    vectorize = True

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=scale,
                         guide=scale_parametrized_guide,
                         optim=pyro.optim.SGD({'lr': 0.01, 'momentum': 0.1}),
                         loss=pyro.infer.ReweightedWakeSleep(num_particles=num_particles,
                                                             vectorize_particles=vectorize))

    phi_losses, guesses, q_loc, q_log_scale = [], [], [], []
    num_iterations = 5000
    for i in range(num_iterations):
        _, phi_loss = svi.step(guess, guess_scale, obs_scale, obs)
        phi_losses.append(phi_loss)
        q_loc.append(pyro.param('loc').item())
        q_log_scale.append(pyro.param('log_scale').item())
        if i % 100 == 0:
            print('Iteration {:<5}: phi loss = {}'.format(i, phi_loss))

    fig, axs = plt.subplots(3, 1, dpi=200, sharex=True, figsize=(6, 6))
    axs[0].plot(phi_losses)
    axs[0].set_ylabel(r'$\phi$ loss')
    axs[1].plot(q_loc)
    axs[1].set_ylabel('q loc')
    axs[1].axhline(true_q_loc, color='black')
    axs[2].plot(np.exp(q_log_scale))
    axs[2].set_ylabel('q scale')
    axs[2].axhline(true_q_scale, color='black')
    axs[-1].set_xlabel('iteration')
    fig.tight_layout()
    filename = 'guess_scale_only_q.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))
