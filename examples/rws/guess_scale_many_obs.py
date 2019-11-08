# Example from http://pyro.ai/examples/intro_part_ii.html
import matplotlib.pyplot as plt
import torch
import numpy as np
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


def scale(guess_init, guess_scale, obs_scale, observations=None):
    guess_loc = pyro.param('guess', torch.tensor(guess_init))
    with pyro.plate('obss'):
        weight = pyro.sample('weight', dist.Normal(guess_loc.expand(len(obss)), guess_scale))
        return pyro.sample('measurement', dist.Normal(weight, obs_scale),
                           obs=torch.tensor(observations['measurement']))


def scale_parametrized_guide(guess_init, guess_scale, obs_scale, observations=None):
    loc_mult = pyro.param('loc_mult', torch.tensor(1.))
    loc_add = pyro.param('loc_add', torch.tensor(0.))
    log_scale = pyro.param('log_scale', torch.tensor(0.))
    with pyro.plate('obss'):
        obss_tensor = torch.tensor(observations['measurement'])
        return pyro.sample('weight', dist.Normal(loc_mult * obss_tensor + loc_add,
                                                 torch.exp(log_scale)))


if __name__ == '__main__':
    guess_init = 8.5
    guess_scale = 1.0
    obs_scale = 0.75
    obss = [9.5, 9.1, 9.2]
    true_guess = np.mean(obss)
    true_q_scale = np.sqrt(1 / (1 / guess_scale**2 + 1 / obs_scale**2))
    true_q_loc_mult = true_q_scale**2 / obs_scale**2
    true_q_loc_add = true_q_scale**2 * true_guess / guess_scale**2

    num_particles = 100
    vectorize = True

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=scale,
                         guide=scale_parametrized_guide,
                         optim=pyro.optim.Adam({'lr': 0.1}),
                         loss=pyro.infer.ReweightedWakeSleep(num_particles=num_particles,
                                                             vectorize_particles=vectorize,
                                                             insomnia=1.))

    theta_losses, phi_losses, guesses, q_loc_mult, q_loc_add, q_log_scale = [], [], [], [], [], []
    num_steps = 10000
    for t in range(num_steps):
        theta_loss, phi_loss = svi.step(guess_init, guess_scale, obs_scale,
                                        observations={'measurement': obss})
        theta_losses.append(theta_loss)
        phi_losses.append(phi_loss)
        guesses.append(pyro.param('guess').item())
        q_loc_mult.append(pyro.param('loc_mult').item())
        q_loc_add.append(pyro.param('loc_add').item())
        q_log_scale.append(pyro.param('log_scale').item())
        if t % 100 == 0:
            print('Iteration {:<5}: theta loss = {}, phi loss = {}'.format(
                t, theta_loss, phi_loss))

    fig, axs = plt.subplots(6, 1, dpi=200, sharex=True, figsize=(6, 12))
    axs[0].plot(theta_losses)
    axs[0].set_ylabel(r'$\theta$ loss')
    axs[1].plot(phi_losses)
    axs[1].set_ylabel(r'$\phi$ loss')
    axs[2].plot(q_loc_mult)
    axs[2].set_ylabel('q loc mult')
    axs[2].axhline(true_q_loc_mult, color='black')
    axs[3].plot(q_loc_add)
    axs[3].set_ylabel('q loc add')
    axs[3].axhline(true_q_loc_add, color='black')
    axs[4].plot(np.exp(q_log_scale))
    axs[4].set_ylabel('q scale')
    axs[4].axhline(true_q_scale, color='black')
    axs[5].plot(guesses)
    axs[5].set_ylabel('guess')
    axs[5].axhline(true_guess, color='black')
    axs[-1].set_xlabel('iteration')
    fig.tight_layout()
    filename = 'guess_scale_many_obs.pdf'
    fig.savefig(filename, bbox_inches='tight')
    print('Saved to {}'.format(filename))
