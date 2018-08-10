import torch
from matplotlib import pyplot
import time
import warnings
import os
import errno


def init_plot_utils(args):
    viz = None
    if args.visdom:
        from visdom import Visdom
        viz = Visdom()
        startup_sec = 1
        while not viz.check_connection() and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1
        if not viz.check_connection():
            warnings.warn('No connection could be formed quickly')
            viz = None

    full_exp_dir = None
    if args.exp_name is not None:
        full_exp_dir = os.path.join(args.exp_dir, args.exp_name)
        try:
            os.makedirs(full_exp_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                warnings.warn('Something went wrong creating {}'.format(full_exp_dir))
                full_exp_dir = None

    return viz, full_exp_dir


def plot_solution(observations, p_exists, positions, true_positions, args,
                  emission_noise_scale=None, message='', fig=None, viz=None, env='main', fig_dir=None):
    with torch.no_grad():
        if fig is None:
            fig = pyplot.figure(figsize=(12, 6))
            fig.patch.set_color('white')
        pyplot.plot(true_positions.numpy(), 'k--')
        confidence = observations[..., -1]
        is_observed = (confidence > 0)
        pos = observations[..., 0]
        time = torch.arange(args.num_frames).unsqueeze(-1).expand_as(pos)
        pyplot.scatter(time[is_observed].view(-1).numpy(),
                       pos[is_observed].view(-1).numpy(), color='k', marker='+',
                       s=8 * 10**confidence[is_observed].detach().view(-1).numpy(),
                       label='observation')
        for i in range(p_exists.shape[0]):
            position = positions[:, i].detach().numpy()
            pyplot.plot(position, alpha=p_exists[i].item(), color='C0')
            if emission_noise_scale is not None and p_exists[i].item() > 1e-3:
                pyplot.fill_between(torch.arange(args.num_frames).numpy(),
                                    position - emission_noise_scale,
                                    position + emission_noise_scale,
                                    alpha=max(p_exists[i].item() - 0.5, 0.03), color='C0')
        if args.expected_num_objects == 1:
            mean = (p_exists * positions).sum(-1) / p_exists.sum(-1)
            pyplot.plot(mean.detach().numpy(), 'r--', alpha=0.5, label='mean')
        pyplot.title('Truth, observations, and {:0.1f} predicted tracks {}'.format(
                     p_exists.sum().item(), message))
        pyplot.plot([], 'k--', label='truth')
        pyplot.plot([], color='C0', label='prediction')
        pyplot.legend(loc='best')
        pyplot.xlabel('time step')
        pyplot.ylabel('position')
        pyplot.tight_layout()
        if fig_dir is not None:
            fig.savefig(os.path.join(fig_dir, "solution.png"), bbox_inches='tight', dpi=100)
        if viz is not None:
            viz.matplot(pyplot, env=env)
        else:
            pyplot.close(fig)

def plot_exists_prob(p_exists, viz=None, env='main', fig_dir=None):
    p_exists = p_exists.detach().numpy()
    if viz is not None:
        viz.line(Y=sorted(p_exists),
                 X=torch.arange(p_exists.size).numpy(),
                 opts=dict(xlabel='rank', ylabel='p_exists',
                           title='Prob(exists) of {} potential objects, total = {:0.2f}'.format(len(p_exists),
                                                                                                p_exists.sum())),
                 env=env)
    if (fig_dir is not None) or (viz is None):
        fig = pyplot.figure(figsize=(6, 4))
        fig.patch.set_color('white')
        pyplot.plot(sorted(p_exists))
        pyplot.ylim(0, None)
        pyplot.xlim(0, len(p_exists))
        pyplot.ylabel('p_exists')
        pyplot.xlabel('rank')
        pyplot.title('Prob(exists) of {} potential objects, total = {:0.2f}'.format(
            len(p_exists), p_exists.sum()))
        pyplot.tight_layout()
        if fig_dir is not None:
            fig.savefig(os.path.join(fig_dir, "exists_prob.png"), bbox_inches='tight', dpi=100)
            pyplot.close(fig)


def plot_list(list_values, title, viz=None, env='main', fig_dir=None):
    if viz is not None:
        viz.line(Y=list_values, X=torch.arange(len(list_values)).numpy(), opts=dict(title=title), env=env)

    if (fig_dir is not None) or (viz is None):
        fig = pyplot.figure(figsize=(6, 4))
        fig.patch.set_color('white')
        pyplot.plot(list_values)
        pyplot.ylim(0, None)
        pyplot.xlim(0, len(list_values))
        pyplot.title(title)
        pyplot.tight_layout()
        if fig_dir is not None:
            fig.savefig(os.path.join(fig_dir, title.lower().replace(' ', '_') + ".png"), bbox_inches='tight', dpi=100)
            pyplot.close(fig)
