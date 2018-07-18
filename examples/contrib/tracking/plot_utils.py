import torch
from matplotlib import pyplot


def plot_solution(observations, p_exists, positions, true_positions, args, message='', fig=None):
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


def plot_exists_prob(p_exists):
    p_exists = p_exists.detach().numpy()
    pyplot.figure(figsize=(6, 4)).patch.set_color('white')
    pyplot.plot(sorted(p_exists))
    pyplot.ylim(0, None)
    pyplot.xlim(0, len(p_exists))
    pyplot.ylabel('p_exists')
    pyplot.xlabel('rank')
    pyplot.title('Prob(exists) of {} potential objects, total = {:0.2f}'.format(
        len(p_exists), p_exists.sum()))
    pyplot.tight_layout()
