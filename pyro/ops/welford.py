from collections import OrderedDict
import torch


class WelfordCovariance(object):
    """
    Implements Welford's online scheme for estimating (co)variance (see :math:`[1]`).
    Useful for adapting diagonal and dense mass structures for HMC.

    **References**

    [1] `The Art of Computer Programming`,
    Donald E. Knuth
    """
    def __init__(self, diagonal=True):
        self.diagonal = diagonal
        self.reset()

    def _deltas(self, z):
        deltas = OrderedDict()
        for site_name, site_value in z.items():
            mean = self.means.get(site_name, 0.)
            delta_pre = site_value - mean
            updated_mean = mean + delta_pre / self.n_samples
            delta_post = site_value - updated_mean
            deltas[site_name] = (updated_mean, delta_pre, delta_post)
        return deltas

    def _unroll(self, z):
        unrolled = OrderedDict()
        for site_name, rolled in z.items():
            for i, x in enumerate(rolled.view(-1)):
                unrolled["{}_{}".format(site_name, i)] = x
        return unrolled

    def reset(self):
        self.means = OrderedDict()
        self.variances = OrderedDict()
        self.n_samples = 0

    def update(self, z):
        self.n_samples += 1
        deltas = self._deltas(self._unroll(z))
        for i, (site_x, (mean_x, delta_x_pre, delta_x_post)) in enumerate(deltas.items()):
            self.means[site_x] = mean_x
            for j, (site_y, (mean_y, delta_y_pre, delta_y_post)) in enumerate(deltas.items()):
                # Only compute the upper triangular covariance.
                if i == j or (i < j and not self.diagonal):
                    self.variances[(site_x, site_y)] = self.variances.get((site_x, site_y), 0.) + \
                                                       delta_x_pre * delta_y_post

    def get_estimates(self, regularize=True):
        if self.n_samples < 2:
            raise RuntimeError('Insufficient samples to estimate covariance')
        rows = []
        for i, site_x in enumerate(self.means.keys()):
            row = []
            for j, site_y in enumerate(self.means.keys()):
                if i == j or not self.diagonal:
                    key = (site_x, site_y) if i <= j else (site_y, site_x)
                    estimate = self.variances[key] / (self.n_samples - 1)
                    if regularize:
                        # Regularization from stan
                        scaled_estimate = (self.n_samples / (self.n_samples + 5.)) * estimate
                        shrinkage = 1e-3 * (5. / (self.n_samples + 5.0)) if i == j else 0.
                        estimate = scaled_estimate + shrinkage
                    row.append(estimate)
            rows.append(row)
        return torch.tensor(rows)