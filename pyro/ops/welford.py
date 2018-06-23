from collections import defaultdict

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

    def _deltas(self, sample):
        deltas = []
        for i, x in enumerate(sample):
            if i > len(self.means) - 1:
                self.means.append(0.)
            mean = self.means[i]
            delta_pre = x - mean
            updated_mean = mean + delta_pre / self.n_samples
            delta_post = x - updated_mean
            deltas.append((updated_mean, delta_pre, delta_post))
        return deltas

    def _unroll(self, sample):
        unrolled = []
        for rolled in sample:
            unrolled.extend(rolled.reshape(-1).data.tolist())
        return unrolled

    def reset(self):
        self.means = []
        self.variances = defaultdict(float)
        self.n_samples = 0

    def update(self, sample):
        self.n_samples += 1
        deltas = self._deltas(self._unroll(sample))
        for i, (mean_x, delta_x_pre, delta_x_post) in enumerate(deltas):
            self.means[i] = mean_x
            for j, (mean_y, delta_y_pre, delta_y_post) in enumerate(deltas):
                # Only compute the upper triangular covariance.
                if i == j or (i < j and not self.diagonal):
                    self.variances[(i, j)] += delta_x_pre * delta_y_post

    def get_estimates(self, regularize=True):
        if self.n_samples < 2:
            raise RuntimeError('Insufficient samples to estimate covariance')
        rows = []
        for i in range(len(self.means)):
            row = []
            for j in range(len(self.means)):
                if i == j or not self.diagonal:
                    key = (i, j) if i <= j else (j, i)
                    estimate = self.variances[key] / (self.n_samples - 1)
                    if regularize:
                        # Regularization from stan
                        scaled_estimate = (self.n_samples / (self.n_samples + 5.)) * estimate
                        shrinkage = 1e-3 * (5. / (self.n_samples + 5.0)) if i == j else 0.
                        estimate = scaled_estimate + shrinkage
                    row.append(estimate)
            rows.append(row)
        return torch.tensor(rows)
