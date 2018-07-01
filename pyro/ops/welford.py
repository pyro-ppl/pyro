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

    def _unroll(self, sample):
        unrolled = []
        for rolled in sample:
            unrolled.extend(rolled.reshape(-1).data.tolist())
        return unrolled

    def reset(self):
        self.means = 0.
        self.variances = 0.
        self.n_samples = 0

    def update(self, sample):
        self.n_samples += 1
        delta_pre = sample - self.means
        self.means = self.means + delta_pre / self.n_samples
        delta_post = sample - self.means

        if self.diagonal:
            self.variances += delta_pre * delta_post
        else:
            self.variances += delta_pre * delta_post.reshape(-1, 1)

    def get_estimates(self, regularize=True):
        if self.n_samples < 2:
            raise RuntimeError('Insufficient samples to estimate covariance')
        cov = self.variances / (self.n_samples - 1)
        if regularize:
            # Regularization from stan
            scaled_cov = (self.n_samples / (self.n_samples + 5.)) * cov
            shrinkage = 1e-3 * (5. / (self.n_samples + 5.0))
            if self.diagonal:
                cov = scaled_cov + shrinkage
            else:
                cov = scaled_cov + torch.diagflat(shrinkage)
        return cov
