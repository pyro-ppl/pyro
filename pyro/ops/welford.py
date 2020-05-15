# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


class WelfordCovariance:
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

    def reset(self):
        self._mean = 0.
        self._m2 = 0.
        self.n_samples = 0

    def update(self, sample):
        self.n_samples += 1
        delta_pre = sample - self._mean
        self._mean = self._mean + delta_pre / self.n_samples
        delta_post = sample - self._mean

        if self.diagonal:
            self._m2 += delta_pre * delta_post
        else:
            self._m2 += torch.ger(delta_post, delta_pre)

    def get_covariance(self, regularize=True):
        if self.n_samples < 2:
            raise RuntimeError('Insufficient samples to estimate covariance')
        cov = self._m2 / (self.n_samples - 1)
        if regularize:
            # Regularization from stan
            scaled_cov = (self.n_samples / (self.n_samples + 5.)) * cov
            shrinkage = 1e-3 * (5. / (self.n_samples + 5.0))
            if self.diagonal:
                cov = scaled_cov + shrinkage
            else:
                scaled_cov.view(-1)[::scaled_cov.size(0) + 1] += shrinkage
                cov = scaled_cov
        return cov


class WelfordArrowheadCovariance:
    """
    Likes :class:`WelfordCovariance` but generalized to the arrowhead structure.
    """
    def __init__(self, head_size=0):
        self.head_size = head_size
        self.reset()

    def reset(self):
        self._mean = 0.
        self._m2_top = 0.  # upper part, shape: head_size x matrix_size
        self._m2_bottom_diag = 0.  # lower right part, shape: (matrix_size - head_size)
        self.n_samples = 0

    def update(self, sample):
        self.n_samples += 1
        delta_pre = sample - self._mean
        self._mean = self._mean + delta_pre / self.n_samples
        delta_post = sample - self._mean
        if self.head_size > 0:
            self._m2_top = self._m2_top + torch.ger(delta_post[:self.head_size], delta_pre)
        else:
            self._m2_top = sample.new_empty(0, sample.size(0))
        self._m2_bottom_diag = self._m2_bottom_diag + delta_post[self.head_size:] * delta_pre[self.head_size:]

    def get_covariance(self, regularize=True):
        """
        Gets the covariance in arrowhead form: (top, bottom_diag) where `top = cov[:head_size]`
        and `bottom_diag = cov.diag()[head_size:]`.
        """
        if self.n_samples < 2:
            raise RuntimeError('Insufficient samples to estimate covariance')
        top = self._m2_top / (self.n_samples - 1)
        bottom_diag = self._m2_bottom_diag / (self.n_samples - 1)
        if regularize:
            top = top * (self.n_samples / (self.n_samples + 5.))
            bottom_diag = bottom_diag * (self.n_samples / (self.n_samples + 5.))
            shrinkage = 1e-3 * (5. / (self.n_samples + 5.0))
            top.view(-1)[::top.size(-1) + 1] += shrinkage
            bottom_diag = bottom_diag + shrinkage

        return top, bottom_diag
