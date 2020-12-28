# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
import torch
from pyro.nn import PyroModule


def mg2k(m, g):
    """Convert from (m, g) indexing to k indexing."""
    return 2*m + 1 - g


class profile(PyroModule):

    def __init__(self, M, epsilon=1e-32):
        super().__init__()
        self.M = M
        self.K = 2*(M+1)
        self.epsilon = epsilon

        self._make_transfer()

    def _make_transfer(self):
        """Set up linear transformations (transfer matrices) for converting
        from profile HMM parameters to standard HMM parameters."""
        M, K = self.M, self.K

        # Overview:
        # r -> insertion parameters
        # u -> deletion parameters
        # indices: m in {0, ..., M} and j in {0, 1, 2}; final index corresponds
        # to simplex dimensions, i.e. 1 - r and r (in that order)
        # null -> locations in the transition matrix equal to 0
        # ...transf_0 -> initial transition vector
        # ...transf -> transition matrix
        self.register_buffer('r_transf_0',
                             torch.zeros((M+1, 3, 2, K)))
        self.register_buffer('u_transf_0',
                             torch.zeros((M+1, 3, 2, K)))
        self.register_buffer('null_transf_0',
                             torch.zeros((K,)))
        m, g = -1, 0
        for mp in range(M+1):
            for gp in range(2):
                kp = mg2k(mp, gp)
                if m + 1 - g == mp and gp == 0:
                    self.r_transf_0[m+1-g, g, 0, kp] = 1
                    self.u_transf_0[m+1-g, g, 0, kp] = 1

                elif m + 1 - g < mp and mp <= M and gp == 0:
                    self.r_transf_0[m+1-g, g, 0, kp] = 1
                    self.u_transf_0[m+1-g, g, 1, kp] = 1
                    for mpp in range(m+2-g, mp):
                        self.r_transf_0[mpp, 2, 0, kp] = 1
                        self.u_transf_0[mpp, 2, 1, kp] = 1
                    self.r_transf_0[mp, 2, 0, kp] = 1
                    self.u_transf_0[mp, 2, 0, kp] = 1

                elif m + 1 - g == mp and gp == 1:
                    self.r_transf_0[m+1-g, g, 1, kp] = 1

                elif m + 1 - g < mp and mp <= M and gp == 1:
                    self.r_transf_0[m+1-g, g, 0, kp] = 1
                    self.u_transf_0[m+1-g, g, 1, kp] = 1
                    for mpp in range(m+2-g, mp):
                        self.r_transf_0[mpp, 2, 0, kp] = 1
                        self.u_transf_0[mpp, 2, 1, kp] = 1
                    self.r_transf_0[mp, 2, 1, kp] = 1

                else:
                    self.null_transf_0[kp] = 1
        self.u_transf_0[-1, :, :, :] = 0.

        self.register_buffer('r_transf',
                             torch.zeros((M+1, 3, 2, K, K)))
        self.register_buffer('u_transf',
                             torch.zeros((M+1, 3, 2, K, K)))
        self.register_buffer('null_transf',
                             torch.zeros((K, K)))
        for m in range(M+1):
            for g in range(2):
                for mp in range(M+1):
                    for gp in range(2):
                        k, kp = mg2k(m, g), mg2k(mp, gp)
                        if m + 1 - g == mp and gp == 0:
                            self.r_transf[m+1-g, g, 0, k, kp] = 1
                            self.u_transf[m+1-g, g, 0, k, kp] = 1

                        elif m + 1 - g < mp and mp <= M and gp == 0:
                            self.r_transf[m+1-g, g, 0, k, kp] = 1
                            self.u_transf[m+1-g, g, 1, k, kp] = 1
                            for mpp in range(m+2-g, mp):
                                self.r_transf[mpp, 2, 0, k, kp] = 1
                                self.u_transf[mpp, 2, 1, k, kp] = 1
                            self.r_transf[mp, 2, 0, k, kp] = 1
                            self.u_transf[mp, 2, 0, k, kp] = 1

                        elif m + 1 - g == mp and gp == 1:
                            self.r_transf[m+1-g, g, 1, k, kp] = 1

                        elif m + 1 - g < mp and mp <= M and gp == 1:
                            self.r_transf[m+1-g, g, 0, k, kp] = 1
                            self.u_transf[m+1-g, g, 1, k, kp] = 1
                            for mpp in range(m+2-g, mp):
                                self.r_transf[mpp, 2, 0, k, kp] = 1
                                self.u_transf[mpp, 2, 1, k, kp] = 1
                            self.r_transf[mp, 2, 1, k, kp] = 1

                        elif not (m == M and mp == M and g == 0 and gp == 0):
                            self.null_transf[k, kp] = 1
        self.u_transf[-1, :, :, :, :] = 0.

        self.register_buffer('vx_transf',
                             torch.zeros((M+1, K)))
        self.register_buffer('vc_transf',
                             torch.zeros((M+1, K)))
        for m in range(M+1):
            for g in range(2):
                k = mg2k(m, g)
                if g == 0:
                    self.vx_transf[m, k] = 1
                elif g == 1:
                    self.vc_transf[m, k] = 1

    def forward(self, ancestor_seq_logits, insert_seq_logits,
                insert_logits, delete_logits, subsitute_logits=None):
        """Assemble the HMM parameters based on the transfer matrices."""

        initial_logits = (
            torch.einsum('...ijk,...ijkl->...l',
                         delete_logits, self.u_transf_0) +
            torch.einsum('...ijk,...ijkl->...l',
                         insert_logits, self.r_transf_0) +
            (-1/self.epsilon)*self.null_transf_0)
        transition_logits = (
             torch.einsum('...ijk,...ijklf->...lf',
                          delete_logits, self.u_transf) +
             torch.einsum('...ijk,...ijklf->...lf',
                          insert_logits, self.r_transf) +
             (-1/self.epsilon)*self.null_transf)
        seq_logits = (
             torch.einsum('...ij,...ik->...kj',
                          ancestor_seq_logits, self.vx_transf) +
             torch.einsum('...ij,...ik->...kj',
                          insert_seq_logits, self.vc_transf))

        # Option to include the substitution matrix.
        if subsitute_logits is not None:
            observation_logits = torch.logsumexp(
                seq_logits[:, :, None] + subsitute_logits[None, :, :], dim=1)
        else:
            observation_logits = seq_logits

        return initial_logits, transition_logits, observation_logits
