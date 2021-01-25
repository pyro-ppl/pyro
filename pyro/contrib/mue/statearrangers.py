# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn


class Profile(nn.Module):
    """
    Profile HMM state arrangement. Parameterizes an HMM according to
    Equation S40 in [1]. For further background on profile HMMs see [2].

    **References**

    [1] E. N. Weinstein, D. S. Marks (2020)
    "Generative probabilistic biological sequence models that account for
    mutational variability"
    https://www.biorxiv.org/content/10.1101/2020.07.31.231381v1.full.pdf
    [2] R. Durbin, S. R. Eddy, A. Krogh, and G. Mitchison (1998)
    "Biological sequence analysis: probabilistic models of proteins and nucleic
    acids"
    Cambridge university press

    :param M: Length of precursor (ancestral) sequence.
    :type M: int
    :param epsilon: Small value for approximate zeros in log space.
    :type epsilon: float
    """
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

    def forward(self, precursor_seq_logits, insert_seq_logits,
                insert_logits, delete_logits, substitute_logits=None):
        """
        Assemble HMM parameters given profile parameters.

        :param ~torch.Tensor precursor_seq_logits: Initial (relaxed) sequence
            *log(x)*. Should have rightmost dimension ``(M+1, D)`` and be
            broadcastable to ``(batch_size, M+1, D)``, where
            D is the latent alphabet size. Should be normalized to one along the
            final axis, i.e. ``precursor_seq_logits.logsumexp(-1) = zeros``.
        :param ~torch.Tensor insert_seq_logits: Insertion sequence *log(c)*.
            Should have rightmost dimension ``(M+1, D)`` and be broadcastable
            to ``(batch_size, M+1, D)``. Should be normalized
            along the final axis.
        :param ~torch.Tensor insert_logits: Insertion probabilities *log(r)*.
            Should have rightmost dimension ``(M+1, 3, 2)`` and be broadcastable
            to ``(batch_size, M+1, 3, 2)``. Should be normalized along the
            final axis.
        :param ~torch.Tensor delete_logits: Deletion probabilities *log(u)*.
            Should have rightmost dimension ``(M+1, 3, 2)`` and be broadcastable
            to ``(batch_size, M+1, 3, 2)``. Should be normalized along the
            final axis.
        :param ~torch.Tensor substitute_logits: Substiution probabilities
            *log(l)*. Should have rightmost dimension ``(D, B)``, where
            B is the alphabet size of the data, and broadcastable to
            ``(batch_size, D, B)``. Must be normalized along the
            final axis.
        :return: *initial_logits*, *transition_logits*, and
            *observation_logits*. These parameters can be used to directly
            initialize the MissingDataDiscreteHMM distribution.
        :rtype: ~torch.Tensor, ~torch.Tensor, ~torch.Tensor
        """
        initial_logits = (
            torch.einsum('...ijk,ijkl->...l',
                         delete_logits, self.u_transf_0) +
            torch.einsum('...ijk,ijkl->...l',
                         insert_logits, self.r_transf_0) +
            (-1/self.epsilon)*self.null_transf_0)
        transition_logits = (
             torch.einsum('...ijk,ijklf->...lf',
                          delete_logits, self.u_transf) +
             torch.einsum('...ijk,ijklf->...lf',
                          insert_logits, self.r_transf) +
             (-1/self.epsilon)*self.null_transf)
        seq_logits = (
             torch.einsum('...ij,ik->...kj',
                          precursor_seq_logits, self.vx_transf) +
             torch.einsum('...ij,ik->...kj',
                          insert_seq_logits, self.vc_transf))

        # Option to include the substitution matrix.
        if substitute_logits is not None:
            observation_logits = torch.logsumexp(
                seq_logits.unsqueeze(-1) + substitute_logits.unsqueeze(-3),
                dim=-2)
        else:
            observation_logits = seq_logits

        return initial_logits, transition_logits, observation_logits


def mg2k(m, g):
    """Convert from (m, g) indexing to k indexing."""
    return 2*m + 1 - g
