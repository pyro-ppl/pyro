import torch
import pyro
fudge = 1e-13
from pyro.distributions import *

def kl_divergence_normal(m_q, log_sig_q, m_p, log_sig_p):
    kl = (log_sig_p-log_sig_q) + ((torch.exp(2*log_sig_q) + torch.pow((m_q-m_p), 2)) / (2*torch.exp(2*log_sig_p))) - 0.5
    return kl


class KLdiv():
    """
    class to evaluate KL divergences between distributions
    """
    def __init__(self, dist_q=None, dist_p=None, num_samples=1, *args, **kwargs):
        self.dist_q = dist_q
        self.dist_p = dist_p
        self.num_samples = num_samples

    def eval_sampled_kl(self, num_samples=10):
        """
        evaluate kldiv through sampling from the provided distributions and scoring their ratio
        """
        r_iis = 0
        for iis in range(num_samples):
            sq = self.dist_q.sample()
            r_iis += self.dist_q.log_pdf(sq) - self.dist_p.log_pdf(sq)
        kld = r_iis/num_samples
        return kld

    def eval(self, analytical=True, num_samples=10):
        """
        Evaluate the KL divergence either analytically, if an expression is available, or through sampling.
        Here we can add more analytical criteria through use of entropy and cross-entropy expressions for
        distributions at a later point.
        """
        if analytical:
            if isinstance(self.dist_q, DiagNormal):
                if isinstance(self.dist_p, DiagNormal):
                    mu_q = self.dist_q.mu
                    log_sigma_q = torch.log(self.dist_q.sigma+fudge)
                    mu_p = self.dist_p.mu
                    log_sigma_p = torch.log(self.dist_p.sigma+fudge)
                    kld = kl_divergence_normal(mu_q, log_sigma_q, mu_p, log_sigma_p)
                else:
                    kld = self.eval_sampled_kl(num_samples)

            else:
                kld = self.eval_sampled_kl(num_samples)

        else:
            kld = self.eval_sampled_kl(num_samples)
        return kld
