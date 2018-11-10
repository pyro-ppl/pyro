import torch


def density(samples, normalize=True, adjust=1, method="..."):
    """Kernel density estimate of samples"""
    pass


def quantile(samples, probs, dim=0):
    """quantile"""
    pass


def pi(samples, probs, dim=0):
    """percentile interval"""
    pass


def hpdi(samples, prob, dim=0):
    """highest posterior density interval"""
    pass


def waic(log_likehoods):
    """widely applicable information criterion"""
    pass


def gelman_rubin(samples, dim=0):
    """Compute R-hat over chains of samples. Chain dimension is specified by `dim`."""
    pass


def split_gelman_rubin(samples, dim=0):
    """Compute split R-hat over chains of samples. Chain dimension is specified by `dim`."""
    pass


def effective_number(samples, dim=0):
    """Compute effective number of samples"""
    pass
