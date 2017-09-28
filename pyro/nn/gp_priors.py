
import numpy as np
import torch
from torch.autograd import Variable


### Mean functions
def muConstant(x, hypers):
    
    n, d = x.size()
    return Variable(torch.ones(n).float()) * hypers["constant_mean"]

def muLogistic(X, hypers):

    # Note: X is time
    
    amplitude = torch.exp(hypers["log_alpha"])
    growth_rate = torch.exp(hypers["log_beta"])
    location = hypers["location"]

    tt = X[:, 0]
    mu_ = amplitude / (1 + torch.exp(-growth_rate*(tt - location)))

    return mu_ 

    

### Covariance functions
def covIID(x, xp, hypers):

    iid_var = torch.exp(hypers["log_iid_var"])
    n, d = x.size()
    m, d = xp.size()
    if n == m:
        jitter_ = 1e-5
        return (iid_var + jitter_).expand(n, n) * Variable(torch.eye(n))
    else:
        # no cross covariance
        return Variable(torch.zeros(n, m))
        

def covARD(x, xp, hypers):

    kk = [hn for hn in hypers.keys() if "log_signal_var" in hn] # accomodate suffixes for e.g. covSum
    assert len(kk) == 1, "signal variance not specified"
    signal_var = torch.exp(hypers[kk[0]])

    kk = [hn for hn in hypers.keys() if "log_lengthscales" in hn] # accomodate suffixes for e.g. covSum
    assert len(kk) == 1, "lengthscales not specified"
    lengthscales = torch.exp(hypers[kk[0]]) * 2.

    n, d = x.size()
    m, d = xp.size()

    assert len(lengthscales) == d, "invalid lengthscale hyper, needs to be of size (%i,)" % d
    
    xnorm = x.div(lengthscales)
    xp_norm = xp.div(lengthscales)


    cross_ = 2 * torch.mm(xnorm, xp_norm.transpose(0,1))

    x_sq = torch.bmm(xnorm.view(n, 1, d), xnorm.view(n, d, 1))
    xp_sq = torch.bmm(xp_norm.view(m, 1, d), xp_norm.view(m, d, 1))

    x_sq = x_sq.view(n, 1).expand(n, m)
    xp_sq = xp_sq.view(1, m).expand(n, m)

    res = x_sq + xp_sq - cross_
    res = signal_var.expand_as(res) * torch.exp(-0.5*res)

    return res


def covSum(x, xp, hypers):
    K = None
    n, d = x.size()
    assert len(hypers) == d, "invalid hypers for covSum, need a config per dimension" 
    for dd, kernel_dict in enumerate(hypers):
        if K is None:
            K = kernel_dict["K"](x[:, [dd]], xp[:, [dd]], kernel_dict["hypers"])
        else:
            K += kernel_dict["K"](x[:, [dd]], xp[:, [dd]], kernel_dict["hypers"])
    return K


### Noise functions
def covNoise(x, hypers):

    n, d = x.size()
    jitter_ = 1e-5
    K_noise = (torch.exp(hypers["log_noise"]) + jitter_).expand(n, n) * Variable(torch.eye(n))
    return K_noise



# Helper functions
def build_meanfunc(mean_func_handle, optimize_hypers=True, **hypers_init):
    # wrap each hyper as an optionally optimizable hyper
    hypers_ = {}
    for hname, hyper in hypers_init.items():
        hypers_[hname] = Variable((hyper*torch.ones(1)).float(), 
                                  requires_grad=optimize_hypers)
        
    mean_func = {"mu": mean_func_handle, "hypers": hypers_}
    
    return mean_func


def build_covfunc(cov_func_handle, optimize_hypers=True, **hypers_init):
    # wrap each hyper as an optionally optimizable hyper
    hypers_ = {}
    for hname, hyper in hypers_init.items():
        if type(hyper) == np.ndarray:
            assert len(hyper.shape) == 1, "only vector hypers are allowed"
            hypers_[hname] = Variable((torch.from_numpy(hyper)).float(), 
                                      requires_grad=optimize_hypers)
        else:
            assert not hasattr(hyper, '__iter__'), "invalid hyper"
            hypers_[hname] = Variable((hyper*torch.ones(1)).float(), 
                                      requires_grad=optimize_hypers)

        
    cov_func = {"K": cov_func_handle, "hypers": hypers_}
    
    return cov_func



### NO LONGER REQUIRED, kept around just in case

# class Cholesky(torch.autograd.Function):

#     def forward(ctx, a):
#         l = torch.potrf(a, False)
#         ctx.save_for_backward(l)
#         return l

#     def backward(ctx, grad_output):
#         l, = ctx.saved_tensors
#         # TODO: use gaussian elimintation instead of inverse once support from pytorch
#         linv =  l.inverse()
#         inner = torch.tril(torch.mm(l.t(),grad_output))*torch.tril(1.0-l.new(l.size(1)).fill_(0.5).diag())
#         s = torch.mm(linv.t(), torch.mm(inner, linv))
#         #s = (s+s.t())/2.0
#         return s




    
