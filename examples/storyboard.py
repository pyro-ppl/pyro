import pyro


# import torch.nn as nn
# import pyro.nn as nn
# nn.Linear == pyro.nn.Linear
from pyro.nn import Module

##############
# class NetworkThingy(Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden = nn.Linear(X.shape[1], n_hidden)
#         self.hidden2 = nn.Linear(n_hidden, n_hidden)
#         self.out   = nn.Linear(n_hidden, 1)

#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = F.tanh(self.hidden2(x))
#         x = self.out(x)
#     return x


# function 1: memoize
# function 2: call pytorch module under the hood
MNet = memo_and_pyro_module("dope_name", Net)

net = Net(my_args)  # pyro_name="my_neural_network")

for mm in net.parmeters():
    pyro.param(grp_name, mm)


def model(data):
    rr = pyro.param("l1", torch.Tensor(5, 10))

    p_net_first = pyro.module(pyro_name="p1", Net, my_args)
    p_net_first = pyro.module(pyro_name="p1", Net(my_args))
    p_net_first = pyro.module(pyro_name="p1", net)

    p_net_second = pyro.module(pyro_name="p2", Net, my_args)

    p_net_model.behave_different = flip()

    #pyro.module("dope_name", MNet())

    # else:
    # p_net = pyro.module("dope_other", SuperNet())
    # do somethign with rr
    # model_forward = torch.mm(rr,data)
    model_forward = p_net.forward(data)

    # net.paramters()

    return model_forward


def guide(data):

    p_net_guide = pyro.module("dope_name", Net())

    # net = Net(pyro_name="my_neural_network")

    rr = pyro.param("l1", torch.Tensor(5, 10))
    # p_net_first = MNet(pyro_name="p1", my_args)

    p_net_first = pyro.module(pyro_name="p1", Net(my_args))

    p_net = pyro.module("group_name", net)
    # model_forward = torch.mm(rr,data)
    model_forward = p_net.forward(data)

    return model_forward * pyro.sample(Normal(0, 1))


##############
# this is an example of a model with no global rvs, and doing minibatches
# "by hand"


def local_model(i, datum):
    c = pyro.sample("class_of _datum_" + i, Cat([1 1 1]))
    m = pyro.param("mean_of_class_" + c, [1])  # do MLE for class means
    pyro.observe("obs" + i, Gaussian(m, 1), datum)
    return c


def local_guide(i, datum):
    guide_params = pyro.params("class_posterior_", ones([3])) * datum
    c = pyro.sample("class" + i, Cat(guide_params))
    return c


# construct a function that steps the model and guide params via an elbo estimate
# note that the returned function expects same args as model and guide...
grad_step = ELBo(local_model, local_guide, model_ML=true, optimizer="adam")

# apply it to minibatches of data by hand:
for b in range(10):
    batch = select_from(data, batch_size)
    for i, d in batch:
        grad_step(i, d)

# now we have a trained model and a trained guide. we can eg classify a
# new obs:
new_class_guess = local_guide("new", new_datum)

#################
# now here's an example where we do bayesian inference about the global class means,
# we use the mapData construct to automate batching etc.


def model(data):

    class_means = pyro.sample(DiagGaussian(zeros([3])))

    def local_model(i, datum):
        c = pyro.sample("class" + i, Bernoulli([1 1 1]))
        m = class_means[c]
        pyro.observe("obs" + i, Gaussian(m, 1), datum)
        return c

    mapData(local_model, data, batch_size=10)

    return class_means


def guide(data):

    class_posterior_params = pyro.param("class_post_means", zeros([3]))
    class_means = pyro.sample(DiagGaussian(class_posterior_params))

    def local_guide(i, datum):
        guide_params = pyro.params("class_posterior_" + i, ones([3]))
        c = pyro.sample("class" + i, Bernoulli(guide_params))
        return c

    # hmm, need to synch model and guide batches.
    mapData(local_guide, data, batch_size=10)

    return class_means


# the grad_step fn now will do a full minibatch step for local and global vars.
# note we're now doing full variational Bayes, so model has no params of
# it own.
grad_step = ELBo(model, guide, optimizer="adam")

for i in range(10):
    grad_step(data)

# posterior approx is trained! it only easy here to get class means...
class_mean_sample = guide([])


#################
# here's one where we use the approximate posteriors from one dataset as
# the priors for the next (i.e. the chain rule for probs)

def local_model(i, datum):
    # use the mean-field param of previous datum as prior here,
    # if i=0 then it initializes (at 0)
    mean = pyro.sample(Gaussian(pyro.param("m" + (i - 1), [0])))
    pyro.observe("obs" + i, Gaussian(mean, 1), datum)
    return c


def local_guide(i, datum):
    mean = pyro.sample(Gaussian(pyro.param("m" + i, [0])))
    return c


grad_step = ELBo(local_model, local_guide, model_ML=false, optimizer="adam")

for i, d in data:
    for j in range(10):
        grad_step(i, d)


# LDA
# PCFG
# VAE
# AIR
# poisson process
#
# bayesian core knowledge -- omniglot
# timeseries forecasting?
#
