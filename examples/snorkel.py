#!/usr/bin/env python

import numpy as np
import torch
import sys

from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal, Bernoulli, Categorical

from pyro.infer.kl_qp import KL_QP


### MODEL specification
p_fraud = Variable(torch.ones(1)*1e-3)

def local_model(i, datum):
    f = pyro.sample("is_fraud_%i" % i, Bernoulli(p_fraud))        
    for d in xrange(len(datum)):
        alpha_lbl_given_fraud = pyro.param("alpha_lbl_given_fraud_%i" % d,
                                           Variable(torch.ones(3)/3., requires_grad=True))
        alpha_lbl_given_legit = pyro.param("alpha_lbl_given_legit_%i" % d,
                                           Variable(torch.ones(3)/3., requires_grad=True))
        if f:
            pyro.observe("labels_if_fraud_%i" %i,
                         Categorical(alpha_lbl_given_fraud), datum[d])
        else:
            pyro.observe("labels_if_legit_%i" %i,
                         Categorical(alpha_lbl_given_legit), datum[d])

def model(data, indices):
    pyro.map_data(data, lambda i, x: local_model(indices[i], x), batch_size=32)


def local_guide(i, datum):

    # can sample from true posterior, exemplary guide
    log_p_fraud_datum = torch.log(p_fraud)
    for d in xrange(len(datum)):
        alpha_lbl_given_fraud = pyro.param("alpha_lbl_given_fraud_%i" % d,
                                           Variable(torch.ones(3)/3., requires_grad=True))
        alpha_lbl_given_legit = pyro.param("alpha_lbl_given_legit_%i" % d,
                                           Variable(torch.ones(3)/3., requires_grad=True))

        val, indx = datum[d].max(0)
        log_p_fraud_datum += torch.log(pyro.param("alpha_lbl_given_fraud_%i" % d)[indx.data])
    log_p_legit_datum = torch.log(1.0 - p_fraud)
    for d in xrange(len(datum)):
        val, indx = datum[d].max(0)
        log_p_legit_datum += torch.log(pyro.param("alpha_lbl_given_legit_%i" % d)[indx.data])

    # TODO may need logsumexp here
    
    posterior_f = torch.exp(log_p_fraud_datum) / \
                  (torch.exp(log_p_fraud_datum) + torch.exp(log_p_legit_datum))
    f = pyro.sample("is_fraud_%i" % i, Bernoulli(posterior_f))
    return posterior_f


def guide(data, indices):
    pyro.map_data(data, lambda i, x: local_guide(indices[i], x), batch_size=32)
    


def torchify_fraud_data(data):

    # one of the best one-liners
    def one_hot_encoded(class_numbers, num_classes):
        return np.eye(num_classes, dtype=float)[class_numbers]

    # data is in form -1, 0, 1, make it 0, 1, 2
    data_ = np.copy(data)
    data_[data_ == 1] = 2
    data_[data_ == 0] = 1
    data_[data_ == -1] = 0

    data_tensor = np.zeros((data_.shape[0], data_.shape[1], 3))
    for ii in xrange(data_.shape[0]):
        data_tensor[ii] = one_hot_encoded(data_[ii].astype(np.int32), 3)

    return Variable(torch.from_numpy(data_tensor).float())


def train(data_path):

    fraud_data = np.load(data_path)
    optim_fct = pyro.optim(torch.optim.Adam, {'lr': .0001})

    data = torchify_fraud_data(fraud_data["X"][:10000])
    nr_samples = data.size(0)
    print nr_samples

    # TODO this is a batch size of 1, how do you do minibatches
    grad_step = KL_QP(model, guide, optim_fct)

    batch_size = 32
    nr_epochs = 50
    # apply it to minibatches of data by hand:
    for epoch in range(nr_epochs):
        total_loss = 0.
        total_num_samples = 0
        while total_num_samples < nr_samples:
            indices = np.random.choice(range(nr_samples), batch_size, replace=False)
            loss_sample = grad_step(data[torch.from_numpy(indices).long()], indices)
            total_loss += loss_sample
            total_num_samples += batch_size
        print("loss per epoch {}".format(total_loss / nr_samples))


if __name__ == "__main__":

    data_path = sys.argv[1]
    train(data_path)
