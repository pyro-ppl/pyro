
import numpy as np
import torch
import pyro
from torch.autograd import Variable
from pyro.distributions import DiagNormal, Bernoulli, Categorical, Delta
from networks import Encoder_c, Encoder_o, Decoder, USE_CUDA
if USE_CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import DatasetWrapper, fn_x_MNIST, fn_y_MNIST
from data import DatasetWrapper, fn_x_MNIST, fn_y_MNIST, bb
from torchvision.datasets import MNIST
from inference import BaseInference
from pyro.infer import SVI
from pyro.optim import Adam

import argparse
parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('-n', '--num-epochs', nargs='?', default=100, type=int)
parser.add_argument('--hack', default=0, type=int, choices=[0,1,2])
parser.add_argument('-sup', '--sup-perc', default=50,
                    type=int, choices=[1,2,5,10,20,25,50])
args = parser.parse_args()

SEED = 0
if SEED is not None:
    torch.manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

OUTPUT_SIZE= 10 # 10 labels in MNIST
LATENT_SIZE = 20
NUM_EPOCHS=args.num_epochs
tensor_sizes= {
    "output_size" : OUTPUT_SIZE,
    "latent_size" : LATENT_SIZE,
    "input_size" : 784,
    "hidden_sizes": [200]
}
BATCH_SIZE = 600
adam_params = {"lr": 0.001}

HACK_ID = args.hack
"""
0 -> basic version with no q(y|x) learning
1 -> adding an extra observe in the model -- in some sense we are changing out prior 
p(y|x) and making it same as the posterior
2 -> extra auxiliary loss 
"""
HACK_MULTIPLIER = 0.1

#networks
nn_alpha_y = Encoder_c(tensor_sizes)
nn_mu_sigma_z = Encoder_o(tensor_sizes)
nn_mu_x = Decoder(tensor_sizes)
if USE_CUDA:
    nn_alpha_y.cuda()
    nn_mu_sigma_z.cuda()
    nn_mu_x.cuda()


def model_classify(ix,xs,ys):
    #this here is the extra Term to yield an extra loss that we do gradient descend on separately, different to the Kingma paper. Also requries an extra kl-qp class later.
    nn_alpha = pyro.module("nn_alpha_y", nn_alpha_y)
    alpha = nn_alpha.forward(xs)
    pyro.observe("y_hack", Categorical(alpha), ys, log_pdf_mask=HACK_MULTIPLIER)
    pass

def guide_classify(ix,xs,ys):
    return None


"""
    The model corresponds to:
        p(z) = DiagNormal(0,I)
        p(y|x) = Categorical(I/10.)
        p(x|y,z) = Bernoulli(mu(y,z))
    mu == nn_mu_x is a neural network
"""
def model(ix,xs,ys):
    const_mu = Variable(torch.zeros([BATCH_SIZE, LATENT_SIZE]))
    const_sigma = Variable(torch.ones([BATCH_SIZE, LATENT_SIZE]))
    zs = pyro.sample("z", DiagNormal(const_mu, const_sigma))

    if is_not_supervised(ix):
        alpha = Variable(torch.ones([BATCH_SIZE, OUTPUT_SIZE]) / (1.0 * OUTPUT_SIZE))
        ys = pyro.sample("y", Categorical(alpha))

    nn_mu = pyro.module("nn_mu_x", nn_mu_x)
    mu = nn_mu.forward(zs, ys)
    pyro.observe("x", Bernoulli(mu), xs)
    if is_supervised(ix):
        alpha_prior = Variable(torch.ones([BATCH_SIZE, OUTPUT_SIZE]) / (1.0 * OUTPUT_SIZE))
        pyro.observe("y", Categorical(alpha_prior), ys)
        if HACK_ID == 1:
            nn_alpha = pyro.module("nn_alpha_y", nn_alpha_y)
            alpha = nn_alpha.forward(xs)
        #else: #HACK_ID 0 or 2
        #    alpha = Variable(torch.ones([BATCH_SIZE, OUTPUT_SIZE]) / (1.0 * OUTPUT_SIZE))
            pyro.observe("y_hack", Categorical(alpha), ys, log_pdf_mask=HACK_MULTIPLIER)



"""
    The guide corresponds to:
        q(y|x) = Categorical(alpha(x))
        q(z|x,y) = DiagNormal(mu(x,y),sigma(x,y))
    mu, sigma are given by a neural network nn_mu_sigma_z
    alpha is given by a neural network nn_alpha_y
"""
def guide(ix,xs,ys):

    if is_not_supervised(ix):
        nn_alpha = pyro.module("nn_alpha_y", nn_alpha_y)
        alpha = nn_alpha.forward(xs)
        ys = pyro.sample("y", Categorical(alpha))

    nn_mu_sigma = pyro.module("nn_mu_sigma_z", nn_mu_sigma_z)
    mu, sigma = nn_mu_sigma.forward(xs, ys)
    zs = pyro.sample("z", DiagNormal(mu, sigma))



data = DatasetWrapper(MNIST, y_transform=fn_y_MNIST,loading_batch_size=BATCH_SIZE,
                      x_transform=fn_x_MNIST,training_batch_size=BATCH_SIZE,
                      testing_batch_size=BATCH_SIZE)

num_supervised_batches = args.sup_perc*data.train_data_size/(100.0*BATCH_SIZE)
assert int(num_supervised_batches) == num_supervised_batches, "assuming simplicity of batching math"
periodic_interval  = 100/args.sup_perc
assert data.train_data_size % BATCH_SIZE == 0, "assuming simplicity of batching math"
assert int(data.train_data_size/BATCH_SIZE) % periodic_interval == 0 , "assuming simplicity of batching math"

def is_supervised(ix,xs=None,ys=None):
    return np.mod(ix,periodic_interval) == 0
def is_not_supervised(ix,xs=None,ys=None):
    return np.mod(ix,periodic_interval) != 0

class SSVAEInfer(BaseInference):
    def __init__(self, data, techniques, conditions):
        super(SSVAEInfer,self).__init__(data, techniques, conditions)
        self.num_bactches = self.data.num_train_batches

    def classify(self,xs):
        global nn_alpha_y
        alpha = nn_alpha_y.forward(xs)
        res, ind = torch.topk(alpha, 1)  # Do MLE
        #ys = pyro.util.to_one_hot(ind,alpha) <-- type error FloatTensor vs LongTensor
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys

adam = Adam(adam_params)
loss_observed = SVI(model, guide, adam, loss="ELBO")
loss_latent = SVI(model, guide, adam, loss="ELBO", enum_discrete=True)
losses = [loss_observed,loss_latent]
conditions = [is_supervised,is_not_supervised]
if HACK_ID == 2:
    loss_aux = SVI(model_classify, guide_classify, adam, loss="ELBO")
    losses.append(loss_aux)
    conditions.append(is_supervised)
inference = SSVAEInfer(data, losses, conditions)
inference.run(num_epochs=NUM_EPOCHS)