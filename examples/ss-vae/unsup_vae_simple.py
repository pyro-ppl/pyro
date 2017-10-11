
import torch
import pyro
from torch.autograd import Variable
from pyro.distributions import DiagNormal, Bernoulli, Categorical, Delta
from networks import Encoder_c, Encoder_o, Decoder, USE_CUDA
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import DatasetWrapper, fn_x_MNIST, fn_y_MNIST
from data import DatasetWrapper, fn_x_MNIST, fn_y_MNIST, bb
from torchvision.datasets import MNIST
from inference import BaseInference
from pyro.infer.kl_qp import KL_QP


SEED = 0
if SEED is not None:
    torch.manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

OUTPUT_SIZE= 10 # 10 labels in MNIST
LATENT_SIZE = 20
NUM_EPOCHS=10
tensor_sizes= {
    "output_size" : OUTPUT_SIZE,
    "latent_size" : LATENT_SIZE,
    "input_size" : 784,
    "hidden_sizes": [400]
}
BATCH_SIZE = 600
adam_params = {"lr": 0.0001}

#networks
nn_alpha_y = Encoder_c(tensor_sizes)
nn_mu_sigma_z = Encoder_o(tensor_sizes)
nn_mu_x = Decoder(tensor_sizes)
if USE_CUDA:
    nn_alpha_y.cuda()
    nn_mu_sigma_z.cuda()
    nn_mu_x.cuda()
    
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

    alpha = Variable(torch.ones([BATCH_SIZE, OUTPUT_SIZE]) / (1.0 * OUTPUT_SIZE))
    ys = pyro.sample("y", Categorical(alpha))

    nn_mu = pyro.module("nn_mu_x", nn_mu_x)
    mu = nn_mu.forward(zs, ys)
    pyro.observe("x", Bernoulli(mu), xs)

"""
    The guide corresponds to:
        q(y|x) = Categorical(alpha(x))
        q(z|x,y) = DiagNormal(mu(x,y),sigma(x,y))
    mu, sigma are given by a neural network nn_mu_sigma_z
    alpha is given by a neural network nn_alpha_y
"""
def guide(ix,xs,ys):

    nn_alpha = pyro.module("nn_alpha_y", nn_alpha_y)
    alpha = nn_alpha.forward(xs)
    ys = pyro.sample("y", Categorical(alpha))

    nn_mu_sigma = pyro.module("nn_mu_sigma_z", nn_mu_sigma_z)
    mu, sigma = nn_mu_sigma.forward(xs, ys)
    zs = pyro.sample("z", DiagNormal(mu, sigma))



data = DatasetWrapper(MNIST, y_transform=fn_y_MNIST,loading_batch_size=BATCH_SIZE,
                      x_transform=fn_x_MNIST,training_batch_size=BATCH_SIZE,
                      testing_batch_size=BATCH_SIZE)

class SSVAEInfer(BaseInference):
    def __init__(self,data,inference_technique):
        super(SSVAEInfer,self).__init__(data, inference_technique)
        self.num_bactches = (len(self.data.train_batch_end_points)-1)

    def classify(self,xs):
        global nn_alpha_y
        alpha = nn_alpha_y.forward(xs)
        res, ind = torch.topk(alpha, 1)  # Do MLE
        #ys = pyro.util.to_one_hot(ind,alpha) <-- type error FloatTensor vs LongTensor
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys

optim=pyro.optim(torch.optim.Adam, adam_params)
inference_technique = KL_QP(model, guide, optim)
inference = SSVAEInfer(data, inference_technique)
inference.run(num_epochs=NUM_EPOCHS)