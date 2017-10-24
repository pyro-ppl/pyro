
import numpy as np
import torch
import pyro
from torch.autograd import Variable
from pyro.distributions import DiagNormal, Bernoulli, Categorical, Delta
from networks import Encoder_c, Encoder_o, Decoder, USE_CUDA
if USE_CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
#from torchvision.datasets import MNIST
from inference import MNIST_cached as MNIST
from inference import BaseInference
from pyro.infer import SVI
from pyro.optim import Adam

class SSVAEInfer(BaseInference):
    def __init__(self, dataset, batch_size, techniques, is_supervised_loss, nn_alpha_y, **kwargs):
        super(SSVAEInfer,self).__init__(dataset, batch_size, techniques, is_supervised_loss, **kwargs)
        self.nn_alpha_y = nn_alpha_y

    def classify(self,xs):
        alpha = self.nn_alpha_y.forward(xs)
        res, ind = torch.topk(alpha, 1)  # Do MLE
        #ys = pyro.util.to_one_hot(ind,alpha) <-- type error FloatTensor vs LongTensor
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys

class SSVAE(object):

    """
    arguments in args:
        seed: random seed for fixing computation: default=None, type=int
        cuda: type=bool
    parser.add_argument('-ne', '--num-epochs', default=100, type=int)

    not kingma_loss -> basic version with no [log q(y|x)] extra loss
    kingma_loss 1 -> adding an extra observe in the model
    kingma_loss 2 -> explicit extra auxiliary loss

    parser.add_argument('-kingma-loss', action="store_true")
    parser.add_argument('-klt','--kingma-loss-type',default=2, type=int)
    parser.add_argument('-km', '--kingma-multiplier', default=0.1, type=float)

    parser.add_argument('-sup', '--sup-perc', default=5,
                        type=int, choices=[1, 2, 5, 10, 20, 25, 50])
    parser.add_argument('-ll', '--latent-layer', default=20, type=int)
    parser.add_argument('-hl', '--hidden-layers', nargs= '+', default = [400], type=int)
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float)
    parser.add_argument('-bs', '--batch-size', default=100, type=int)
    """
    def __init__(self,args):
        #initializing the class with all args as arguments obtained from the command line
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
        self.setup_networks()

    def setup_networks(self):
        global USE_CUDA
        self.tensor_sizes = {
            "output_size": 10,  # 10 labels in MNIST
            "latent_size": self.latent_layer,
            "input_size": 784,  # MNIST images are 28*28 = 784 flattened
            "hidden_sizes": self.hidden_layers
        }

        #modifying a global flag USE_CUDA in networks.py
        USE_CUDA = self.cuda

        # instantiating networks
        self.nn_alpha_y = Encoder_c(self.tensor_sizes)
        self.nn_mu_sigma_z = Encoder_o(self.tensor_sizes)
        self.nn_mu_x = Decoder(self.tensor_sizes)
        if USE_CUDA:
            self.nn_alpha_y.cuda()
            self.nn_mu_sigma_z.cuda()
            self.nn_mu_x.cuda()


    def model_classify(self, is_supervised, xs, ys):
        # this here is the extra Term to yield an extra loss that we do gradient descend on separately,
        #  similar to the Kingma paper. Also requries an extra kl-qp class later.
        nn_alpha = pyro.module("nn_alpha_y", self.nn_alpha_y)
        alpha = nn_alpha.forward(xs)
        pyro.observe("y_hack", Categorical(alpha), ys, log_pdf_mask=self.kingma_multiplier)
        pass

    def guide_classify(self, is_supervised, xs, ys):
        return None


    """
        The model corresponds to:
            p(z) = DiagNormal(0,I)
            p(y|x) = Categorical(I/10.)
            p(x|y,z) = Bernoulli(mu(y,z))
        mu == nn_mu_x is a neural network
    """
    def model(self, is_supervised, xs, ys):
        const_mu = Variable(torch.zeros([self.batch_size, self.latent_layer]))
        const_sigma = Variable(torch.ones([self.batch_size, self.latent_layer]))
        zs = pyro.sample("z", DiagNormal(const_mu, const_sigma))

        alpha_prior = Variable(torch.ones([self.batch_size, self.tensor_sizes["output_size"]])
                               / (1.0 * self.tensor_sizes["output_size"]))
        if not is_supervised:
            ys = pyro.sample("y", Categorical(alpha_prior))

        nn_mu = pyro.module("nn_mu_x", self.nn_mu_x)
        mu = nn_mu.forward(zs, ys)
        pyro.observe("x", Bernoulli(mu), xs)
        if is_supervised:
            pyro.observe("y", Categorical(alpha_prior), ys)
            if self.kingma_loss_type == 1:
                nn_alpha = pyro.module("nn_alpha_y", self.nn_alpha_y)
                alpha = nn_alpha.forward(xs)
                pyro.observe("y_hack", Categorical(alpha), ys, log_pdf_mask=self.kingma_multiplier)



    """
        The guide corresponds to:
            q(y|x) = Categorical(alpha(x))
            q(z|x,y) = DiagNormal(mu(x,y),sigma(x,y))
        mu, sigma are given by a neural network nn_mu_sigma_z
        alpha is given by a neural network nn_alpha_y
    """
    def guide(self,is_supervised,xs,ys):

        if (not is_supervised):
            nn_alpha = pyro.module("nn_alpha_y", self.nn_alpha_y)
            alpha = nn_alpha.forward(xs)
            ys = pyro.sample("y", Categorical(alpha))

        nn_mu_sigma = pyro.module("nn_mu_sigma_z", self.nn_mu_sigma_z)
        mu, sigma = nn_mu_sigma.forward(xs, ys)
        zs = pyro.sample("z", DiagNormal(mu, sigma))

    """
        Main function that sets up the losses and initializes the optimization/inference
    """
    def optimize(self):
        adam_params = {"lr": self.learning_rate}
        adam = Adam(adam_params)

        loss_observed = SVI(self.model, self.guide, adam, loss="ELBO")
        loss_latent = SVI(self.model, self.guide, adam, loss="ELBO", enum_discrete=True)

        losses = [loss_observed, loss_latent]
        is_supervised_loss = [True, False]

        if self.kingma_loss_type == 2:
            loss_aux = SVI(self.model_classify, self.guide_classify, adam, loss="ELBO")
            losses.append(loss_aux)
            is_supervised_loss.append(True)

        inference = SSVAEInfer(MNIST, self.batch_size, losses, is_supervised_loss,
                               sup_perc=self.sup_perc, nn_alpha_y=self.nn_alpha_y)

        assert (inference.train_size_sup+inference.train_size_unsup) % self.batch_size == 0, \
            "assuming simplicity of batching math"
        assert int((inference.train_size_sup+inference.train_size_unsup)/self.batch_size)\
               % inference.periodic_interval_batches == 0 , "assuming simplicity of batching math"

        inference.run(num_epochs=self.num_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('-cuda',action='store_true') #default is False
    parser.add_argument('-ne', '--num-epochs', default=100, type=int)
    """
    not kingma_loss -> basic version with no [log q(y|x)] extra loss
    kingma_loss 1 -> adding an extra observe in the model 
    kingma_loss 2 -> explicit extra auxiliary loss 
    """
    parser.add_argument('-kingma-loss', action="store_true")
    parser.add_argument('-klt','--kingma-loss-type',default=2, type=int)
    parser.add_argument('-km', '--kingma-multiplier', default=0.1, type=float)

    parser.add_argument('-sup', '--sup-perc', default=5,
                        type=int, choices=[1, 2, 5, 10, 20, 25, 50])
    parser.add_argument('-ll', '--latent-layer', default=20, type=int)
    parser.add_argument('-hl', '--hidden-layers', nargs= '+', default = [400], type=int)
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float)
    parser.add_argument('-bs', '--batch-size', default=100, type=int)

    args = parser.parse_args()

    assert len(args.hidden_layers) == 1, "only 1 hidden layer supported"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    ss_vae = SSVAE(args)
    ss_vae.optimize()