
import numpy as np
import torch
import pyro
from torch.autograd import Variable
from pyro.distributions import DiagNormal
from pyro.distributions import Categorical # as SimpleCategorical
from pyro.distributions import Bernoulli #as SimpleBernoulli
from inference import MNISTCached as MNIST
from inference import SSVAEInfer
from pyro.infer import SVI
from pyro.optim import Adam
from mlp import MLP, Exp, EpsilonScaledSigmoid, EpsilonScaledSoftmax
import torch.nn as nn

class SSVAE(nn.Module):

    def __init__(self,dargs):
        #initializing the class with all args as arguments obtained from the command line
        super(SSVAE, self).__init__()
        for arg in dargs:
            setattr(self, arg, dargs[arg])
        self.setup_networks()
        adam_params = {"lr": self.learning_rate}
        self.optimizer = Adam(adam_params)
        self.start_epoch = 0

        if self.checkpoint_load_file is not None:
            self.load_checkpoint(self.checkpoint_load_file)
        else:
            self.checkpoint = None
        self.latest_checkpoints = {}

        self.kingma_multiplier = Variable(torch.ones(self.batch_size,1)*self.kingma_multiplier)

    def classify(self,xs):
        alpha = self.nn_alpha_y.forward(xs)
        res, ind = torch.topk(alpha, 1)  # Do MLE
        #ys = pyro.util.to_one_hot(ind,alpha) <-- type error FloatTensor vs LongTensor
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys

    def save_checkpoint(self, epoch, train_acc, test_acc, post="last"):
        import os.path
        dirname = self.checkpoint_dir
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        state = {
            'epoch': epoch + 1,
            'ss_vae': self.state_dict(),
            'optimizer' : self.optimizer.get_state(),
            'rand_state' : torch.get_rng_state(),
            'rand_cuda_state' : torch.cuda.get_rng_state(),
            'train_acc' : train_acc,
            'test_acc' : test_acc
        }
        filename = os.path.join(dirname,"ss_vae_{}_{}".format(self.expt_name,post))
        torch.save(state, filename)
        self.latest_checkpoints[post] = filename

        #print("checkpoint saved: {}".format(filename))

    def load_checkpoint_randomness(self):
        if self.checkpoint is not None:
            # delayed until data loads and gets cached!
            torch.set_rng_state(self.checkpoint['rand_state'])
            torch.cuda.set_rng_state(self.checkpoint['rand_cuda_state'])

    def load_checkpoint(self,filename):
        self.checkpoint = torch.load(filename)
        self.optimizer.set_state(self.checkpoint ['optimizer'])
        self.load_state_dict(self.checkpoint ['ss_vae'])
        #self.nn_alpha_y = self.nn_alpha_y.load_state_dict(self.checkpoint ['nn_alpha_y'])
        #self.nn_mu_x = self.nn_mu_x.load_state_dict(self.checkpoint ['nn_mu_x'])
        #self.nn_mu_sigma_z = self.nn_mu_sigma_z.load_state_dict(self.checkpoint ['nn_mu_sigma_z'])

        self.start_epoch = self.checkpoint ['epoch']

    def setup_networks(self):
        self.output_size = 10  # 10 labels in MNIST
        latent_size = self.latent_layer
        self.input_size = 784  # MNIST images are 28*28 = 784 flattened
        hidden_sizes = self.hidden_layers

        # instantiating networks
        self.nn_alpha_y = MLP([self.input_size]+hidden_sizes+[self.output_size],
                              activation=nn.ReLU,output_activation=EpsilonScaledSoftmax,
                              epsilon_scale=self.epsilon_scale)
        self.nn_mu_sigma_z = MLP([[self.input_size,self.output_size]]+hidden_sizes+ [[latent_size,latent_size]],
                              activation=nn.ReLU,output_activation=[None,Exp])
        self.nn_mu_x = MLP([[latent_size,self.output_size]]+hidden_sizes+ [self.input_size],
                              activation=nn.ReLU,output_activation=EpsilonScaledSigmoid,
                              epsilon_scale=self.epsilon_scale)
        if self.cuda:
            self.nn_alpha_y.cuda()
            self.nn_mu_sigma_z.cuda()
            self.nn_mu_x.cuda()



    def model_classify(self, is_supervised, xs, ys):
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # this here is the extra Term to yield an extra loss that we do gradient descend on separately,
        #  similar to the Kingma paper. Also requries an extra kl-qp class later.
        #nn_alpha = pyro.module("nn_alpha_y", self.nn_alpha_y)
        alpha = self.nn_alpha_y.forward(xs)
        pyro.observe("y_hack", Categorical(alpha), ys, log_pdf_mask=self.kingma_multiplier)
        pass

    def guide_classify(self, is_supervised, xs, ys):
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        return None

    def check_nan_params_in_nns(self):
        if self.check_nans:
            assert np.isfinite(self.nn_alpha_y.sum_params())
            assert np.isfinite(self.nn_mu_sigma_z.sum_params())
            assert np.isfinite(self.nn_mu_x.sum_params())

    """
        The model corresponds to:
            p(z) = DiagNormal(0,I)
            p(y|x) = Categorical(I/10.)
            p(x|y,z) = Bernoulli(mu(y,z))
        mu == nn_mu_x is a neural network
    """
    def model(self, is_supervised, xs, ys):
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)
        self.check_nan_params_in_nns()

        const_mu = Variable(torch.zeros([self.batch_size, self.latent_layer]))
        const_sigma = Variable(torch.ones([self.batch_size, self.latent_layer]))
        zs = pyro.sample("z", DiagNormal(const_mu, const_sigma))

        alpha_prior = Variable(torch.ones([self.batch_size, self.output_size])
                               / (1.0 * self.output_size))

        if not is_supervised:
            ys = pyro.sample("y", Categorical(alpha_prior))

        #nn_mu = pyro.module("nn_mu_x", self.nn_mu_x)
        mu = self.nn_mu_x.forward([zs, ys])
        #assert torch.sum(mu ==0).data[0] == 0, "mu nn x produced a zero!"
        #assert torch.sum(mu == 1.0).data[0] == 0, "mu nn x produced a one!"

        pyro.observe("x", Bernoulli(mu), xs)
        if is_supervised:
            pyro.observe("y", Categorical(alpha_prior), ys)
            if self.kingma_loss and self.kingma_loss_type == 1:
                #nn_alpha = pyro.module("nn_alpha_y", self.nn_alpha_y)
                alpha = self.nn_alpha_y.forward(xs)
                pyro.observe("y_hack", Categorical(alpha), ys, log_pdf_mask=self.kingma_multiplier)



    """
        The guide corresponds to:
            q(y|x) = Categorical(alpha(x))
            q(z|x,y) = DiagNormal(mu(x,y),sigma(x,y))
        mu, sigma are given by a neural network nn_mu_sigma_z
        alpha is given by a neural network nn_alpha_y
    """
    def guide(self,is_supervised,xs,ys):
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)
        self.check_nan_params_in_nns()

        if (not is_supervised):
            #nn_alpha = pyro.module("nn_alpha_y", self.nn_alpha_y)
            alpha = self.nn_alpha_y.forward(xs)
            ys = pyro.sample("y", Categorical(alpha))

        #nn_mu_sigma = pyro.module("nn_mu_sigma_z", self.nn_mu_sigma_z)
        mu, sigma = self.nn_mu_sigma_z.forward([xs, ys])
        assert not np.isnan(torch.sum(mu).data[0]), "mu nn z produced a nan"
        assert not np.isnan(torch.sum(sigma).data[0]), "sigma nn z produced a nan"

        zs = pyro.sample("z", DiagNormal(mu, sigma))

    """
        Main function that sets up the losses and initializes the optimization/inference
    """
    def optimize(self):

        loss_observed = SVI(self.model, self.guide, self.optimizer, loss="ELBO")
        loss_latent = SVI(self.model, self.guide, self.optimizer, loss="ELBO", enum_discrete=self.enum_discrete)

        losses = [loss_observed, loss_latent]
        is_supervised_loss = [True, False]
        try:
            if self.logfile:
                logger= open(self.logfile,"w")
            else:
                logger = None

            if self.kingma_loss and self.kingma_loss_type == 2:
                loss_aux = SVI(self.model_classify, self.guide_classify, self.optimizer, loss="ELBO")
                losses.append(loss_aux)
                is_supervised_loss.append(True)

            check_nan_fn = self.check_nan_params_in_nns

            inference = SSVAEInfer(MNIST, self.batch_size, losses, is_supervised_loss, self.classify,
                                   sup_perc=self.sup_perc, checkpoint_fn=self.save_checkpoint,
                                   start_epoch=self.start_epoch, check_nans=check_nan_fn,
                                   use_cuda = self.cuda, logger=logger)

            assert (inference.train_size_sup+inference.train_size_unsup) % self.batch_size == 0, \
                "assuming simplicity of batching math"
            assert int((inference.train_size_sup+inference.train_size_unsup)/self.batch_size)\
                   % inference.periodic_interval_batches == 0 , "assuming simplicity of batching math"

            #load randomness from a checkpoint now that data has been loaded
            self.load_checkpoint_randomness()
            inference.run(num_epochs=self.num_epochs)
        finally:
            if self.logfile:
                logger.close()


def main(dargs):
    if dargs["seed"] is not None:
        torch.manual_seed(dargs["seed"])
        if  dargs["cuda"]:
            torch.cuda.manual_seed(dargs["seed"])
    if  dargs["cuda"]:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ss_vae = SSVAE(dargs)
    ss_vae.optimize()
    return ss_vae

"""

python example.py --seed 0 -cuda -ne 2 -sup 5 -ll 20 -hl 400 200 -lr 0.001 -bs 200 -enum -kingma-loss -klt 2 -km 300 -cdir ./checkpoints -en "kingma_test3" -log ./kingma_test3.log
"""
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
    parser.add_argument('-klt','--kingma-loss-type',default=0, type=int, choices = [0,1,2])
    parser.add_argument('-km', '--kingma-multiplier', default=0.1, type=float)
    parser.add_argument('-enum','--enum-discrete', action="store_true")
    #parser.add_argument('-ya', '--y-activation', default="Softmax", type=str, choices=["Softmax"])

    parser.add_argument('-sup', '--sup-perc', default=5,
                        type=float, choices=[0.2, 1, 2, 5, 10, 20, 25, 50])
    parser.add_argument('-ll', '--latent-layer', default=20, type=int)
    parser.add_argument('-hl', '--hidden-layers', nargs= '+', default = [400,200], type=int)
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float)
    parser.add_argument('-bs', '--batch-size', default=100, type=int)
    parser.add_argument('-eps', '--epsilon-scale', default=1e-7, type=float)

    parser.add_argument('-cdir', '--checkpoint-dir', default="./checkpoints", type=str)
    parser.add_argument('-cload', '--checkpoint-load-file', default=None, type=str)
    parser.add_argument('-en', '--expt-name', required=True, type=str)
    parser.add_argument('-log', '--logfile', default=None, type=str)

    parser.add_argument('-check-nans', action="store_true")

    args = parser.parse_args()

    main(vars(args))