import torch
from torch.autograd import Variable
import pyro
from pyro.distributions import DiagNormal, Bernoulli, Categorical, Delta
from pyro.infer.kl_qp import KL_QP
from networks import Encoder_c, Encoder_o, Decoder, USE_CUDA
from data import DatasetWrapper, fn_x_MNIST, fn_y_MNIST, bb
from torchvision.datasets import MNIST
from inference import BaseInference
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class SSVAEInfer(BaseInference):
    def __init__(self,nn_alpha_y,data,inference_technique):
        super(SSVAEInfer,self).__init__(data, inference_technique)
        self.nn_alpha_y = nn_alpha_y
        self.num_bactches = (len(self.data.train_batch_end_points)-1)

    def classify(self,xs):
        alpha = self.nn_alpha_y.forward(xs)
        res, ind = torch.topk(alpha, 1)  # Do MLE
        #ys = pyro.util.to_one_hot(ind,alpha) <-- type error FloatTensor vs LongTensor
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys


class SSVAEController(object):
    def __init__(self, data, inference_class, optim, supervised_batches=None,
                 tensor_sizes=None, enum_discrete=False):
        if tensor_sizes is None:
            tensor_sizes = {"hidden_sizes": (400), "latent_size": 20}
        self.tensor_sizes = tensor_sizes
        if "input_size" not in tensor_sizes:
            self.tensor_sizes["input_size"] = data.x_size
        if "output_size" not in tensor_sizes:
            self.tensor_sizes["output_size"] = data.y_size

        self.build_networks(self.tensor_sizes)

        inference_technique = inference_class(self.model, self.guide, optim, enum_discrete=enum_discrete)
        self.inference = SSVAEInfer(self.nn_alpha_y, data, inference_technique)

        if supervised_batches is None:
            supervised_batches = set([])
        self.supervised_batches = supervised_batches

        self.hack_loss_in_model = False
        self.hack_loss_separate_infer = False

    def build_networks(self,tensor_sizes):
        self.nn_alpha_y = Encoder_c(tensor_sizes)
        if USE_CUDA:
            self.nn_alpha_y.cuda()
        self.nn_mu_sigma_z = Encoder_o(tensor_sizes)
        if USE_CUDA:
            self.nn_mu_sigma_z.cuda()
        self.nn_mu_x = Decoder(tensor_sizes)
        if USE_CUDA:
            self.nn_mu_x.cuda()

    def model_classify(self, ix, xs, ys):
        if self.is_supervised(ix):
            self.observe_y_posterior(xs, ys)

    def guide_classify(self, ix, xs, ys):
        pass


    def sample_z_prior(self, batch_size):
        const_mu = Variable(torch.zeros([batch_size, self.tensor_sizes["latent_size"]]))
        const_sigma = Variable(torch.ones([batch_size, self.tensor_sizes["latent_size"]]))
        zs = pyro.sample("z", DiagNormal(const_mu, const_sigma))
        return zs

    def sample_z_posterior(self,xs,ys):
        nn_mu_sigma = pyro.module("nn_mu_sigma_z", self.nn_mu_sigma_z)
        mu, sigma = nn_mu_sigma.forward(xs, ys)
        zs = pyro.sample("z", DiagNormal(mu, sigma))
        return zs

    def sample_y_prior(self, batch_size):
        alpha = Variable(
                torch.ones([batch_size, self.tensor_sizes["output_size"]]) / (1.0 * self.tensor_sizes["output_size"]))
        ys = pyro.sample("y", Categorical(alpha))
        return ys

    def sample_y_posterior(self, xs):
        nn_alpha = pyro.module("nn_alpha_y", self.nn_alpha_y)
        alpha = nn_alpha.forward(xs)
        ys = pyro.sample("y", Categorical(alpha))
        return ys

    def observe_y_posterior(self, xs, ys):
        nn_alpha = pyro.module("nn_alpha_y", self.nn_alpha_y)
        alpha = nn_alpha.forward(xs)
        pyro.observe("y", Categorical(alpha),ys)

    def observe_x_prior(self, xs, ys, zs):
        nn_mu = pyro.module("nn_mu_x", self.nn_mu_x)
        mu = nn_mu.forward(zs, ys)
        pyro.observe("x", Bernoulli(mu), xs)

    """
        applied to a batch of data with inputs
        (batch_index ix: int, [bs x input_size], [bs x output_size])
        The model corresponds to:
            p(z) = DiagNormal(0,I)
            p(y|x) = Categorical(I/10.)
            p(x|y,z) = Bernoulli(mu(y,z))
        mu is a feed-forward neural network
        NOTE 1: first self.num_supervised_batches have observed ys (supervised case)
                the rest batches sample ys (unsupervised case)
        NOTE 2: [HACK] There is an observe on y to 
                add a loss term log q(y|x) as in the Kingma paper
    """
    def model(self, ix, xs, ys):
        batch_size = xs.size(0)

        if not self.is_supervised(ix):
            ys = self.sample_y_prior(batch_size)

        zs = self.sample_z_prior(batch_size)

        self.observe_x_prior(xs,ys,zs)

        if self.hack_loss_in_model:
            self.observe_y_posterior(xs,ys)

    """
        applied to a batch of data with inputs
        (batch_index ix: int, [bs x input_size], [bs x output_size])
        The guide corresponds to:
            q(y|x) = Categorical(alpha(x))
            q(z|x,y) = DiagNormal(mu(x,y),sigma(x,y))
        alpha, mu,sigma are produced from feed-forward neural networks
    """
    def guide(self, ix, xs, ys):

        if not self.is_supervised(ix):
            ys = self.sample_y_posterior(xs)

        # using a Delta distribution to get the appropriate loss in the supervised case
        #ys_tilde = pyro.sample("y~", Delta(ys))

        self.sample_z_posterior(xs,ys)

    def is_supervised(self,ix):
        return ix in self.supervised_batches

    """
        To set the amount of supervision percentage
    """
    def add_supervision(self,supervised_perc=10.):
        #keep only a percentage of batches as supervised
        num_batches = self.inference.num_bactches
        num_supervised_batches = int(num_batches* (supervised_perc)/100.0)

        if num_supervised_batches > num_batches/2:
            self.supervised_batches = set(range(num_supervised_batches))
        else:
            #alternating batches for supervision
            self.supervised_batches = set([])
            for ix in range(num_batches):
                if len(self.supervised_batches) == num_supervised_batches:
                    break
                if ix % 2 == 0:
                    self.supervised_batches.add(ix)


        train_data_size = self.inference.data.train_data_size
        training_batch_size = self.inference.data.training_batch_size
        num_sup_labels = min(train_data_size,len(self.supervised_batches)*training_batch_size)
        print(" Supervised fraction of data: {}/{}".format(num_sup_labels, train_data_size))


def main(args):
    data = DatasetWrapper(MNIST, y_transform=fn_y_MNIST,
                          loading_batch_size=args.batch_size, x_transform=fn_x_MNIST,
                          training_batch_size=args.batch_size, testing_batch_size=args.batch_size,
                          training_size=args.training_size)
    adam_params = {"lr": args.learning_rate}
    tensor_sizes = {
        "latent_size": args.latent_size,
        "hidden_sizes": args.hyper_hidden_sizes
    }
    SSVAE = SSVAEController(data, KL_QP, pyro.optim(torch.optim.Adam, adam_params),
                            tensor_sizes=tensor_sizes, enum_discrete=args.enum_discrete)
    SSVAE.add_supervision(args.supervised_percentage)

    """
        args.hack_loss 
            == 0 (no loss hacking)
            == 1 (adding a statement to the model) TODO: add annealing param for extra loss
            == 2 (adding a separate KL_QP infer object) TODO: add annealing param for extra loss
            == 3 (add noisy labels with narrow Gaussian instead of Delta) 
                 TODO: add annealing param for supervised vs unsupervised losses
    """
    SSVAE.hack_loss_in_model = args.hack_loss == 1
    SSVAE.hack_loss_separate_infer = args.hack_loss == 2
    assert args.hack_loss in [0,1,2], "hacking the loss is not implemented for other modes"
    if SSVAE.hack_loss_separate_infer:
        def condition_apply_aux(ix,xs,ys):
            return SSVAE.is_supervised(ix)
        SSVAE.inference.add_aux_loss(condition_apply_aux, KL_QP(SSVAE.model_classify, SSVAE.guide_classify,
                                                                pyro.optim(torch.optim.Adam, adam_params)))

    SSVAE.inference.run(num_epochs=args.epochs, acc_cutoff=args.accuracy_cutoff)

"""
Sample usage:
    python <filename> -sup 5 -bs 1500 -lr 0.0001 --epochs 200 -hiddens 200 -lsize 20 --hack-loss 1
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch-size", help="batch size for loading, training and testing",
                        type=int,default=1000)
    parser.add_argument("-sup","--supervised-percentage", help="what percentage is supervised (default = 100%)",
                        type=float, default=100.)
    parser.add_argument("-lr", "--learning-rate", help="learning rate for Adam Optimizer", type=float, default=0.001)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=10)
    parser.add_argument("--hack-loss", help="which hack to use for the loss function ", type=int, default=1)
    parser.add_argument("--accuracy-cutoff", help="cutoff for training accuracy i.e. when to stop training",
                        type=float, default=0.995)

    parser.add_argument("-hiddens","--hyper-hidden-sizes",help="hidden layer size in the parameter neural networks",
                        nargs='+',type=int,default=[400])
    parser.add_argument("-lsize","--latent-size",help="latent RV output vector size",type=int, default=20)
    parser.add_argument("--training-size", help="Set the size of the training set", type=int, default=None)
    parser.add_argument( "--enum-discrete", help="Enumerate discrete variables in KL_QP",
                        action="store_true", required=False)
    parser.add_argument("--seed",help="random seed for computation",type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if USE_CUDA:
            torch.cuda.manual_seed(args.seed)
    with torch.cuda.device(0):
        main(args)

