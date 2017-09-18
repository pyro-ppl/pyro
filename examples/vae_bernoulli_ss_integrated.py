import argparse
import torch
import pyro
from torch.autograd import Variable
from pyro.infer.kl_qp import KL_QP
from pyro.distributions import DiagNormal, Normal, Bernoulli, Multinomial, Categorical
from torch import nn

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import visdom
import pdb as pdb

# load mnist dataset
root = './data'
download = True
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(
    root=root,
    train=True,
    transform=trans,
    download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)


train_loader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])),
    batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=128, shuffle=True)


batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False, **kwargs)

# network


def workflow(data, classes):
    z_mu,z_sigma = pt_encode_o.forward(data,classes)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    from  sklearn import manifold
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2,random_state=0)
    z_states = z_mu.data.cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)

    classes = classes.data.cpu().numpy()
    fig666= plt.figure()

    colors = []
    for ic in range(10):
        ind_vec = np.zeros_like(classes)
        ind_vec[:,ic]=1
        ind_class = classes[:,ic]==1
        #bb()
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class,0],z_embed[ind_class,1],s=10,color=color)
        plt.title("Latent Variable Embeddings colour coded by class for VAE-SS")
        fig666.savefig('./results/vaeSS_embedding_'+str(ic)+'.png')
        #bb()


    fig666.savefig('./results/vaeSS_embedding.png')
    pass

class Encoder_c(nn.Module):
    def __init__(self):
        super(Encoder_c, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        # self.exp = nn.Exp()

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = self.relu(self.fc1(x))
        return self.softmax(self.fc21(h1))

class Encoder_o(nn.Module):
    def __init__(self):
        super(Encoder_o, self).__init__()
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.relu = nn.ReLU()
        # self.exp = nn.Exp()

    def forward(self, x, cll):
        x = x.view(-1, 784)
        # bb()
        input_vec = torch.cat((x, cll), 1)
        h1 = self.relu(self.fc1(input_vec))
        return self.fc21(h1), torch.exp(self.fc22(h1))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20 + 10, 400)
        self.fc4 = nn.Linear(400, 1 * 784)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z, cll):
        input_vec = torch.cat((z, cll), 1)
        h3 = self.relu(self.fc3(input_vec))
        rv = self.sigmoid(self.fc4(h3))
        # reshape to capture mu, sigma params for every pixel
        rvs = rv.view(z.size(0), -1, 1)
        # send back two params
        return rv  # rvs[:,:, 0]


pt_encode_c = Encoder_c()
pt_encode_o = Encoder_o()
pt_decode = Decoder()


def model_latent_backup(data):
    # wrap params for use in model -- required
    decoder = pyro.module("decoder", pt_decode)
    # sample from prior
    z_mu, z_sigma = Variable(torch.zeros([data.size(0), 20])), Variable(
        torch.ones([data.size(0), 20]))

    # sample
    z = pyro.sample("latent_z", DiagNormal(z_mu, z_sigma))

    alpha = Variable(torch.ones([data.size(0), 10])) / 10.
    # c = pyro.sample('latent_class', Multinomial(alpha,1))#Categorical(alpha))
    cll = pyro.sample('latent_class', Categorical(alpha))  # Categorical(alpha))

    # bb()
    # decode into size of imgx2 for mu/sigma
    img_mu = decoder.forward(z, cll)
    # score against actual images
    pyro.observe("obs", Bernoulli(img_mu), data.view(-1, 784))


def model_latent(data):
    """
    analytically integrate over all classes
    """
    nr_classes = 10
    alpha = Variable(torch.ones([data.size(0), 10])) / 10.
    #cll = pyro.sample('latent_class', Categorical(alpha))
    for ic in range(nr_classes):
        cll = Variable(torch.zeros([data.size(0), 10]))
        cll[:,ic] = 1
        pyro.observe("latent_class", Categorical(alpha), cll)
        model_observed(data, cll)
    pass


def model_observed(data, cll):
    # wrap params for use in model -- required
    decoder = pyro.module("decoder", pt_decode)

    # sample from prior
    z_mu, z_sigma = Variable(torch.zeros([data.size(0), 20])), Variable(
        torch.ones([data.size(0), 20]))

    # sample
    z = pyro.sample("latent_z", DiagNormal(z_mu, z_sigma))

    encoder_c = pyro.module("encoder_c", pt_encode_c)
    alpha = encoder_c.forward(data)
    pyro.observe("latent_class", Categorical(alpha), cll)

    # decode into size of imgx2 for mu/sigma
    img_mu = decoder.forward(z, cll)

    # score against actual images
    pyro.observe("obs", Bernoulli(img_mu), data.view(-1, 784))


def guide_observed(data, cll):
    encoder = pyro.module("encoder_o", pt_encode_o)
    z_mu, z_sigma = encoder.forward(data, cll)
    z = pyro.sample("latent_z", DiagNormal(z_mu, z_sigma))


def guide_observed2(data, cll):
    encoder_c = pyro.module("encoder_c", pt_encode_c)
    alpha = encoder_c.forward(data)
    pyro.observe("latent_class", Categorical(alpha), cll)

    encoder = pyro.module("encoder_o", pt_encode_o)
    z_mu, z_sigma = encoder.forward(data, cll)
    z = pyro.sample("latent_z", DiagNormal(z_mu, z_sigma))


def guide_latent(data):
    encoder_c = pyro.module("encoder_c", pt_encode_c)
    alpha = encoder_c.forward(data)
    cll = pyro.sample("latent_class", Categorical(alpha))
    guide_observed(data, cll)


def guide_latent2(data):
    encoder_c = pyro.module("encoder_c", pt_encode_c)
    alpha = encoder_c.forward(data)
    cll = pyro.sample("latent_class", Categorical(alpha))

    encoder = pyro.module("encoder_o", pt_encode_o)
    z_mu, z_sigma = encoder.forward(data, cll)
    z = pyro.sample("latent_z", DiagNormal(z_mu, z_sigma))


def model_sample(cll=None):
    # wrap params for use in model -- required
    # decoder = pyro.module("decoder", pt_decode)

    # sample from prior
    z_mu, z_sigma = Variable(torch.zeros(
        [1, 20])), Variable(torch.ones([1, 20]))

    # sample
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    alpha = Variable(torch.ones([1, 10]) / 10.)

    if cll.data.cpu().numpy() is None:
        bb()
        cll = pyro.sample('class', Categorical(alpha))
        print('sampling class')

    # decode into size of imgx1 for mu
    img_mu = pt_decode.forward(z, cll)
    # bb()
    # img=Bernoulli(img_mu).sample()
    # score against actual images
    img = pyro.sample("sample", Bernoulli(img_mu))
    # return img
    return img, img_mu


def model_sample_given_image(data=None):
    pass


def classify(data):
    alpha_mu = pt_encode_c.forward(data)
    cll = pyro.sample("sample_cll", Categorical(alpha_mu))
    return cll, alpha_mu


def per_param_args(name, param):
    if name == "decoder":
        return {"lr": .00001}
    else:
        return {"lr": .00001}


# or alternatively
adam_params = {"lr": .00001}
# optim.SGD(lr=.0001)

inference_latent_class = KL_QP(model_latent, guide_latent, pyro.optim(optim.Adam, adam_params))

inference_observed_class = KL_QP(
    model_observed, guide_observed, pyro.optim(
        optim.Adam, adam_params))

inference_observed_class_scored = KL_QP(
    model_observed, guide_observed2, pyro.optim(
        optim.Adam, adam_params))

mnist_data = Variable(train_loader.dataset.train_data.float() / 255.)
mnist_labels = Variable(train_loader.dataset.train_labels)
mnist_size = mnist_data.size(0)
batch_size = 128  # 64

mnist_data_test = Variable(test_loader.dataset.test_data.float() / 255.)
mnist_labels_test_raw = Variable(test_loader.dataset.test_labels)
mnist_labels_test = torch.zeros(mnist_labels_test_raw.size(0), 10)
mnist_labels_test.scatter_(1, mnist_labels_test_raw.data.view(-1, 1), 1)
mnist_labels_test = Variable(mnist_labels_test)


# TODO: batches not necessarily
all_batches = np.arange(0, mnist_size, batch_size)

if all_batches[-1] != mnist_size:
    all_batches = list(all_batches) + [mnist_size]

vis = visdom.Visdom(env='vae_ss_integrated_400')

# rand_ix = 0
# sam_cnt = 50
# for i in range(sam_cnt):
#   vis.image(mnist_data[rand_ix].data.numpy())
#   vis.image(Bernoulli(mnist_data[rand_ix])().data.numpy())
# bb()


cll_clamp0 = Variable(torch.zeros(1, 10))
cll_clamp3 = Variable(torch.zeros(1, 10))
cll_clamp9 = Variable(torch.zeros(1, 10))

cll_clamp0[0, 0] = 1
cll_clamp3[0, 3] = 1
cll_clamp9[0, 9] = 1


loss_training = []


def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', nargs='?', default=1000, type=int)
    args = parser.parse_args()
    for i in range(args.num_epochs):

        epoch_loss = 0.
        for ix, batch_start in enumerate(all_batches[:-1]):
            batch_end = all_batches[ix + 1]

            #print('Batch '+str(ix))
            # get batch
            batch_data = mnist_data[batch_start:batch_end]
            bs_size = batch_data.size(0)
            batch_class_raw = mnist_labels[batch_start:batch_end]
            batch_class = torch.zeros(bs_size, 10)  # maybe it needs a FloatTensor
            batch_class.scatter_(1, batch_class_raw.data.view(-1, 1), 1)
            batch_class = Variable(batch_class)

            if np.mod(ix, 1) == 0:
                epoch_loss += inference_observed_class.step(batch_data, batch_class)
                #epoch_loss += inference_observed_class_scored.step(batch_data,batch_class)
            else:
                epoch_loss += inference_latent_class.step(batch_data)
            #
            # bb()
        loss_training.append(epoch_loss / float(mnist_size))

        if np.mod(i,5)==0:
            if i>0:
                workflow(mnist_data_test,mnist_labels_test)

        if 0:#np.mod(i,8)==0:
            for rr in range(5):
                sample0, sample_mu0 = model_sample(cll=cll_clamp0)
                sample3, sample_mu3 = model_sample(cll=cll_clamp3)
                sample9, sample_mu9 = model_sample(cll=cll_clamp9)

                vis.image(sample_mu0[0].view(28, 28).data.numpy())
                vis.image(sample_mu3[0].view(28, 28).data.numpy())
                vis.image(sample_mu9[0].view(28, 28).data.numpy())
        print("epoch "+str(i)+" avg loss {}".format(epoch_loss / float(mnist_size)))

        pass

if __name__ == '__main__':
    main()
    
