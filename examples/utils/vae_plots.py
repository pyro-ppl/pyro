import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.datasets as dset
from torch.autograd import Variable

def plot_conditional_samples_ssvae(ssvae=None, visdom_session=None):
    vs = visdom_session
    """
    This is a method to do conditional sampling in visdom
    """
    cll_clamp0 = Variable(torch.zeros(1, 10))
    cll_clamp3 = Variable(torch.zeros(1, 10))
    cll_clamp9 = Variable(torch.zeros(1, 10))

    cll_clamp0[0, 0] = 1
    cll_clamp3[0, 3] = 1
    cll_clamp9[0, 9] = 1
    if 1:
        for rr in range(5):
            sample0, sample_mu0 = ssvae.model_sample(cll=cll_clamp0)
            sample3, sample_mu3 = ssvae.model_sample(cll=cll_clamp3)
            sample9, sample_mu9 = ssvae.model_sample(cll=cll_clamp9)
            vis.line(np.array(loss_training), opts=dict({'title': 'my title'}))
            vis.image(batch_data[0].view(28, 28).data.numpy())
            #vis.image(sample[0].view(28, 28).data.numpy())
            vis.image(sample_mu0[0].view(28, 28).data.numpy())
            vis.image(sample_mu3[0].view(28, 28).data.numpy())
            vis.image(sample_mu9[0].view(28, 28).data.numpy())
    pass


def plot_llk(train_elbo, test_elbo):
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    import seaborn as sns
    import pandas as pd
    fig01 = plt.figure(figsize=(30,10))
    sns.set_style("whitegrid")
    import pdb as pdb
    data = np.concatenate([np.arange(len(test_elbo))[:,sp.newaxis], test_elbo[:,sp.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=['Training Epoch', 'Test NLL'])
    g = sns.FacetGrid(df, size=10, aspect=1.5)
    g.map(plt.scatter, "Training Epoch", "Test NLL")
    g.map(plt.plot, "Training Epoch", "Test NLL")
    plt.savefig('./vae_results/test_nll_vae.png')
    plt.close('all')
    pass

def mnist_test_tsne(vae=None, train_loader=None):
    """
    This is used to generate a t-sne embedding of the vae
    """
    name = 'VAE'
    data = Variable(train_loader.dataset.train_data.float())
    mnist_labels_raw = Variable(train_loader.dataset.train_labels)
    mnist_labels = torch.zeros(mnist_labels_raw.size(0), 10)
    mnist_labels.scatter_(1, mnist_labels_raw.data.view(-1, 1), 1)
    mnist_labels = Variable(mnist_labels)
    z_mu, z_sigma = vae.encoder(data)
    plot_tsne(z_mu, mnist_labels, name)
    pass

def mnist_test_tsne_ssvae(ssvae=None, train_loader=None):
    """
    This is used to generate a t-sne embedding of the ss-vae
    """
    name = 'SS-VAE'
    data = Variable(train_loader.dataset.train_data.float())
    mnist_labels_raw = Variable(train_loader.dataset.train_labels)
    mnist_labels = torch.zeros(mnist_labels_raw.size(0), 10)
    mnist_labels.scatter_(1, mnist_labels_raw.data.view(-1, 1), 1)
    mnist_labels = Variable(mnist_labels)
    z_mu, z_sigma = ssvae.encoder(data, mnist_labels)
    plot_tsne(z_mu, mnist_labels, name)
    pass


def plot_tsne(z_mu, classes, name):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
        plt.title("Latent Variable Embeddings colour coded by class for "+str(name))
        fig666.savefig('./vae_results/'+str(name)+'_embedding_'+str(ic)+'.png')
        #bb()
    fig666.savefig('./vae_results/'+str(name)+'_embedding.png')
    pass