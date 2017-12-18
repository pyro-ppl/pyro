import torch
from torch.autograd import Variable
# import numpy as np


def plot_conditional_samples_ssvae(ssvae, visdom_session):
    """
    This is a method to do conditional sampling in visdom
    """
    vis = visdom_session
    ys = {}
    for i in range(10):
        ys[i] = Variable(torch.zeros(1, 10))
        ys[i][0, i] = 1

    for i in range(10):
        images = []
        for rr in range(100):
            sample_i, sample_mu_i = ssvae.model_sample(ys[i])
            img = sample_mu_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)


def plot_llk(train_elbo, test_elbo):
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    import seaborn as sns
    import pandas as pd
    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")
    data = np.concatenate([np.arange(len(test_elbo))[:, sp.newaxis], -test_elbo[:, sp.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=['Training Epoch', 'Test ELBO'])
    g = sns.FacetGrid(df, size=10, aspect=1.5)
    g.map(plt.scatter, "Training Epoch", "Test ELBO")
    g.map(plt.plot, "Training Epoch", "Test ELBO")
    plt.savefig('./vae_results/test_elbo_vae.png')
    plt.close('all')


def plot_vae_samples(vae, visdom_session):
    vis = visdom_session
    for i in range(10):
        images = []
        for rr in range(100):
            sample_i, sample_mu_i = vae.model_sample()
            img = sample_mu_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)


def mnist_test_tsne(vae=None, test_loader=None):
    """
    This is used to generate a t-sne embedding of the vae
    """
    name = 'VAE'
    data = Variable(test_loader.dataset.test_data.float())
    mnist_labels = Variable(test_loader.dataset.test_labels)
    z_mu, z_sigma = vae.encoder(data)
    plot_tsne(z_mu, mnist_labels, name)


def mnist_test_tsne_ssvae(name=None, ssvae=None, test_loader=None):
    """
    This is used to generate a t-sne embedding of the ss-vae
    """
    if name is None:
        name = 'SS-VAE'
    data = Variable(test_loader.dataset.test_data.float())
    mnist_labels = Variable(test_loader.dataset.test_labels)
    z_mu, z_sigma = ssvae.encoder_z([data, mnist_labels])
    plot_tsne(z_mu, mnist_labels, name)


def plot_tsne(z_mu, classes, name):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_mu.data.cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.data.cpu().numpy()
    fig666 = plt.figure()
    for ic in range(10):
        ind_vec = np.zeros_like(classes)
        ind_vec[:, ic] = 1
        ind_class = classes[:, ic] == 1
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.title("Latent Variable T-SNE per Class")
        fig666.savefig('./vae_results/'+str(name)+'_embedding_'+str(ic)+'.png')
    fig666.savefig('./vae_results/'+str(name)+'_embedding.png')
