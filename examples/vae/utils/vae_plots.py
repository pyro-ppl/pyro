import torch


def plot_conditional_samples_ssvae(ssvae, visdom_session):
    """
    This is a method to do conditional sampling in visdom
    """
    vis = visdom_session
    ys = {}
    for i in range(10):
        ys[i] = torch.zeros(1, 10)
        ys[i][0, i] = 1
    xs = torch.zeros(1, 784)

    for i in range(10):
        images = []
        for rr in range(100):
            # get the loc from the model
            sample_loc_i = ssvae.model(xs, ys[i])
            img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)


def plot_llk(train_elbo, test_elbo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy as sp
    import seaborn as sns
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
    x = torch.zeros([1, 784])
    for i in range(10):
        images = []
        for rr in range(100):
            # get loc from the model
            sample_loc_i = vae.model(x)
            img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)


def mnist_test_tsne(vae=None, test_loader=None):
    """
    This is used to generate a t-sne embedding of the vae
    """
    name = 'VAE'
    data = test_loader.dataset.test_data.float()
    mnist_labels = test_loader.dataset.test_labels
    z_loc, z_scale = vae.encoder(data)
    plot_tsne(z_loc, mnist_labels, name)


def mnist_test_tsne_ssvae(name=None, ssvae=None, test_loader=None):
    """
    This is used to generate a t-sne embedding of the ss-vae
    """
    if name is None:
        name = 'SS-VAE'
    data = test_loader.dataset.test_data.float()
    mnist_labels = test_loader.dataset.test_labels
    z_loc, z_scale = ssvae.encoder_z([data, mnist_labels])
    plot_tsne(z_loc, mnist_labels, name)


def plot_tsne(z_loc, classes, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        ind_vec = np.zeros_like(classes)
        ind_vec[:, ic] = 1
        ind_class = classes[:, ic] == 1
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig('./vae_results/'+str(name)+'_embedding_'+str(ic)+'.png')
    fig.savefig('./vae_results/'+str(name)+'_embedding.png')
