from pyro.infer.svi import SVI


def MAP(model, optim, *args, **kwargs):
    """
    :param model: the model (callable containing Pyro primitives)
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: pyro.optim.PyroOptim

    Maximum A Posteriori inference.

    MAP inference is useful for estimating parameters of models that have no
    latent variables, or where all latent variables have been replaced by
    parameters. In MAP models, the likelihood is specified by ``pyro.observe``
    statements that observe evidence, and the prior is specified by
    ``pyro.observe`` statements that observe parameter values.
    """

    def guide(*args, **kwargs):
        pass

    return SVI(model, guide, optim, loss="ELBO", *args, **kwargs)
