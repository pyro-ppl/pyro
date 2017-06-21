import pyro

dim = 10
# x_obs = load_data()


def model(args=[obs=x_obs]):
    latent_sample = pyro.sample(name="latent", pyro.distributions.beta(torch.ones([dim]), torch.ones([dim])))
    x_dist = pyro.distributions.bernoulli(latent_sample)
    x = pyro.observe(x_dist, x_obs, name="obs")
    return latent_sample
    # return latent_sample[0]


# marginal inference
infer = pyro.SMC(p=model)
# distribution object; here an Empirical
marginal_posterior = infer(x_obs)

# marginal log-likelihood
Logp_post = infer.log_pdf(x_obs)

# posterior inference


def q_model():
    q_latent_sample = pyro.sample(name="latent2", pyro.distributions.beta(pyro.param("p1", torch.ones([dim])), pyro.param("p2", torch.ones([dim])))
return q_latent_sample

infer_meanfield=pyro.VI_Klqp(p=model)  # defaults to meanfield
infer2=pyro.VI_KLqp(p=model, q=q_model, optimizer=some_optimizer)
# assume q_model contains all necessary named random variables
infer3=pyro.VI_Klqp(p=model, q=q_model, latents={"latent": "latent2"})

# return nothing?
infer3(x_obs)

# return updated q?
Q_learned=infer3(x_obs)
