import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from tests.integration_tests.test_conjugate_gaussian_models import GaussianChain
from pyro.infer import SVI
from tests.common import assert_equal
import pyro.optim as optim
from pyro.poutine.util import prune_subsample_sites
from torch.distributions import transform_to


# simple test model to test ADVI guide construction
def test_model():
    pyro.sample("z1", dist.Normal(0.0, 1.0))
    pyro.sample("z2", dist.Normal(torch.zeros(3), 2.0*torch.ones(3)))


# helper function
def product(shape):
    if shape == ():
        return 1
    else:
        result = 1
        for k in range(len(shape)):
            result *= shape[k]
        return result


# takes a model and returns an advi model and guide pair
def advi_multivariate(model, *args, **kwargs):
    """
    Assumes model structure and latent dimension are fixed.
    """
    # run the model so we can inspect its structure
    prototype_trace = poutine.trace(model).get_trace(*args, **kwargs)
    prototype_trace = prune_subsample_sites(prototype_trace)

    latent_dim = sum(site["value"].view(-1).size(0)
                     for site in prototype_trace.nodes.values()
                     if site["type"] == "sample" and not site["is_observed"])
    print("total latent random variable dimension: ", latent_dim)

    # sample the single multivariate normal latent used in the advi guide
    def sample_advi_latent():
        # TODO: proper initialization
        loc = pyro.param("_advi_loc", torch.tensor(0.2 * torch.randn(latent_dim), requires_grad=True))
        log_sigma = pyro.param("_advi_cholesky_log_sigma", torch.tensor(0.05 * torch.randn(latent_dim),
                                                                        requires_grad=True))
        off_diag = pyro.param("_advi_cholesky_off_diag", torch.tensor(0.05 * torch.randn(latent_dim, latent_dim),
                                                                      requires_grad=True))
        L = torch.diag(torch.exp(log_sigma)) + torch.tril(off_diag, -1)
        cov = torch.mm(L, L.t())
        return pyro.sample("_advi_latent", dist.MultivariateNormal(loc, cov))

    def guide(*args, **kwargs):
        latent = sample_advi_latent()
        pos = 0
        for name, site in prototype_trace.nodes.items():
            if site["type"] == "sample" and not site["is_observed"]:
                shape = site["fn"].shape()
                size = product(shape)
                unconstrained_value = latent[pos:pos + size].view(shape)
                pos += size
                value = transform_to(site["fn"].support)(unconstrained_value)
                pyro.sample(name, dist.Delta(value))

    def wrapped_model(*args, **kwargs):
        # wrap mvn sample with a 0.0 poutine.scale to zero out unwanted score
        with poutine.scale("advi_scope", 0.0):
            sample_advi_latent()
        # actual model sample statements shouldn't be zeroed out
        return model(*args, **kwargs)

    return wrapped_model, guide


# let's test the above logic on test_model
wrapped_model, guide = advi_multivariate(test_model)
guide_trace = poutine.trace(guide).get_trace()
wrapped_model_trace = poutine.trace(poutine.replay(wrapped_model, guide_trace)).get_trace()

guide_trace.compute_batch_log_pdf()
wrapped_model_trace.compute_batch_log_pdf()

print("\n *** INSPECTING WRAPPED MODEL TRACE *** ")
assert wrapped_model_trace.nodes['_advi_latent']['log_pdf'].item() == 0.0
assert wrapped_model_trace.nodes['z1']['log_pdf'].item() != 0.0
for name, site in wrapped_model_trace.nodes.items():
    if site["type"] == "sample":
        print("value at %s:" % name, site['value'].detach().numpy())
        print("log_pdf at %s: %.3f" % (name, site['log_pdf'].item()))

print("\n *** INSPECTING GUIDE TRACE *** ")
assert guide_trace.nodes['_advi_latent']['log_pdf'].item() != 0.0
assert guide_trace.nodes['z1']['log_pdf'].item() == 0.0
for name, site in guide_trace.nodes.items():
    if site["type"] == "sample":
        print("value at %s:" % name, site['value'].detach().numpy())
        print("log_pdf at %s: %.3f" % (name, site['log_pdf'].item()))


# conjugate model to test ADVI logic from end-to-end (this has a non-mean-field posterior)
class ADVIGaussianChain(GaussianChain):

    def compute_target(self, N):
        self.target_mu_q = torch.zeros(N)
        for i in range(1, N + 1):
            target_mu_q_i = self.target_mus[i]
            self.target_mu_q[i - 1] = target_mu_q_i.item()

    def do_test_advi(self, N, reparameterized, n_steps=10000):
        print("\nGoing to do ADVIGaussianChain test...")
        pyro.clear_param_store()
        self.setUp()
        self.setup_chain(N)
        self.compute_target(N)
        self.advi_model, self.advi_guide = advi_multivariate(self.model, reparameterized)
        print("approximate target mu:", self.target_mu_q.detach().numpy())

        adam = optim.Adam({"lr": .0006})
        svi = SVI(self.advi_model, self.advi_guide, adam, loss="ELBO", trace_graph=False)

        for k in range(n_steps):
            svi.step(reparameterized)

            if k % 500 == 0 and k > 0:
                advi_loc = pyro.param("_advi_loc")
                print("[step %d] advi mean parameter:" % k, advi_loc.detach().numpy())

        assert_equal(pyro.param("_advi_loc"), self.target_mu_q, prec=0.10,
                     msg="advi mean parameter off (don't expect to be exact)")


chain = ADVIGaussianChain()
chain.do_test_advi(7, True)
