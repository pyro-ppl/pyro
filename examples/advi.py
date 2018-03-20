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
from pyro.infer import ADVIMultivariateNormal
import sys


# simple test model to test ADVI guide construction
def test_model():
    pyro.sample("z1", dist.Normal(0.0, 1.0))
    pyro.sample("z2", dist.Normal(torch.zeros(3), 2.0 * torch.ones(3)))


# let's test the above logic on test_model
advi = ADVIMultivariateNormal(test_model)
guide_trace = poutine.trace(advi.guide).get_trace()
model_trace = poutine.trace(poutine.replay(advi.model, guide_trace)).get_trace()

guide_trace.compute_batch_log_pdf()
model_trace.compute_batch_log_pdf()

print("\n *** INSPECTING WRAPPED MODEL TRACE *** ")
assert model_trace.nodes['_advi_latent']['log_pdf'].item() == 0.0
assert model_trace.nodes['z1']['log_pdf'].item() != 0.0
for name, site in model_trace.nodes.items():
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

sys.exit()

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
