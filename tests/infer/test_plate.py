from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# This file exercises our complete range of syntaxes for pyro.plate.


def assert_ok(model, guide):
    """
    Assert that inference works without warnings or errors.
    """
    pyro.clear_param_store()
    elbo = Trace_ELBO()
    inference = SVI(model, guide, Adam({"lr": 1e-6}), elbo)
    inference.step()


data = torch.tensor([0., 1., 8., 9., 10.])


def test_no_plate():

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()
        for i in range(len(data)):
            loc = pyro.param("prior_loc_{}".format(i), torch.tensor(0.))
            z = pyro.sample("z_{}".format(i), dist.Normal(loc, 1.0))
            x = pyro.sample("x_{}".format(i), dist.Normal(z, scale), obs=data[i])
            assert loc.shape == ()
            assert z.shape == ()
            assert x.shape == ()

    def guide():
        for i in range(len(data)):
            loc = pyro.param("post_loc_{}".format(i), torch.tensor(0.))
            scale = pyro.param("post_scale_{}".format(i), torch.tensor(1.))
            z = pyro.sample("z_{}".format(i), dist.Normal(loc, scale))
            assert loc.shape == ()
            assert scale.shape == ()
            assert z.shape == ()

    assert_ok(model, guide)


def test_plate_sequential():

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()
        for i in pyro.plate("plate", len(data)):
            loc = pyro.param("prior_loc_{}".format(i), torch.tensor(0.))
            z = pyro.sample("z_{}".format(i), dist.Normal(loc, 1.0))
            x = pyro.sample("x_{}".format(i), dist.Normal(z, scale), obs=data[i])
            assert loc.shape == ()
            assert z.shape == ()
            assert x.shape == ()

    def guide():
        for i in pyro.plate("plate", len(data)):
            loc = pyro.param("post_loc_{}".format(i), torch.tensor(0.))
            scale = pyro.param("post_scale_{}".format(i), torch.tensor(1.))
            z = pyro.sample("z_{}".format(i), dist.Normal(loc, scale))
            assert loc.shape == ()
            assert scale.shape == ()
            assert z.shape == ()

    assert_ok(model, guide)


def test_plate_sequential_subsample(subsample_size=2):

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()
        for i in pyro.plate("plate", len(data)):
            loc = pyro.param("prior_loc_{}".format(i), torch.tensor(0.))
            z = pyro.sample("z_{}".format(i), dist.Normal(loc, 1.0))
            x = pyro.sample("x_{}".format(i), dist.Normal(z, scale), obs=data[i])
            assert loc.shape == ()
            assert z.shape == ()
            assert x.shape == ()

    def guide():
        for i in pyro.plate("plate", len(data), subsample_size=subsample_size):
            loc = pyro.param("post_loc_{}".format(i), torch.tensor(0.))
            scale = pyro.param("post_scale_{}".format(i), torch.tensor(1.))
            z = pyro.sample("z_{}".format(i), dist.Normal(loc, scale))
            assert loc.shape == ()
            assert scale.shape == ()
            assert z.shape == ()

    assert_ok(model, guide)


def test_plate_parallel():

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()
        with pyro.plate("plate", len(data)):
            loc = pyro.param("prior_loc", torch.tensor(0.).expand(len(data)))
            z = pyro.sample("z", dist.Normal(loc, scale))
            x = pyro.sample("x", dist.Normal(z, scale), obs=data)
            assert loc.shape == (len(data),)
            assert z.shape == (len(data),)
            assert x.shape == (len(data),)

    def guide():
        with pyro.plate("plate", len(data)):
            scale = pyro.param("post_scale", torch.tensor(1.).expand(len(data)))
            loc = pyro.param("post_loc", torch.tensor(0.).expand(len(data)))
            z = pyro.sample("z", dist.Normal(loc, scale))
            assert scale.shape == (len(data),)
            assert loc.shape == (len(data),)
            assert z.shape == (len(data),)

    assert_ok(model, guide)


def test_plate_parallel_subsample(subsample_size=2):

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()
        with pyro.plate("plate", len(data)) as ind:
            loc = pyro.param("prior_loc", torch.tensor(0.).expand(len(data)))[ind]
            z = pyro.sample("z", dist.Normal(loc, scale))
            x = pyro.sample("x", dist.Normal(z, scale), obs=data[ind])
            assert loc.shape == (subsample_size,)
            assert z.shape == (subsample_size,)
            assert x.shape == (subsample_size,)

    def guide():
        with pyro.plate("plate", len(data), subsample_size=2) as ind:
            scale = pyro.param("post_scale", torch.tensor(1.).expand(len(data)))[ind]
            loc = pyro.param("post_loc", torch.tensor(0.).expand(len(data)))[ind]
            z = pyro.sample("z", dist.Normal(loc, scale))
            assert scale.shape == (subsample_size,)
            assert loc.shape == (subsample_size,)
            assert z.shape == (subsample_size,)

    assert_ok(model, guide)


def test_plate_parallel_autoguide():

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()
        with pyro.plate("plate", len(data)):
            loc = pyro.param("prior_loc", torch.tensor(0.).expand(len(data)))
            z = pyro.sample("z", dist.Normal(loc, scale))
            x = pyro.sample("x", dist.Normal(z, scale), obs=data)
            assert loc.shape == (len(data),)
            assert z.shape == (len(data),)
            assert x.shape == (len(data),)

    guide = AutoDiagonalNormal(model)

    assert_ok(model, guide)


def test_plate_parallel_decorator():

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()
        with pyro.plate("plate", len(data)):
            loc = pyro.param("prior_loc", torch.tensor(0.).expand(len(data)))
            z = pyro.sample("z", dist.Normal(loc, scale))
            x = pyro.sample("x", dist.Normal(z, scale), obs=data)
            assert loc.shape == (len(data),)
            assert z.shape == (len(data),)
            assert x.shape == (len(data),)

    @pyro.plate("plate", len(data))
    def guide():
        scale = pyro.param("post_scale", torch.tensor(1.).expand(len(data)))
        loc = pyro.param("post_loc", torch.tensor(0.).expand(len(data)))
        z = pyro.sample("z", dist.Normal(loc, scale))
        assert scale.shape == (len(data),)
        assert loc.shape == (len(data),)
        assert z.shape == (len(data),)

    assert_ok(model, guide)


def test_plate_sequential_nested():
    x_size, y_size = 4, 5
    x_data = torch.randn(x_size)
    y_data = torch.randn(y_size)
    xy_data = torch.randn(x_size, y_size)

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()

        x_plate = pyro.plate("x_plate", x_size)
        y_plate = pyro.plate("y_plate", y_size)
        for i in x_plate:
            pyro.sample("x_{}".format(i), dist.Normal(0., scale), obs=x_data[i])
        for j in y_plate:
            pyro.sample("y_{}".format(j), dist.Normal(0., scale), obs=y_data[j])
        for i in x_plate:
            for j in y_plate:
                pyro.sample("z_{}_{}".format(i, j), dist.Normal(0., scale), obs=xy_data[i, j])

    def guide():
        pass

    assert_ok(model, guide)


def test_plate_parallel_nested():
    x_size, y_size = 4, 5
    x_data = torch.randn(x_size)
    y_data = torch.randn(y_size)
    xy_data = torch.randn(x_size, y_size)

    def model():
        scale = pyro.param("prior_scale", torch.tensor(1.))
        assert scale.shape == ()

        x_plate = pyro.plate("x_plate", x_size, dim=-2)
        y_plate = pyro.plate("y_plate", y_size, dim=-1)
        with x_plate:
            pyro.sample("x", dist.Normal(0., scale), obs=x_data.unsqueeze(-1))  # gross
        with y_plate:
            pyro.sample("y", dist.Normal(0., scale), obs=y_data)
        with x_plate, y_plate:
            pyro.sample("z", dist.Normal(0., scale), obs=xy_data)

    def guide():
        pass

    assert_ok(model, guide)
