import argparse
import functools
from collections import OrderedDict

from pyro.generic import distributions as dist
from pyro.generic import handlers, ops, pyro, pyro_backend, transforms

MODELS = OrderedDict()


def register(rng_seed=None):
    def _register_fn(fn):
        @functools.wraps(fn)
        def _fn():
            with pyro.generic.model(rng_seed):
                return fn()
        MODELS[fn.__name__] = _fn

    return _register_fn


@register(rng_seed=1)
def logistic_regression():
    N, dim = 3000, 3
    # generic way to sample from distributions
    data = pyro.sample('data', dist.Normal(0., 1.), sample_shape=(N, dim))
    true_coefs = ops.arange(1., dim + 1.)
    logits = ops.sum(true_coefs * data, axis=-1)
    labels = pyro.sample('labels', dist.Bernoulli(logits=logits))

    def model(x, y):
        coefs = pyro.sample('coefs', dist.Normal(ops.zeros(dim), ops.ones(dim)))
        intercept = pyro.sample('intercept', dist.Normal(0., 1.))
        logits = ops.sum(coefs * x, axis=-1) + intercept
        return pyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

    return {'model': model, 'model_args': (data, labels)}


@register(rng_seed=1)
def neals_funnel():
    def model(dim):
        y = pyro.sample('y', dist.Normal(0, 3))
        pyro.sample('x', dist.TransformedDistribution(
            dist.Normal(ops.zeros(dim - 1), 1), transforms.AffineTransform(0, ops.exp(y / 2))))

    return {'model': model, 'model_args': (10,)}


@register(rng_seed=1)
def eight_schools():
    data = {
        "J": 8,
        "y": ops.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": ops.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }

    def model(J, y, sigma):
        mu = pyro.sample('mu', dist.Normal(0, 5))
        tau = pyro.sample('tau', dist.HalfCauchy(5))
        with pyro.plate('J', J):
            theta = pyro.sample('theta', dist.Normal(mu, tau))
            pyro.sample('obs', dist.Normal(theta, sigma), obs=y)

    return {'model': model, 'model_kwargs': data}


@register(rng_seed=1)
def beta_binomial():
    true_probs = ops.tensor([[0.7, 0.4], [0.6, 0.4]])
    total_count = ops.tensor([[1000, 600], [400, 800]])
    data = pyro.sample('data', dist.Binomial(total_count=total_count, probs=true_probs),
                       sample_shape=(10,))

    def model(data):
        with pyro.plate("plate_0", data.shape[-1]):
            alpha = pyro.sample("alpha", dist.HalfCauchy(1.))
            beta = pyro.sample("beta", dist.HalfCauchy(1.))
            with pyro.plate("plate_1", data.shape[-2]):
                probs = pyro.sample("probs", dist.Beta(alpha, beta))
                with pyro.plate("data", data.shape[0]):
                    pyro.sample("binomial", dist.Binomial(probs=probs, total_count=total_count), obs=data)

    return {'model': model, 'model_args': (data,)}


def check_model(backend, name):
    get_model = MODELS[name]
    print('Running model "{}" on backend "{}".'.format(name, args.backend))
    with pyro_backend(backend):
        f = get_model()
        model, model_args, model_kwargs = f['model'], f.get('model_args', ()), f.get('model_kwargs', {})
        print('Sample from prior...')
        model(*model_args, **model_kwargs)
        print('Trace model...')
        handlers.trace(model).get_trace(*model_args, **model_kwargs)


def main(args):
    for name in MODELS:
        check_model(args.backend, name)
        # if args.run_mcmc:
        #     print('Run inference using MCMC...')
        #     nuts_kernel = infer.NUTS(model=model)
        #     mcmc = infer.MCMC(nuts_kernel, num_samples=100, warmup_steps=100)
        #     mcmc.run(*model_args, **model_kwargs)
        #     mcmc.summary()


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.4.1')
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-b", "--backend", default="pyro")
    parser.add_argument("--run-mcmc", default=True, action="store_true")
    args = parser.parse_args()
    main(args)
