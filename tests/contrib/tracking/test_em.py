from __future__ import absolute_import, division, print_function

import math

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.tracking.assignment import MarginalAssignment
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from pyro.optim.multi import MixedMultiOptimizer, Newton, PyroMultiOptimizer


def make_args():
    args = type('Args', (), {})  # A fake ArgumentParser.parse_args()
    args.max_num_objects = 4
    args.num_real_detections = 13
    args.num_fake_detections = 3
    args.expected_num_objects = 2
    args.init_noise_scale = 0.1

    # TODO Is it correct to detach gradients of assignments?
    # Detaching is indeed required for the Hessian to be block-diagonal,
    # but it is unclear whether convergence would be faster if we applied
    # a blockwise method (Newton) to the full Hessian, without detaching.
    args.assignment_grad = False

    return args


@poutine.broadcast
def model(detections, args):
    noise_scale = pyro.param('noise_scale')
    objects = pyro.param('objects_loc').squeeze(-1)
    num_detections, = detections.shape
    max_num_objects, = objects.shape

    # Existence part.
    p_exists = args.expected_num_objects / max_num_objects
    with pyro.iarange('objects_iarange', max_num_objects):
        exists = pyro.sample('exists', dist.Bernoulli(p_exists))
        with poutine.scale(scale=exists):
            pyro.sample('objects', dist.Normal(0., 1.), obs=objects)

    # Assignment part.
    p_fake = args.num_fake_detections / num_detections
    with pyro.iarange('detections_iarange', num_detections):
        assign_probs = torch.empty(max_num_objects + 1)
        assign_probs[:-1] = (1 - p_fake) / max_num_objects
        assign_probs[-1] = p_fake
        assign = pyro.sample('assign', dist.Categorical(logits=assign_probs))
        is_fake = (assign == assign.shape[-1] - 1)
        is_real = ~is_fake
        with poutine.scale(scale=is_real.type_as(assign_probs)):
            objects_plus_bogus = torch.zeros(max_num_objects + 1)
            objects_plus_bogus[:max_num_objects] = objects
            pyro.sample('real_detections', dist.Normal(objects_plus_bogus[assign], noise_scale),
                        obs=detections)
        with poutine.scale(scale=is_fake.type_as(assign_probs)):
            pyro.sample('fake_detections', dist.Normal(0., 1.),
                        obs=detections)


# This should match detection_model's existence part.
def exists_log_likelihood(objects, args):
    p_exists = args.expected_num_objects / args.max_num_objects
    real_part = dist.Normal(0., 1.).log_prob(objects)
    real_part = real_part + math.log(p_exists)
    spurious_part = torch.empty(real_part.shape).fill_(math.log(1 - p_exists))
    return torch.stack([spurious_part, real_part], -1)


# This should match detection_model's assignment part.
def assign_log_likelihood(objects, detections, noise_scale, args):
    num_detections = len(detections)
    p_fake = args.num_fake_detections / num_detections
    real_part = dist.Normal(objects, noise_scale).log_prob(detections)
    real_part = real_part + math.log((1 - p_fake) / args.max_num_objects)
    fake_part = dist.Normal(0., 1.).log_prob(detections)
    fake_part = fake_part + math.log(p_fake)
    return torch.cat([real_part, fake_part], -1)


def guide(detections, args):
    noise_scale = pyro.param('noise_scale')  # trained by SVI
    objects = pyro.param('objects_loc').squeeze(-1)  # trained by M-step of EM
    num_detections, = detections.shape
    max_num_objects, = objects.shape

    with torch.set_grad_enabled(args.assignment_grad):
        # Evaluate log likelihoods. TODO make this more pyronic.
        exists_loglike = exists_log_likelihood(objects, args)
        assign_loglike = assign_log_likelihood(objects, detections.unsqueeze(-1), noise_scale, args)
        assert exists_loglike.shape == (max_num_objects, 2)
        assert assign_loglike.shape == (num_detections, max_num_objects + 1)

        # Compute soft assignments.
        exists_logits = exists_loglike[:, 1] - exists_loglike[:, 0]
        assign_logits = assign_loglike[:, :-1] - assign_loglike[:, -1:]
        assignment = MarginalAssignment(exists_logits, assign_logits, bp_iters=10)

    with pyro.iarange('objects_iarange', max_num_objects):
        pyro.sample('exists', assignment.exists_dist,
                    infer={'enumerate': 'parallel'})
    with pyro.iarange('detections_iarange', num_detections):
        pyro.sample('assign', assignment.assign_dist,
                    infer={'enumerate': 'parallel'})


def generate_data(args):
    num_objects = args.expected_num_objects
    true_objects = torch.randn(num_objects)
    true_assign = dist.Categorical(torch.ones(args.num_real_detections, num_objects)).sample()
    real_detections = true_objects[true_assign]
    real_detections = real_detections + args.init_noise_scale * torch.randn(real_detections.shape)
    fake_detections = torch.randn(args.num_fake_detections)
    detections = torch.cat([real_detections, fake_detections])
    assert detections.shape == (args.num_real_detections + args.num_fake_detections,)
    return detections


@pytest.mark.parametrize('assignment_grad', [False, True])
def test_em(assignment_grad):
    args = make_args()
    args.assignment_grad = assignment_grad
    detections = generate_data(args)

    pyro.clear_param_store()
    pyro.param('noise_scale', torch.tensor(args.init_noise_scale),
               constraint=constraints.positive)
    pyro.param('objects_loc', torch.randn(args.max_num_objects, 1))

    # Learn object_loc via EM algorithm.
    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    newton = Newton(trust_radii={'objects_loc': 1.0})
    for step in range(10):
        # Detach previous iterations.
        objects_loc = pyro.param('objects_loc').detach_().requires_grad_()
        loss = elbo.differentiable_loss(model, guide, detections, args)  # E-step
        newton.step(loss, {'objects_loc': objects_loc})  # M-step
        print('step {}, loss = {}'.format(step, loss.item()))


@pytest.mark.parametrize('assignment_grad', [False, True])
def test_em_nested_in_svi(assignment_grad):
    args = make_args()
    args.assignment_grad = assignment_grad
    detections = generate_data(args)

    pyro.clear_param_store()
    pyro.param('noise_scale', torch.tensor(args.init_noise_scale),
               constraint=constraints.positive)
    pyro.param('objects_loc', torch.randn(args.max_num_objects, 1))

    # Learn object_loc via EM and noise_scale via SVI.
    optim = Adam({'lr': 0.1})
    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    newton = Newton(trust_radii={'objects_loc': 1.0})
    svi = SVI(poutine.block(model, hide=['objects_loc']),
              poutine.block(guide, hide=['objects_loc']), optim, elbo)
    for svi_step in range(50):
        for em_step in range(2):
            objects_loc = pyro.param('objects_loc').detach_().requires_grad_()
            assert pyro.param('objects_loc').grad_fn is None
            loss = elbo.differentiable_loss(model, guide, detections, args)  # E-step
            updated = newton.get_step(loss, {'objects_loc': objects_loc})  # M-step
            assert updated['objects_loc'].grad_fn is not None
            pyro.get_param_store().replace_param('objects_loc', updated['objects_loc'], objects_loc)
            assert pyro.param('objects_loc').grad_fn is not None
        loss = svi.step(detections, args)
        print('step {: >2d}, loss = {:0.6f}, noise_scale = {:0.6f}'.format(
            svi_step, loss, pyro.param('noise_scale').item()))


def test_svi_multi():
    args = make_args()
    args.assignment_grad = True
    detections = generate_data(args)

    pyro.clear_param_store()
    pyro.param('noise_scale', torch.tensor(args.init_noise_scale),
               constraint=constraints.positive)
    pyro.param('objects_loc', torch.randn(args.max_num_objects, 1))

    # Learn object_loc via Newton and noise_scale via Adam.
    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    adam = PyroMultiOptimizer(Adam({'lr': 0.1}))
    newton = Newton(trust_radii={'objects_loc': 1.0})
    optim = MixedMultiOptimizer([(['noise_scale'], adam),
                                 (['objects_loc'], newton)])
    for svi_step in range(50):
        with poutine.trace(param_only=True) as param_capture:
            loss = elbo.differentiable_loss(model, guide, detections, args)
        params = {name: pyro.param(name).unconstrained()
                  for name in param_capture.trace.nodes.keys()}
        optim.step(loss, params)
        print('step {: >2d}, loss = {:0.6f}, noise_scale = {:0.6f}'.format(
            svi_step, loss.item(), pyro.param('noise_scale').item()))
