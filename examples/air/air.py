"""
An implementation of the model described in [1].

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from modules import Identity, Encoder, Decoder, MLP, Predict
from pyro.distributions import Bernoulli, Uniform, Delta
from pyro.util import ng_zeros, ng_ones, zeros


# Default prior success probability for z_pres.
def default_z_pres_prior_p(t):
    return 0.5


ModelState = namedtuple('ModelState', ['x', 'z_pres', 'z_where'])
GuideState = namedtuple('GuideState', ['h', 'c', 'bl_h', 'bl_c', 'z_pres', 'z_where', 'z_what'])


class AIR(nn.Module):
    def __init__(self,
                 num_steps,
                 x_size,
                 window_size,
                 z_what_size,
                 rnn_hidden_size,
                 encoder_net=[],
                 decoder_net=[],
                 predict_net=[],
                 embed_net=None,
                 bl_predict_net=[],
                 non_linearity='ReLU',
                 decoder_output_bias=None,
                 decoder_output_use_sigmoid=False,
                 use_masking=True,
                 use_baselines=True,
                 baseline_scalar=None,
                 fudge_z_pres=False,
                 use_cuda=False):

        super(AIR, self).__init__()

        self.num_steps = num_steps
        self.x_size = x_size
        self.window_size = window_size
        self.z_what_size = z_what_size
        self.rnn_hidden_size = rnn_hidden_size
        self.use_masking = use_masking and not fudge_z_pres
        self.use_baselines = use_baselines and not fudge_z_pres
        self.baseline_scalar = baseline_scalar
        self.fudge_z_pres = fudge_z_pres
        self.use_cuda = use_cuda

        self.z_pres_size = 1
        self.z_where_size = 3
        # By making these parameters they will be moved to the gpu
        # when necessary. (They are not registered with pyro for
        # optimization.)
        self.z_where_mu_prior = nn.Parameter(torch.FloatTensor([3.0, 0, 0]), requires_grad=False)
        self.z_where_sigma_prior = nn.Parameter(torch.FloatTensor([0.1, 1, 1]), requires_grad=False)

        # Create nn modules.
        rnn_input_size = x_size ** 2 if embed_net is None else embed_net[-1]
        rnn_input_size += self.z_where_size + z_what_size + self.z_pres_size
        nl = getattr(nn, non_linearity)

        self.rnn = nn.LSTMCell(rnn_input_size, rnn_hidden_size)
        self.encode = Encoder(window_size ** 2, encoder_net, z_what_size, nl)
        self.decode = Decoder(window_size ** 2, decoder_net, z_what_size,
                              decoder_output_bias, decoder_output_use_sigmoid, nl)
        self.predict = Predict(rnn_hidden_size, predict_net, self.z_pres_size, self.z_where_size, nl)
        self.embed = Identity() if embed_net is None else MLP(x_size ** 2, embed_net, nl, True)

        self.bl_rnn = nn.LSTMCell(rnn_input_size, rnn_hidden_size)
        self.bl_predict = MLP(rnn_hidden_size, bl_predict_net + [1], nl)
        self.bl_embed = Identity() if embed_net is None else MLP(x_size ** 2, embed_net, nl, True)

        # Create parameters.
        self.h_init = zeros(1, rnn_hidden_size)
        self.c_init = zeros(1, rnn_hidden_size)
        self.bl_h_init = zeros(1, rnn_hidden_size)
        self.bl_c_init = zeros(1, rnn_hidden_size)
        self.z_where_init = zeros(1, self.z_where_size)
        self.z_what_init = zeros(1, self.z_what_size)

        if use_cuda:
            self.cuda()

    def prior(self, n, **kwargs):
        pyro.module("decode", self.decode)
        return self.local_model(n, **kwargs)

    def model(self, data, batch_size, **kwargs):
        pyro.module("decode", self.decode)
        with pyro.iarange('data', data.size(0), use_cuda=self.use_cuda) as ix:
            return self.local_model(batch_size, data[ix], **kwargs)

    def local_model(self, n, batch=None, **kwargs):

        state = ModelState(
            x=self.ng_zeros([n, self.x_size, self.x_size]),
            z_pres=self.ng_ones([n, self.z_pres_size]),
            z_where=None)

        z_pres = []
        z_where = []

        for t in range(self.num_steps):
            state = self.model_step(t, n, state, batch, **kwargs)
            z_where.append(state.z_where)
            z_pres.append(state.z_pres)

        return (z_where, z_pres), state.x

    def model_step(self, t, n, prev, batch, z_pres_prior_p=default_z_pres_prior_p):

        # Sample presence indicators.
        if not self.fudge_z_pres:
            z_pres_dist = Bernoulli(z_pres_prior_p(t) * prev.z_pres)
        else:
            z_pres_dist = Uniform(self.ng_zeros(n, 1), self.ng_ones(n, 1))
        z_pres = pyro.sample('z_pres_{}'.format(t), z_pres_dist)

        # If zero is sampled for a data point, then no more objects
        # will be added to its output image. We can't
        # straight-forwardly avoid generating further objects, so
        # instead we zero out the log_pdf of future choices.
        sample_mask = z_pres if self.use_masking else None

        # Sample attention window position.
        z_where = pyro.sample('z_where_{}'.format(t),
                              dist.normal,
                              self.z_where_mu_prior,
                              self.z_where_sigma_prior,
                              batch_size=n,
                              log_pdf_mask=sample_mask)

        # Sample latent code for contents of the attention window.
        z_what = pyro.sample('z_what_{}'.format(t),
                             dist.normal,
                             self.ng_zeros([self.z_what_size]),
                             self.ng_ones([self.z_what_size]),
                             batch_size=n,
                             log_pdf_mask=sample_mask)

        # Map latent code to pixel space.
        y_att = self.decode(z_what)

        # Position/scale attention window within larger image.
        y = window_to_image(z_where, self.window_size, self.x_size, y_att)

        # Combine the image generated at this step with the image so far.
        # (Note that there's no notion of occlusion here. Overlapping
        # objects can create pixel intensities > 1.)
        x = prev.x + (y * z_pres.view(-1, 1, 1))

        if batch is not None:
            # Add observations.

            # Observations are made as soon as we are done generating
            # objects for a data point. This ensures that future
            # discrete choices are not included in the ELBO. i.e. The
            # corresponding log(q/p) in the objectives will be zero
            # since we mask out all future choices and make no further
            # observations for data points that are complete.

            if not self.use_masking:
                observe_mask = 1.0
            elif t == (self.num_steps - 1):
                observe_mask = prev.z_pres
            else:
                observe_mask = prev.z_pres - z_pres

            if self.use_masking or t == (self.num_steps - 1):
                pyro.observe("obs_{}".format(t),
                             dist.normal,
                             batch.view(n, -1),
                             x.view(n, -1),
                             self.ng_ones(x.view(n, -1).size()) * 0.3,
                             log_pdf_mask=observe_mask)

        return ModelState(x=x, z_pres=z_pres, z_where=z_where)

    def guide(self, data, batch_size, **kwargs):
        pyro.module('rnn', self.rnn),
        pyro.module('predict', self.predict),
        pyro.module('encode', self.encode),
        pyro.module('embed', self.embed),
        pyro.module('bl_rnn', self.bl_rnn, tags='baseline'),
        pyro.module('bl_predict', self.bl_predict, tags='baseline'),
        pyro.module('bl_embed', self.bl_embed, tags='baseline')

        pyro.param('h_init', self.h_init)
        pyro.param('c_init', self.c_init)
        pyro.param('z_where_init', self.z_where_init)
        pyro.param('z_what_init', self.z_what_init)
        pyro.param('bl_h_init', self.bl_h_init, tags='baseline')
        pyro.param('bl_c_init', self.bl_c_init, tags='baseline')

        with pyro.iarange('data', data.size(0), subsample_size=batch_size, use_cuda=self.use_cuda) as ix:
            batch = data[ix]
            return self.local_guide(batch.size(0), batch)

    def local_guide(self, n, batch):

        # Embed inputs.
        flattened_batch = batch.view(n, -1)
        inputs = {
            'raw': batch,
            'embed': self.embed(flattened_batch),
            'bl_embed': self.bl_embed(flattened_batch)
        }

        # Initial state.
        state = GuideState(
            h=batch_expand(self.h_init, n),
            c=batch_expand(self.c_init, n),
            bl_h=batch_expand(self.bl_h_init, n),
            bl_c=batch_expand(self.bl_c_init, n),
            z_pres=self.ng_ones(n, self.z_pres_size),
            z_where=batch_expand(self.z_where_init, n),
            z_what=batch_expand(self.z_what_init, n))

        z_pres = []
        z_where = []

        for t in range(self.num_steps):
            state = self.guide_step(t, n, state, inputs)
            z_where.append(state.z_where)
            z_pres.append(state.z_pres)

        return z_where, z_pres

    def guide_step(self, t, n, prev, inputs):

        rnn_input = torch.cat((inputs['embed'], prev.z_where, prev.z_what, prev.z_pres), 1)
        h, c = self.rnn(rnn_input, (prev.h, prev.c))
        z_pres_p, z_where_mu, z_where_sigma = self.predict(h)

        # Compute baseline estimates for discrete choice z_pres.
        bl_value, bl_h, bl_c = self.baseline_step(prev, inputs)

        # Sample presence.
        if not self.fudge_z_pres:
            z_pres_dist = Bernoulli(z_pres_p * prev.z_pres)
        else:
            z_pres_dist = Delta(z_pres_p * prev.z_pres)
        z_pres = pyro.sample('z_pres_{}'.format(t),
                             z_pres_dist,
                             baseline=dict(baseline_value=bl_value))

        log_pdf_mask = z_pres if self.use_masking else None

        z_where = pyro.sample('z_where_{}'.format(t),
                              dist.normal,
                              z_where_mu + self.z_where_mu_prior,
                              z_where_sigma * self.z_where_sigma_prior,
                              log_pdf_mask=log_pdf_mask)

        # Figure 2 of [1] shows x_att depending on z_where and h,
        # rather than z_where and x as here, but I think this is
        # correct.
        x_att = image_to_window(z_where, self.window_size, self.x_size, inputs['raw'])

        # Encode attention windows.
        z_what_mu, z_what_sigma = self.encode(x_att)

        z_what = pyro.sample('z_what_{}'.format(t),
                             dist.normal,
                             z_what_mu,
                             z_what_sigma,
                             log_pdf_mask=log_pdf_mask)

        return GuideState(h=h, c=c, bl_h=bl_h, bl_c=bl_c, z_pres=z_pres, z_where=z_where, z_what=z_what)

    def baseline_step(self, prev, inputs):
        if not self.use_baselines:
            return None, None, None

        # Prevent gradients flowing back from baseline loss to
        # inference net by detaching from graph here.
        rnn_input = torch.cat((inputs['bl_embed'],
                               prev.z_where.detach(),
                               prev.z_what.detach(),
                               prev.z_pres.detach()), 1)
        bl_h, bl_c = self.bl_rnn(rnn_input, (prev.bl_h, prev.bl_c))
        bl_value = self.bl_predict(bl_h)

        # Zero out values for finished data points. This avoids adding
        # superfluous terms to the loss.
        if self.use_masking:
            bl_value = bl_value * prev.z_pres

        # The value that the baseline net is estimating can be very
        # large. An option to scale the nets output is provided
        # to make it easier for the net to output values of this
        # scale.
        if self.baseline_scalar is not None:
            bl_value = bl_value * self.baseline_scalar

        return bl_value, bl_h, bl_c

    # Helpers to create zeros/ones on cpu/gpu as appropriate.
    def ng_zeros(self, *args, **kwargs):
        t = ng_zeros(*args, **kwargs)
        if self.use_cuda:
            t = t.cuda()
        return t

    def ng_ones(self, *args, **kwargs):
        t = ng_ones(*args, **kwargs)
        if self.use_cuda:
            t = t.cuda()
        return t


# Spatial transformer helpers.

expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])


def expand_z_where(z_where):
    # Take a batch of three-vectors, and massages them into a batch of
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = z_where.size(0)
    out = torch.cat((ng_zeros([1, 1]).type_as(z_where).expand(n, 1), z_where), 1)
    ix = Variable(expansion_indices)
    if z_where.is_cuda:
        ix = ix.cuda()
    out = torch.index_select(out, 1, ix)
    out = out.view(n, 2, 3)
    return out


# Scaling by `1/scale` here is unsatisfactory, as `scale` could be
# zero.
def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    n = z_where.size(0)
    out = torch.cat((ng_ones([1, 1]).type_as(z_where).expand(n, 1), -z_where[:, 1:]), 1)
    # Divide all entries by the scale.
    out = out / z_where[:, 0:1]
    return out


def window_to_image(z_where, window_size, image_size, windows):
    n = windows.size(0)
    assert windows.size(1) == window_size ** 2, 'Size mismatch.'
    theta = expand_z_where(z_where)
    grid = F.affine_grid(theta, torch.Size((n, 1, image_size, image_size)))
    out = F.grid_sample(windows.view(n, 1, window_size, window_size), grid)
    return out.view(n, image_size, image_size)


def image_to_window(z_where, window_size, image_size, images):
    n = images.size(0)
    assert images.size(1) == images.size(2) == image_size, 'Size mismatch.'
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = F.affine_grid(theta_inv, torch.Size((n, 1, window_size, window_size)))
    out = F.grid_sample(images.view(n, 1, image_size, image_size), grid)
    return out.view(n, -1)


# Helper to expand parameters to the size of the mini-batch. I would
# like to remove this and just write `t.expand(n, -1)` inline, but the
# `-1` argument of `expand` doesn't seem to work with PyTorch 0.2.0.
def batch_expand(t, n):
    return t.expand(n, t.size(1))
