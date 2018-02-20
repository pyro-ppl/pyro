from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.infer.tracegraph_elbo import _compute_downstream_costs
from pyro.poutine.util import prune_subsample_sites
from tests.common import assert_equal

import sys

def model(include_obs=True):
    p0 = Variable(torch.Tensor([0.25 + 0.25 * include_obs]), requires_grad=True)
    pyro.sample("a1", dist.Bernoulli(p0))
    pyro.sample("a2", dist.Normal(p0, p0))
    with pyro.iarange("iarange_outer0", 5, 5) as ind_outer:
        pyro.sample("b0", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_outer), 1]))
    with pyro.iarange("iarange_outer", 2, 2) as ind_outer:
        pyro.sample("b1", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_outer), 1]))
        pyro.sample("b2", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_outer), 1]))
        with pyro.iarange("iarange_inner0", 3, 3) as ind_inner:
            c0 = pyro.sample("c0", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner)]))
            c00 = pyro.sample("c00", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_outer), len(ind_inner)]))
            c000 = pyro.sample("c000", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner)]))
        with pyro.iarange("iarange_inner", 4, 4) as ind_inner:
            c1 = pyro.sample("c1", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner)]))
            c2 = pyro.sample("c2", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_outer), len(ind_inner)]))
            c3 = pyro.sample("c3", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner)]))
            c4 = pyro.sample("c4", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_outer), len(ind_inner)]))
            if include_obs:
                pyro.sample("obs", dist.Bernoulli(c1 * c2), obs=Variable(torch.ones(c2.size())))

def test_compute_downstream_costs():
    guide_trace = poutine.trace(model,
    				graph_type="dense").get_trace(include_obs=False)
    if 0:
        print("*** GUIDE TRACE ***\n", guide_trace.nodes)
        for node in guide_trace.nodes:
            print(guide_trace.nodes[node])
    model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                graph_type="dense").get_trace(include_obs=True)
    #model_trace.compute_batch_log_pdf()
    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)
    model_trace.compute_batch_log_pdf()
    guide_trace.compute_batch_log_pdf()
    if 0:
        print("\n\n*** MODEL TRACE ***\n", model_trace.nodes)
        for node in model_trace.nodes:
            print(model_trace.nodes[node])

    guide_vec_md_info = guide_trace.graph["vectorized_map_data_info"]
    model_vec_md_info = model_trace.graph["vectorized_map_data_info"]
    guide_vec_md_condition = guide_vec_md_info['rao-blackwellization-condition']
    model_vec_md_condition = model_vec_md_info['rao-blackwellization-condition']
    do_vec_rb = guide_vec_md_condition and model_vec_md_condition
    guide_vec_md_nodes = guide_vec_md_info['nodes'] if do_vec_rb else set()
    model_vec_md_nodes = model_vec_md_info['nodes'] if do_vec_rb else set()
    non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)
    #print("stacks", guide_vec_md_info['vec_md_stacks'])

    dc, dc_nodes = _compute_downstream_costs(model_trace, guide_trace,
                                   model_vec_md_nodes, guide_vec_md_nodes,
                                   non_reparam_nodes, include_nodes=True)

    for k in dc:
        print('dc[%s]' % k, dc[k].size(), dc_nodes[k])

    #print("downstreamcost nodes", dc_nodes)
    expected_nodes = {'b1': {'b1', 'c3', 'c1', 'b2', 'c2', 'c4'}, 'c3': {'c3', 'c4'}, 'c1': {'c3', 'c1', 'c2', 'c4'},
                      'b2': {'c3', 'c1', 'b2', 'c2', 'c4'}, 'a2': {'c1', 'c2', 'b1', 'c3', 'a2', 'b2', 'c4'},
                      'a1': {'c1', 'a1', 'c2', 'b1', 'c3', 'a2', 'b2', 'c4'}, 'c2': {'c3', 'c2', 'c4'}, 'c4': {'c4'}}
    #assert(dc_nodes == expected_nodes)
    expected_c3 = model_trace.nodes['c3']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf']
    #print("c3 terms",  model_trace.nodes['c3']['batch_log_pdf'], guide_trace.nodes['c3']['batch_log_pdf'])
    expected_c3 += (model_trace.nodes['c4']['batch_log_pdf'] - guide_trace.nodes['c4']['batch_log_pdf']).sum(0)
    #print("ex3", expected_c3)
    expected_c3 += model_trace.nodes['obs']['batch_log_pdf'].sum(0)
    expected_c4 = (model_trace.nodes['c4']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf'])
    expected_c4 += model_trace.nodes['obs']['batch_log_pdf']
    #print("q(4)", guide_trace.nodes['c4']['batch_log_pdf'], guide_trace.nodes['c4']['value'])
    #print("p(4)", model_trace.nodes['c4']['batch_log_pdf'], model_trace.nodes['c4']['value'])
    #print("p(obs)", model_trace.nodes['obs']['batch_log_pdf'], model_trace.nodes['obs']['value'])
    #assert_equal(expected_c4, dc['c4'], prec=1.0e-6)
    #print("expected_c3, dc['c3']", expected_c3, dc['c3'], "obs", model_trace.nodes['obs']['batch_log_pdf'].size())
    #assert_equal(expected_c3, dc['c3'], prec=1.0e-6)
    for k in expected_nodes:
        #print(k, guide_trace.nodes[k]['batch_log_pdf'].size(), guide_trace.nodes[k]['fn'].event_shape,
        #        guide_trace.nodes[k]['fn'].batch_shape)
        #:print("<<<", k, "blp", guide_trace.nodes[k]['batch_log_pdf'].size(), "dc", dc[k].size())
        assert(guide_trace.nodes[k]['batch_log_pdf'].size() == dc[k].size())
    return

test_compute_downstream_costs()
