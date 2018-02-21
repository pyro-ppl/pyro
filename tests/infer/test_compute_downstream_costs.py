from __future__ import absolute_import, division, print_function

import numpy as np

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


"""
def model_old(include_obs=True):
    p0 = Variable(torch.Tensor([0.25 + 0.55 * include_obs]), requires_grad=True)
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
"""

def model(include_obs=True, include_single=False, include_inner_1=False):
    if include_obs:
        p0 = Variable(torch.Tensor([np.exp(-0.25)]), requires_grad=True)
    else:
        p0 = Variable(torch.Tensor([np.exp(-0.75)]), requires_grad=True)
    pyro.sample("a1", dist.Bernoulli(p0))
    if include_single:
        with pyro.iarange("iarange_single", 5, 5) as ind_single:
            pyro.sample("b0", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_single)]))
    with pyro.iarange("iarange_outer", 2, 2) as ind_outer:
        pyro.sample("b1", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_outer)]))
        if include_inner_1:
            with pyro.iarange("iarange_inner_1", 3, 3) as ind_inner:
                c1 = pyro.sample("c1", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), 1]))
                c2 = pyro.sample("c2", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), len(ind_outer)]))
                c3 = pyro.sample("c3", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), 1]))
        with pyro.iarange("iarange_inner_2", 4, 4) as ind_inner:
            d1 = pyro.sample("d1", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), 1]))
            d2 = pyro.sample("d2", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), len(ind_outer)]))
            if include_obs:
                pyro.sample("obs", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), len(ind_outer)]),
                            obs=Variable(torch.ones(d2.size())))


def test_compute_downstream_costs(include_inner_1):
    guide_trace = poutine.trace(model,
    				graph_type="dense").get_trace(include_obs=False, include_inner_1=include_inner_1)
    if 0:
        print("*** GUIDE TRACE ***\n", guide_trace.nodes)
        for node in guide_trace.nodes:
            print(guide_trace.nodes[node])
    model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                graph_type="dense").get_trace(include_obs=True, include_inner_1=include_inner_1)
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

    dc, dc_nodes = _compute_downstream_costs(model_trace, guide_trace,
                                   model_vec_md_nodes, guide_vec_md_nodes,
                                   non_reparam_nodes, include_nodes=True)

    #for k in dc:
    #    print('dc[%s]' % k, dc[k].size(), dc_nodes[k])
    #    print('[%s]' % k, guide_trace.nodes[k]['batch_log_pdf'].size())

    #print("guide trace:", guide_trace.nodes['c4'])
    #print("guide fn:", guide_trace.nodes['c4']['fn'].base_dist.probs)
    #print("model trace:", model_trace.nodes['c4'])
    #print("model fn:", model_trace.nodes['c4']['fn'].base_dist.probs)

    #print("downstreamcost nodes", dc_nodes)
    #expected_nodes = {'b1': {'b1', 'c3', 'c1', 'b2', 'c2', 'c4'}, 'c3': {'c3', 'c4'}, 'c1': {'c3', 'c1', 'c2', 'c4'},
    #                  'b2': {'c3', 'c1', 'b2', 'c2', 'c4'}, 'a2': {'c1', 'c2', 'b1', 'c3', 'a2', 'b2', 'c4'},
    #                  'a1': {'c1', 'a1', 'c2', 'b1', 'c3', 'a2', 'b2', 'c4'}, 'c2': {'c3', 'c2', 'c4'}, 'c4': {'c4'}}
    #assert(dc_nodes == expected_nodes)
    expected_b1 = (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf']).sum(0, keepdim=False)
    expected_b1 += 0.5 * (model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']).sum(0, keepdim=False)
    expected_b1 += (model_trace.nodes['b1']['batch_log_pdf'] - guide_trace.nodes['b1']['batch_log_pdf'])
    expected_b1 += model_trace.nodes['obs']['batch_log_pdf'].sum(0, keepdim=False)
    if include_inner_1:
        expected_b1 += 0.5 * (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf']).sum(0, keepdim=False)
        expected_b1 += (model_trace.nodes['c2']['batch_log_pdf'] - guide_trace.nodes['c2']['batch_log_pdf']).sum(0, keepdim=False)
        expected_b1 += 0.5 * (model_trace.nodes['c3']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf']).sum(0, keepdim=False)

    if include_inner_1:
        expected_c3 = (model_trace.nodes['c3']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf'])
        expected_c3 += (model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']).sum()
        expected_c3 += (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf']).sum()
        expected_c3 += model_trace.nodes['obs']['batch_log_pdf'].sum()

        expected_c2 = (model_trace.nodes['c2']['batch_log_pdf'] - guide_trace.nodes['c2']['batch_log_pdf'])
        expected_c2 += (model_trace.nodes['c3']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf'])
        expected_c2 += (model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']).sum(0, keepdim=False)
        expected_c2 += (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf']).sum(0, keepdim=False)
        expected_c2 += model_trace.nodes['obs']['batch_log_pdf'].sum(0, keepdim=False)

    expected_d1 = model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']
    expected_d1 += (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf']).sum(1, keepdim=True)
    expected_d1 += model_trace.nodes['obs']['batch_log_pdf'].sum(1, keepdim=True)

    expected_d2 = (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf'])
    expected_d2 += model_trace.nodes['obs']['batch_log_pdf']

    #print("expected_b1", expected_b1.size(), "\n", expected_b1.data.numpy())
    #print("dc['b1']",  dc['b1'].size(), "\n", dc['b1'].data.numpy())
    #print("delta c4", expected_c4.data.numpy()-dc['c4'].data.numpy())
    assert_equal(expected_c2, dc['c2'], prec=1.0e-6)
    assert_equal(expected_c3, dc['c3'], prec=1.0e-6)
    assert_equal(expected_d2, dc['d2'], prec=1.0e-6)
    assert_equal(expected_d1, dc['d1'], prec=1.0e-6)
##    #assert_equal(expected_b1, dc['b1'], prec=1.0e-6)
    #assert_equal(expected_c3, dc['c3'], prec=1.0e-6)
    #for k in expected_nodes:
        #print(k, guide_trace.nodes[k]['batch_log_pdf'].size(), guide_trace.nodes[k]['fn'].event_shape,
        #        guide_trace.nodes[k]['fn'].batch_shape)
        #:print("<<<", k, "blp", guide_trace.nodes[k]['batch_log_pdf'].size(), "dc", dc[k].size())
        #assert(guide_trace.nodes[k]['batch_log_pdf'].size() == dc[k].size())
    return

print('**********************************')
print("TRUE")
print('**********************************')
test_compute_downstream_costs(include_inner_1=True)
print('**********************************')
print("\nFALSE")
print('**********************************')
#test_compute_downstream_costs(include_inner_1=False)
