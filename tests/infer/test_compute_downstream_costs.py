from __future__ import absolute_import, division, print_function

import math

import networkx
import pytest
import torch
from torch.autograd import Variable, variable

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.tracegraph_elbo import _compute_downstream_costs
from pyro.infer.util import MultiViewTensor
from pyro.poutine.util import prune_subsample_sites
from tests.common import assert_equal


def _brute_force_compute_downstream_costs(model_trace, guide_trace,  #
                                          model_iarange_nodes, guide_iarange_nodes,  #
                                          non_reparam_nodes):

    guide_nodes = [x for x in guide_trace.nodes if guide_trace.nodes[x]["type"] == "sample"]
    downstream_costs, downstream_guide_cost_nodes = {}, {}
    stacks = model_trace.graph["iarange_info"]['iarange_stacks']

    def n_compatible_indices(dest_node, source_node):
        n_compatible = 0
        for xframe, yframe in zip(stacks[source_node], stacks[dest_node]):
            if xframe.name == yframe.name:
                n_compatible += 1
        return n_compatible

    for node in guide_nodes:
        downstream_costs[node] = MultiViewTensor(model_trace.nodes[node]['batch_log_pdf'] -
                                                 guide_trace.nodes[node]['batch_log_pdf'])
        downstream_guide_cost_nodes[node] = set([node])

        descendants = networkx.descendants(guide_trace._graph, node)

        for desc in descendants:
            dims_to_keep = n_compatible_indices(node, desc)
            desc_mvt = MultiViewTensor(model_trace.nodes[desc]['batch_log_pdf'] -
                                       guide_trace.nodes[desc]['batch_log_pdf'])
            summed_desc = desc_mvt.sum_leftmost_all_but(dims_to_keep)
            downstream_costs[node].add(summed_desc)
            downstream_guide_cost_nodes[node].update([desc])

    for site in non_reparam_nodes:
        children_in_model = set()
        for node in downstream_guide_cost_nodes[site]:
            children_in_model.update(model_trace.successors(node))
        children_in_model.difference_update(downstream_guide_cost_nodes[site])
        for child in children_in_model:
            assert (model_trace.nodes[child]["type"] == "sample")
            dims_to_keep = n_compatible_indices(site, child)
            summed_child = MultiViewTensor(model_trace.nodes[child]['batch_log_pdf']).sum_leftmost_all_but(dims_to_keep)
            downstream_costs[site].add(summed_child)
            downstream_guide_cost_nodes[site].update([child])

    for k in downstream_costs:
        downstream_costs[k] = downstream_costs[k].contract_as(guide_trace.nodes[k]['batch_log_pdf'])

    return downstream_costs, downstream_guide_cost_nodes


def big_model_guide(include_obs=True, include_single=False, include_inner_1=False, flip_c23=False,
                    include_triple=False, include_z1=False):
    p0 = variable(math.exp(-0.20), requires_grad=True)
    p1 = variable(math.exp(-0.33), requires_grad=True)
    p2 = variable(math.exp(-0.70), requires_grad=True)
    if include_triple:
        with pyro.iarange("iarange_triple1", 6) as ind_triple1:
            with pyro.iarange("iarange_triple2", 7) as ind_triple2:
                if include_z1:
                    pyro.sample("z1", dist.Bernoulli(p2).reshape(sample_shape=[
                                len(ind_triple2), len(ind_triple1)]))
                with pyro.iarange("iarange_triple3", 9) as ind_triple3:
                    pyro.sample("z0", dist.Bernoulli(p2).reshape(sample_shape=[len(ind_triple3),
                                len(ind_triple2), len(ind_triple1)]))
    pyro.sample("a1", dist.Bernoulli(p0))
    if include_single:
        with pyro.iarange("iarange_single", 5) as ind_single:
            b0 = pyro.sample("b0", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_single)]))
            assert b0.shape == (5,)
    with pyro.iarange("iarange_outer", 2) as ind_outer:
        pyro.sample("b1", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_outer)]))
        if include_inner_1:
            with pyro.iarange("iarange_inner_1", 3) as ind_inner:
                pyro.sample("c1", dist.Bernoulli(p1).reshape(sample_shape=[len(ind_inner), 1]))
                if flip_c23 and not include_obs:
                    pyro.sample("c3", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), 1]))
                    pyro.sample("c2", dist.Bernoulli(p1).reshape(sample_shape=[len(ind_inner), len(ind_outer)]))
                else:
                    pyro.sample("c2", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), len(ind_outer)]))
                    pyro.sample("c3", dist.Bernoulli(p2).reshape(sample_shape=[len(ind_inner), 1]))
        with pyro.iarange("iarange_inner_2", 4) as ind_inner:
            pyro.sample("d1", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), 1]))
            d2 = pyro.sample("d2", dist.Bernoulli(p2).reshape(sample_shape=[len(ind_inner), len(ind_outer)]))
            assert d2.shape == (4, 2)
            if include_obs:
                pyro.sample("obs", dist.Bernoulli(p0).reshape(sample_shape=[len(ind_inner), len(ind_outer)]),
                            obs=Variable(torch.ones(d2.size())))


@pytest.mark.parametrize("include_inner_1", [True, False])
@pytest.mark.parametrize("include_single", [True, False])
@pytest.mark.parametrize("flip_c23", [True, False])
@pytest.mark.parametrize("include_triple", [True, False])
@pytest.mark.parametrize("include_z1", [True, False])
def test_compute_downstream_costs_big_model_guide_pair(include_inner_1, include_single, flip_c23,
                                                       include_triple, include_z1):
    guide_trace = poutine.trace(big_model_guide,
                                graph_type="dense").get_trace(include_obs=False, include_inner_1=include_inner_1,
                                                              include_single=include_single, flip_c23=flip_c23,
                                                              include_triple=include_triple, include_z1=include_z1)
    model_trace = poutine.trace(poutine.replay(big_model_guide, guide_trace),
                                graph_type="dense").get_trace(include_obs=True, include_inner_1=include_inner_1,
                                                              include_single=include_single, flip_c23=flip_c23,
                                                              include_triple=include_triple, include_z1=include_z1)

    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)
    model_trace.compute_batch_log_pdf()
    guide_trace.compute_batch_log_pdf()

    guide_iarange_info = guide_trace.graph["iarange_info"]
    model_iarange_info = model_trace.graph["iarange_info"]
    guide_iarange_condition = guide_iarange_info['rao-blackwellization-condition']
    model_iarange_condition = model_iarange_info['rao-blackwellization-condition']
    do_vec_rb = guide_iarange_condition and model_iarange_condition
    guide_iarange_nodes = guide_iarange_info['nodes'] if do_vec_rb else set()
    model_iarange_nodes = model_iarange_info['nodes'] if do_vec_rb else set()
    non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)

    dc, dc_nodes = _compute_downstream_costs(model_trace, guide_trace,
                                             model_iarange_nodes, guide_iarange_nodes,
                                             non_reparam_nodes)

    dc_brute, dc_nodes_brute = _brute_force_compute_downstream_costs(model_trace, guide_trace,
                                                                     model_iarange_nodes, guide_iarange_nodes,
                                                                     non_reparam_nodes)

    assert dc_nodes == dc_nodes_brute

    expected_nodes_full_model = {'a1': {'c2', 'a1', 'd1', 'c1', 'obs', 'b1', 'd2', 'c3', 'b0'}, 'd2': {'obs', 'd2'},
                                 'd1': {'obs', 'd1', 'd2'}, 'c3': {'d2', 'obs', 'd1', 'c3'},
                                 'b0': {'b0', 'd1', 'c1', 'obs', 'b1', 'd2', 'c3', 'c2'},
                                 'b1': {'obs', 'b1', 'd1', 'd2', 'c3', 'c1', 'c2'},
                                 'c1': {'d1', 'c1', 'obs', 'd2', 'c3', 'c2'},
                                 'c2': {'obs', 'd1', 'c3', 'd2', 'c2'}}
    if not include_triple and include_inner_1 and include_single and not flip_c23:
        assert(dc_nodes == expected_nodes_full_model)

    expected_b1 = (model_trace.nodes['b1']['batch_log_pdf'] - guide_trace.nodes['b1']['batch_log_pdf'])
    expected_b1 += (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf']).sum(0)
    expected_b1 += (model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']).sum(0)
    expected_b1 += model_trace.nodes['obs']['batch_log_pdf'].sum(0, keepdim=False)
    if include_inner_1:
        expected_b1 += (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf']).sum(0)
        expected_b1 += (model_trace.nodes['c2']['batch_log_pdf'] - guide_trace.nodes['c2']['batch_log_pdf']).sum(0)
        expected_b1 += (model_trace.nodes['c3']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf']).sum(0)

    if include_single:
        expected_b0 = (model_trace.nodes['b0']['batch_log_pdf'] - guide_trace.nodes['b0']['batch_log_pdf'])
        expected_b0 += (model_trace.nodes['b1']['batch_log_pdf'] - guide_trace.nodes['b1']['batch_log_pdf']).sum()
        expected_b0 += (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf']).sum()
        expected_b0 += (model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']).sum()
        expected_b0 += model_trace.nodes['obs']['batch_log_pdf'].sum()
        if include_inner_1:
            expected_b0 += (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf']).sum()
            expected_b0 += (model_trace.nodes['c2']['batch_log_pdf'] - guide_trace.nodes['c2']['batch_log_pdf']).sum()
            expected_b0 += (model_trace.nodes['c3']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf']).sum()

    if include_inner_1:
        expected_c3 = (model_trace.nodes['c3']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf'])
        expected_c3 += (model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']).sum()
        expected_c3 += (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf']).sum()
        expected_c3 += model_trace.nodes['obs']['batch_log_pdf'].sum()

        expected_c2 = (model_trace.nodes['c2']['batch_log_pdf'] - guide_trace.nodes['c2']['batch_log_pdf'])
        expected_c2 += (model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']).sum(0)
        expected_c2 += (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf']).sum(0)
        expected_c2 += model_trace.nodes['obs']['batch_log_pdf'].sum(0, keepdim=False)

        expected_c1 = (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf'])

        if flip_c23:
            term = (model_trace.nodes['c2']['batch_log_pdf'] - guide_trace.nodes['c2']['batch_log_pdf'])
            expected_c3 += term.sum(1, keepdim=True)
            expected_c2 += model_trace.nodes['c3']['batch_log_pdf']
        else:
            expected_c2 += (model_trace.nodes['c3']['batch_log_pdf'] - guide_trace.nodes['c3']['batch_log_pdf'])
            term = (model_trace.nodes['c2']['batch_log_pdf'] - guide_trace.nodes['c2']['batch_log_pdf'])
            expected_c1 += term.sum(1, keepdim=True)
        expected_c1 += expected_c3

    expected_d1 = model_trace.nodes['d1']['batch_log_pdf'] - guide_trace.nodes['d1']['batch_log_pdf']
    term = (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf'])
    expected_d1 += term.sum(1, keepdim=True)
    expected_d1 += model_trace.nodes['obs']['batch_log_pdf'].sum(1, keepdim=True)

    expected_d2 = (model_trace.nodes['d2']['batch_log_pdf'] - guide_trace.nodes['d2']['batch_log_pdf'])
    expected_d2 += model_trace.nodes['obs']['batch_log_pdf']

    if include_triple:
        expected_z0 = dc['a1'] + model_trace.nodes['z0']['batch_log_pdf'] - guide_trace.nodes['z0']['batch_log_pdf']
        assert_equal(expected_z0, dc['z0'], prec=1.0e-6)
    if include_single:
        assert_equal(expected_b0, dc['b0'], prec=1.0e-6)
        assert dc['b0'].size() == (5,)
    if include_inner_1:
        assert_equal(expected_c1, dc['c1'], prec=1.0e-6)
        assert_equal(expected_c2, dc['c2'], prec=1.0e-6)
        assert_equal(expected_c3, dc['c3'], prec=1.0e-6)
    assert_equal(expected_d2, dc['d2'], prec=1.0e-6)
    assert_equal(expected_d1, dc['d1'], prec=1.0e-6)
    assert_equal(expected_b1, dc['b1'], prec=1.0e-6)

    assert dc['b1'].size() == (2,)
    assert dc['d2'].size() == (4, 2)

    for k in dc:
        assert(guide_trace.nodes[k]['batch_log_pdf'].size() == dc[k].size())
        assert_equal(dc[k], dc_brute[k])


def diamond_model(dim):
    p0 = variable(math.exp(-0.20), requires_grad=True)
    p1 = variable(math.exp(-0.33), requires_grad=True)
    pyro.sample("a1", dist.Bernoulli(p0))
    pyro.sample("c1", dist.Bernoulli(p1))
    for i in pyro.irange("irange", 2):
        b_i = pyro.sample("b{}".format(i), dist.Bernoulli(p0 * p1))
        assert b_i.shape == ()
    pyro.sample("obs", dist.Bernoulli(p0), obs=variable(1.0))


def diamond_guide(dim):
    p0 = variable(math.exp(-0.70), requires_grad=True)
    p1 = variable(math.exp(-0.43), requires_grad=True)
    pyro.sample("a1", dist.Bernoulli(p0))
    for i in pyro.irange("irange", dim):
        pyro.sample("b{}".format(i), dist.Bernoulli(p1))
    pyro.sample("c1", dist.Bernoulli(p0))


@pytest.mark.parametrize("dim", [2, 3, 7, 11])
def test_compute_downstream_costs_duplicates(dim):
    guide_trace = poutine.trace(diamond_guide,
                                graph_type="dense").get_trace(dim=dim)
    model_trace = poutine.trace(poutine.replay(diamond_model, guide_trace),
                                graph_type="dense").get_trace(dim=dim)

    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)
    model_trace.compute_batch_log_pdf()
    guide_trace.compute_batch_log_pdf()

    guide_iarange_info = guide_trace.graph["iarange_info"]
    model_iarange_info = model_trace.graph["iarange_info"]
    guide_iarange_condition = guide_iarange_info['rao-blackwellization-condition']
    model_iarange_condition = model_iarange_info['rao-blackwellization-condition']
    do_vec_rb = guide_iarange_condition and model_iarange_condition
    guide_iarange_nodes = guide_iarange_info['nodes'] if do_vec_rb else set()
    model_iarange_nodes = model_iarange_info['nodes'] if do_vec_rb else set()
    non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)

    dc, dc_nodes = _compute_downstream_costs(model_trace, guide_trace,
                                             model_iarange_nodes, guide_iarange_nodes,
                                             non_reparam_nodes)

    dc_brute, dc_nodes_brute = _brute_force_compute_downstream_costs(model_trace, guide_trace,
                                                                     model_iarange_nodes, guide_iarange_nodes,
                                                                     non_reparam_nodes)

    assert dc_nodes == dc_nodes_brute

    expected_a1 = (model_trace.nodes['a1']['batch_log_pdf'] - guide_trace.nodes['a1']['batch_log_pdf'])
    for d in range(dim):
        expected_a1 += model_trace.nodes['b{}'.format(d)]['batch_log_pdf']
        expected_a1 -= guide_trace.nodes['b{}'.format(d)]['batch_log_pdf']
    expected_a1 += (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf'])
    expected_a1 += model_trace.nodes['obs']['batch_log_pdf']

    expected_b1 = - guide_trace.nodes['b1']['batch_log_pdf']
    for d in range(dim):
        expected_b1 += model_trace.nodes['b{}'.format(d)]['batch_log_pdf']
    expected_b1 += (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf'])
    expected_b1 += model_trace.nodes['obs']['batch_log_pdf']

    expected_c1 = (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf'])
    for d in range(dim):
        expected_c1 += model_trace.nodes['b{}'.format(d)]['batch_log_pdf']
    expected_c1 += model_trace.nodes['obs']['batch_log_pdf']

    assert_equal(expected_a1, dc['a1'], prec=1.0e-6)
    assert_equal(expected_b1, dc['b1'], prec=1.0e-6)
    assert_equal(expected_c1, dc['c1'], prec=1.0e-6)

    for k in dc:
        assert(guide_trace.nodes[k]['batch_log_pdf'].size() == dc[k].size())
        assert_equal(dc[k], dc_brute[k])


def nested_model_guide(include_obs=True, dim1=11, dim2=7):
    p0 = variable(math.exp(-0.40 - include_obs * 0.2), requires_grad=True)
    p1 = variable(math.exp(-0.33 - include_obs * 0.1), requires_grad=True)
    pyro.sample("a1", dist.Bernoulli(p0 * p1))
    for i in pyro.irange("irange", dim1):
        pyro.sample("b{}".format(i), dist.Bernoulli(p0))
        with pyro.iarange("iarange", dim2 + i) as ind:
            c_i = pyro.sample("c{}".format(i), dist.Bernoulli(p1).reshape(sample_shape=[len(ind)]))
            assert c_i.shape == (dim2 + i,)
            if include_obs:
                obs_i = pyro.sample("obs{}".format(i), dist.Bernoulli(c_i), obs=Variable(torch.ones(c_i.size())))
                assert obs_i.shape == (dim2 + i,)


@pytest.mark.parametrize("dim1", [2, 5, 9])
def test_compute_downstream_costs_iarange_in_irange(dim1):
    guide_trace = poutine.trace(nested_model_guide,
                                graph_type="dense").get_trace(include_obs=False, dim1=dim1)
    model_trace = poutine.trace(poutine.replay(nested_model_guide, guide_trace),
                                graph_type="dense").get_trace(include_obs=True, dim1=dim1)

    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)
    model_trace.compute_batch_log_pdf()
    guide_trace.compute_batch_log_pdf()

    guide_iarange_info = guide_trace.graph["iarange_info"]
    model_iarange_info = model_trace.graph["iarange_info"]
    guide_iarange_condition = guide_iarange_info['rao-blackwellization-condition']
    model_iarange_condition = model_iarange_info['rao-blackwellization-condition']
    do_vec_rb = guide_iarange_condition and model_iarange_condition
    guide_iarange_nodes = guide_iarange_info['nodes'] if do_vec_rb else set()
    model_iarange_nodes = model_iarange_info['nodes'] if do_vec_rb else set()
    non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)

    dc, dc_nodes = _compute_downstream_costs(model_trace, guide_trace,
                                             model_iarange_nodes, guide_iarange_nodes,
                                             non_reparam_nodes)

    dc_brute, dc_nodes_brute = _brute_force_compute_downstream_costs(model_trace, guide_trace,
                                                                     model_iarange_nodes, guide_iarange_nodes,
                                                                     non_reparam_nodes)

    assert dc_nodes == dc_nodes_brute

    expected_c1 = (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf'])
    expected_c1 += model_trace.nodes['obs1']['batch_log_pdf']

    expected_b1 = (model_trace.nodes['b1']['batch_log_pdf'] - guide_trace.nodes['b1']['batch_log_pdf'])
    expected_b1 += (model_trace.nodes['c1']['batch_log_pdf'] - guide_trace.nodes['c1']['batch_log_pdf']).sum()
    expected_b1 += model_trace.nodes['obs1']['batch_log_pdf'].sum()

    expected_c0 = (model_trace.nodes['c0']['batch_log_pdf'] - guide_trace.nodes['c0']['batch_log_pdf'])
    expected_c0 += model_trace.nodes['obs0']['batch_log_pdf']

    expected_b0 = (model_trace.nodes['b0']['batch_log_pdf'] - guide_trace.nodes['b0']['batch_log_pdf'])
    expected_b0 += (model_trace.nodes['c0']['batch_log_pdf'] - guide_trace.nodes['c0']['batch_log_pdf']).sum()
    expected_b0 += model_trace.nodes['obs0']['batch_log_pdf'].sum()

    assert_equal(expected_c1, dc['c1'], prec=1.0e-6)
    assert_equal(expected_b1, dc['b1'], prec=1.0e-6)
    assert_equal(expected_c0, dc['c0'], prec=1.0e-6)
    assert_equal(expected_b0, dc['b0'], prec=1.0e-6)

    for k in dc:
        assert(guide_trace.nodes[k]['batch_log_pdf'].size() == dc[k].size())
        assert_equal(dc[k], dc_brute[k])


def nested_model_guide2(include_obs=True, dim1=3, dim2=2):
    p0 = variable(math.exp(-0.40 - include_obs * 0.2), requires_grad=True)
    p1 = variable(math.exp(-0.33 - include_obs * 0.1), requires_grad=True)
    pyro.sample("a1", dist.Bernoulli(p0 * p1))
    with pyro.iarange("iarange", dim1) as ind:
        c = pyro.sample("c", dist.Bernoulli(p1).reshape(sample_shape=[len(ind)]))
        assert c.shape == (dim1,)
        for i in pyro.irange("irange", dim2):
            b_i = pyro.sample("b{}".format(i), dist.Bernoulli(p0).reshape(sample_shape=[len(ind)]))
            assert b_i.shape == (dim1,)
            if include_obs:
                obs_i = pyro.sample("obs{}".format(i), dist.Bernoulli(b_i), obs=Variable(torch.ones(b_i.size())))
                assert obs_i.shape == (dim1,)


@pytest.mark.parametrize("dim1", [2, 5])
@pytest.mark.parametrize("dim2", [3, 4])
def test_compute_downstream_costs_irange_in_iarange(dim1, dim2):
    guide_trace = poutine.trace(nested_model_guide2,
                                graph_type="dense").get_trace(include_obs=False, dim1=dim1, dim2=dim2)
    model_trace = poutine.trace(poutine.replay(nested_model_guide2, guide_trace),
                                graph_type="dense").get_trace(include_obs=True, dim1=dim1, dim2=dim2)

    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)
    model_trace.compute_batch_log_pdf()
    guide_trace.compute_batch_log_pdf()

    guide_iarange_info = guide_trace.graph["iarange_info"]
    model_iarange_info = model_trace.graph["iarange_info"]
    guide_iarange_condition = guide_iarange_info['rao-blackwellization-condition']
    model_iarange_condition = model_iarange_info['rao-blackwellization-condition']
    do_vec_rb = guide_iarange_condition and model_iarange_condition
    guide_iarange_nodes = guide_iarange_info['nodes'] if do_vec_rb else set()
    model_iarange_nodes = model_iarange_info['nodes'] if do_vec_rb else set()
    non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)

    dc, dc_nodes = _compute_downstream_costs(model_trace, guide_trace,
                                             model_iarange_nodes, guide_iarange_nodes,
                                             non_reparam_nodes)

    dc_brute, dc_nodes_brute = _brute_force_compute_downstream_costs(model_trace, guide_trace,
                                                                     model_iarange_nodes, guide_iarange_nodes,
                                                                     non_reparam_nodes)

    assert dc_nodes == dc_nodes_brute

    for k in dc:
        assert(guide_trace.nodes[k]['batch_log_pdf'].size() == dc[k].size())
        assert_equal(dc[k], dc_brute[k])

    expected_b1 = model_trace.nodes['b1']['batch_log_pdf'] - guide_trace.nodes['b1']['batch_log_pdf']
    expected_b1 += model_trace.nodes['obs1']['batch_log_pdf']
    assert_equal(expected_b1, dc['b1'])

    expected_c = model_trace.nodes['c']['batch_log_pdf'] - guide_trace.nodes['c']['batch_log_pdf']
    for i in range(dim2):
        expected_c += model_trace.nodes['b{}'.format(i)]['batch_log_pdf'] - \
            guide_trace.nodes['b{}'.format(i)]['batch_log_pdf']
        expected_c += model_trace.nodes['obs{}'.format(i)]['batch_log_pdf']
    assert_equal(expected_c, dc['c'])

    expected_a1 = model_trace.nodes['a1']['batch_log_pdf'] - guide_trace.nodes['a1']['batch_log_pdf']
    expected_a1 += expected_c.sum()
    assert_equal(expected_a1, dc['a1'])
