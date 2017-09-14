from __future__ import print_function

import time

import networkx
import numpy as np
import torch
import torch.optim
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer.tracegraph_kl_qp import TraceGraph_KL_QP
from tests.common import TestCase


class GaussianChainTests(TestCase):
    # chain of normals with known covariances and latent means

    def setUp(self):
        self.mu0 = Variable(torch.Tensor([0.2]))
        self.data = []
        self.data.append(Variable(torch.Tensor([-0.1])))
        self.data.append(Variable(torch.Tensor([0.03])))
        self.data.append(Variable(torch.Tensor([0.20])))
        self.data.append(Variable(torch.Tensor([0.10])))
        self.n_data = Variable(torch.Tensor([len(self.data)]))
        self.sum_data = self.data[0] + self.data[1] + self.data[2] + self.data[3]
        self.verbose = True

    def setup_chain(self, N):
        torch.manual_seed(0)
        self.N = N  # number of latent variables in the chain
        lambdas = [1.5 * (k + 1) / N for k in range(N + 1)]
        self.lambdas = list(map(lambda x: Variable(torch.Tensor([x])), lambdas))
        self.lambda_tilde_posts = [self.lambdas[0]]
        for k in range(1, self.N):
            lambda_tilde_k = (self.lambdas[k] * self.lambda_tilde_posts[k - 1]) /\
                (self.lambdas[k] + self.lambda_tilde_posts[k - 1])
            self.lambda_tilde_posts.append(lambda_tilde_k)
        self.lambda_posts = [None]  # this is never used (just a way of shifting the indexing by 1)
        for k in range(1, self.N):
            lambda_k = self.lambdas[k] + self.lambda_tilde_posts[k - 1]
            self.lambda_posts.append(lambda_k)
        lambda_N_post = (self.n_data.expand_as(self.lambdas[N]) * self.lambdas[N]) +\
            self.lambda_tilde_posts[N - 1]
        self.lambda_posts.append(lambda_N_post)
        self.target_kappas = [None]
        self.target_kappas.extend([self.lambdas[k] / self.lambda_posts[k] for k in range(1, self.N)])
        self.target_mus = [None]
        self.target_mus.extend([self.mu0 * self.lambda_tilde_posts[k - 1] / self.lambda_posts[k]
                                for k in range(1, self.N)])
        target_mu_N = self.sum_data * self.lambdas[N] / lambda_N_post +\
            self.mu0 * self.lambda_tilde_posts[N - 1] / lambda_N_post
        self.target_mus.append(target_mu_N)
        self.which_nodes_reparam = self.setup_reparam_mask(N)

    # controls which nodes are reparameterized
    def setup_reparam_mask(self, N):
        while True:
            mask = torch.bernoulli(0.30 * torch.ones(N))
            if torch.sum(mask) < 0.40 * N and torch.sum(mask) > 0.5:
                return mask

    def test_elbo_reparameterized(self):
        for N in [3, 8, 17]:
            self.setup_chain(N)
            self.do_elbo_test(True, 7000 + N * 400, 0.0015, 0.04, difficulty=1.0)

    def test_elbo_nonreparameterized(self):
        for N in [3, 5, 7]:
            self.setup_chain(N)
            self.do_elbo_test(False, 4000 + N * 4000, 0.001, 0.05, difficulty=0.6)

    def do_elbo_test(self, reparameterized, n_steps, lr, prec, difficulty=1.0):
        if self.verbose:
            n_repa_nodes = torch.sum(self.which_nodes_reparam) if not reparameterized else self.N
            print(" - - - - - DO GAUSSIAN %d-CHAIN ELBO TEST  [reparameterized = %s; %d/%d] - - - - - " %
                  (self.N, reparameterized, n_repa_nodes, self.N))
            if self.N < 0:
                def array_to_string(y):
                    return str(map(lambda x: "%.3f" % x.data.numpy()[0], y))

                print("lambdas: " + array_to_string(self.lambdas))
                print("target_mus: " + array_to_string(self.target_mus[1:]))
                print("target_kappas: " + array_to_string(self.target_kappas[1:]))
                print("lambda_posts: " + array_to_string(self.lambda_posts[1:]))
                print("lambda_tilde_posts: " + array_to_string(self.lambda_tilde_posts))
        pyro.get_param_store().clear()

        def model(*args, **kwargs):
            next_mean = self.mu0
            for k in range(1, self.N + 1):
                latent_dist = dist.DiagNormal(next_mean, torch.pow(self.lambdas[k - 1], -0.5))
                mu_latent = pyro.sample("mu_latent_%d" % k, latent_dist)
                next_mean = mu_latent

            mu_N = next_mean
            for i, x in enumerate(self.data):
                pyro.observe("obs_%d" % i, dist.diagnormal, x, mu_N,
                             torch.pow(self.lambdas[self.N], -0.5))
            return mu_N

        def guide(*args, **kwargs):
            previous_sample = None
            for k in reversed(range(1, self.N + 1)):
                mu_q = pyro.param("mu_q_%d" % k, Variable(self.target_mus[k].data +
                                                          difficulty * (0.1 * torch.randn(1) - 0.53),
                                                          requires_grad=True))
                log_sig_q = pyro.param("log_sig_q_%d" % k,
                                       Variable(-0.5 * torch.log(self.lambda_posts[k]).data +
                                                difficulty * (0.1 * torch.randn(1) - 0.53),
                                                requires_grad=True))
                sig_q = torch.exp(log_sig_q)
                kappa_q = None if k == self.N \
                    else pyro.param("kappa_q_%d" % k,
                                    Variable(self.target_kappas[k].data +
                                             difficulty * (0.1 * torch.randn(1) - 0.53),
                                             requires_grad=True))
                mean_function = mu_q if k == self.N else kappa_q * previous_sample + mu_q
                latent_dist = dist.DiagNormal(mean_function, sig_q)
                node_flagged = True if self.which_nodes_reparam[k - 1] == 1.0 else False
                repa = True if reparameterized else node_flagged
                mu_latent = pyro.sample("mu_latent_%d" % k, latent_dist, reparameterized=repa,
                                        decaying_avg_baseline=True)
                previous_sample = mu_latent
            return previous_sample

        kl_optim = TraceGraph_KL_QP(model, guide, pyro.optim(torch.optim.Adam,
                                    {"lr": lr, "betas": (0.95, 0.999)}))

        for step in range(n_steps):
            t0 = time.time()
            kl_optim.step(Verbose=(step == 0))

            if (step % 5000 == 0 or step == n_steps - 1) and self.verbose:
                kappa_errors, log_sig_errors, mu_errors = [], [], []
                for k in range(1, self.N + 1):
                    if k != self.N:
                        kappa_q = pyro.param("kappa_q_%d" % k)
                        kappa_error = torch.sum(torch.pow(self.target_kappas[k] - kappa_q, 2.0))
                        kappa_errors.append(kappa_error.data.numpy()[0])

                    mu_q = pyro.param("mu_q_%d" % k)
                    mu_error = torch.sum(torch.pow(self.target_mus[k] - mu_q, 2.0))
                    mu_errors.append(mu_error.data.numpy()[0])

                    log_sig_q = pyro.param("log_sig_q_%d" % k)
                    target_log_sig = -0.5 * torch.log(self.lambda_posts[k])
                    log_sig_error = torch.sum(torch.pow(target_log_sig - log_sig_q, 2.0))
                    log_sig_errors.append(log_sig_error.data.numpy()[0])

                max_errors = (np.max(mu_errors), np.max(log_sig_errors), np.max(kappa_errors))
                min_errors = (np.min(mu_errors), np.min(log_sig_errors), np.min(kappa_errors))
                mean_errors = (np.mean(mu_errors), np.mean(log_sig_errors), np.mean(kappa_errors))
                print("[max errors]   (mu, log_sigma, kappa) = (%.4f, %.4f, %.4f)" % max_errors)
                print("[min errors]   (mu, log_sigma, kappa) = (%.4f, %.4f, %.4f)" % min_errors)
                print("[mean errors]  (mu, log_sigma, kappa) = (%.4f, %.4f, %.4f)" % mean_errors)
                print("[step time = %.3f;  N = %d;  step = %d]\n" % (time.time() - t0, self.N, step))

        self.assertEqual(0.0, max_errors[0], prec=prec)
        self.assertEqual(0.0, max_errors[1], prec=prec)
        self.assertEqual(0.0, max_errors[2], prec=prec)


class GaussianPyramidTests(TestCase):

    def setUp(self):
        self.mu0 = Variable(torch.Tensor([0.52]))
        self.verbose = True

    def setup_pyramid(self, N):
        # pyramid of normals with known covariances and latent means
        assert(N > 1)
        torch.manual_seed(0)
        self.N = N  # number of layers in the pyramid
        lambdas = [1.1 * (k + 1) / N for k in range(N + 2)]
        self.lambdas = list(map(lambda x: Variable(torch.Tensor([x])), lambdas))
        # generate data
        self.data = []
        self.N_data = 3
        bottom_layer_size = 2 ** (N - 1)
        for i in range(bottom_layer_size):
            data_i = []
            for k in range(self.N_data):
                data_i.append(Variable(torch.Tensor([0.25]) +
                                       (0.1 + 0.4 * (i + 1) / bottom_layer_size) * torch.randn(1)))
            self.data.append(data_i)
        self.data_sums = [sum(self.data[i]) for i in range(bottom_layer_size)]
        self.N_data = Variable(torch.Tensor([self.N_data]))
        self.q_dag = self.construct_q_dag()
        # compute the order in which guide samples are generated
        self.q_topo_sort = networkx.topological_sort(self.q_dag)
        self.which_nodes_reparam = self.setup_reparam_mask(len(self.q_topo_sort))
        self.calculate_variational_targets()
        self.set_model_permutations()

    # for choosing which latents should be reparameterized
    def setup_reparam_mask(self, n):
        while True:
            mask = torch.bernoulli(0.30 * torch.ones(n))
            if torch.sum(mask) < 0.40 * n and torch.sum(mask) > 0.5:
                return mask

    # for doing model sampling in different sequential orders
    def set_model_permutations(self):
        self.model_permutations = []
        self.model_unpermutations = []
        for n in range(1, self.N):
            permutation = list(range(2 ** (n - 1)))
            if n > 1:
                while permutation == list(range(2 ** (n - 1))):
                    permutation = torch.randperm(2 ** (n - 1)).numpy().tolist()
            self.model_permutations.append(permutation)

            unpermutation = list(range(len(permutation)))
            for i in range(len(permutation)):
                unpermutation[permutation[i]] = i
            self.model_unpermutations.append(unpermutation)

    def test_elbo_reparameterized(self):
        for N in [3, 4]:
            self.setup_pyramid(N)
            self.do_elbo_test(True, N * 7000, 0.0015, 0.04, 0.92,
                              difficulty=0.8, model_permutation=False)

    def test_elbo_nonreparameterized(self):
        for N in [2, 3]:
            self.setup_pyramid(N)
            self.do_elbo_test(False, N * 10000 + 5000, 0.001, 0.06, 0.95,
                              difficulty=0.4, model_permutation=False)

    def test_elbo_nonreparameterized_model_permuted(self):
        for N in [2, 3]:
            self.setup_pyramid(N)
            self.do_elbo_test(False, 20000 * N, 0.0007, 0.06, 0.97, difficulty=0.4, model_permutation=True)

    def calculate_variational_targets(self):
        # calculate (some of the) variational parameters corresponding to exact posterior

        def calc_lambda_A(lA, lB, lC):
            return lA + lB + lC

        def calc_lambda_B(lA, lB):
            return (lA * lB) / (lA + lB)

        def calc_lambda_C(lA, lB, lC):
            return ((lA + lB) * lC) / (lA + lB + lC)

        self.target_lambdas = {"1": self.lambdas[0]}
        previous_names = ["1"]
        for n in range(2, self.N + 1):
            new_names = []
            for prev_name in previous_names:
                for LR in ['L', 'R']:
                    new_names.append(prev_name + LR)
                    self.target_lambdas[new_names[-1]] = self.lambdas[n - 1]
            previous_names = new_names

        # recursion to compute the target precisions
        previous_names = ["1"]
        old_left_pivot_lambda = None
        for n in range(2, self.N + 1):
            new_names = []
            for prev_name in previous_names:
                BC_names = []
                for LR in ['L', 'R']:
                    new_names.append(prev_name + LR)
                    BC_names.append(new_names[-1])
                lambda_A0 = self.target_lambdas[prev_name]
                if n == self.N:
                    old_left_pivot_lambda = lambda_A0
                lambda_B0 = self.target_lambdas[BC_names[0]]
                lambda_C0 = self.target_lambdas[BC_names[1]]
                lambda_A = calc_lambda_A(lambda_A0, lambda_B0, lambda_C0)
                lambda_B = calc_lambda_B(lambda_A0, lambda_B0)
                lambda_C = calc_lambda_C(lambda_A0, lambda_B0, lambda_C0)
                self.target_lambdas[prev_name] = lambda_A
                self.target_lambdas[BC_names[0]] = lambda_B
                self.target_lambdas[BC_names[1]] = lambda_C
            previous_names = new_names

        for prev_name in previous_names:
            new_lambda = self.N_data * self.lambdas[-1] + self.target_lambdas[prev_name]
            self.target_lambdas[prev_name] = new_lambda

        leftmost_node_suffix = self.q_topo_sort[0][10:]
        leftmost_lambda = self.target_lambdas[leftmost_node_suffix]
        self.target_leftmost_constant = self.data_sums[0] * self.lambdas[-1] / leftmost_lambda
        self.target_leftmost_constant += self.mu0 * (leftmost_lambda - self.N_data * self.lambdas[-1]) /\
            leftmost_lambda

        almost_leftmost_node_suffix = leftmost_node_suffix[:-1] + 'R'
        almost_leftmost_lambda = self.target_lambdas[almost_leftmost_node_suffix]
        result = self.lambdas[-1] * self.data_sums[1]
        result += (almost_leftmost_lambda - self.N_data * self.lambdas[-1]) \
            * self.mu0 * old_left_pivot_lambda / (old_left_pivot_lambda + self.lambdas[-2])
        self.target_almost_leftmost_constant = result / almost_leftmost_lambda

    # construct dependency structure for the guide
    def construct_q_dag(self):
        g = networkx.DiGraph()

        def add_edge(s):
            deps = []
            if s == "1":
                deps.extend(["1L", "1R"])
            else:
                if s[-1] == 'R':
                    deps.append(s[0:-1] + 'L')
                if len(s) < self.N:
                    deps.extend([s + 'L', s + 'R'])
                for k in range(len(s) - 2):
                    base = s[1:-1 - k]
                    if base[-1] == 'R':
                        deps.append('1' + base[:-1] + 'L')
            for dep in deps:
                g.add_edge("mu_latent_" + dep, "mu_latent_" + s)

        previous_names = ["1"]
        add_edge("1")
        for n in range(2, self.N + 1):
            new_names = []
            for prev_name in previous_names:
                for LR in ['L', 'R']:
                    new_name = prev_name + LR
                    new_names.append(new_name)
                    add_edge(new_name)
            previous_names = new_names

        return g

    def do_elbo_test(self, reparameterized, n_steps, lr, prec, beta1,
                     difficulty=1.0, model_permutation=False):
        if self.verbose:
            n_repa_nodes = torch.sum(self.which_nodes_reparam) if not reparameterized \
                else len(self.q_topo_sort)
            print((" - - - DO GAUSSIAN %d-LAYERED PYRAMID ELBO TEST " +
                  "(with a total of %d RVs) [reparameterized=%s; %d/%d; perm=%s] - - -") %
                  (self.N, (2 ** self.N) - 1, reparameterized, n_repa_nodes,
                   len(self.q_topo_sort), model_permutation))
        pyro.get_param_store().clear()

        def model(*args, **kwargs):
            top_latent_dist = dist.DiagNormal(self.mu0, torch.pow(self.lambdas[0], -0.5))
            previous_names = ["mu_latent_1"]
            top_latent = pyro.sample(previous_names[0], top_latent_dist)
            previous_latents_and_names = list(zip([top_latent], previous_names))

            # for sampling model variables in different sequential orders
            def permute(x, n):
                if model_permutation:
                    return [x[self.model_permutations[n - 1][i]] for i in range(len(x))]
                return x

            def unpermute(x, n):
                if model_permutation:
                    return [x[self.model_unpermutations[n - 1][i]] for i in range(len(x))]
                return x

            for n in range(2, self.N + 1):
                new_latents_and_names = []
                for prev_latent, prev_name in permute(previous_latents_and_names, n - 1):
                    latent_dist = dist.DiagNormal(prev_latent, torch.pow(self.lambdas[n - 1], -0.5))
                    couple = []
                    for LR in ['L', 'R']:
                        new_name = prev_name + LR
                        mu_latent_LR = pyro.sample(new_name, latent_dist)
                        couple.append([mu_latent_LR, new_name])
                    new_latents_and_names.append(couple)
                _previous_latents_and_names = unpermute(new_latents_and_names, n - 1)
                previous_latents_and_names = []
                for x in _previous_latents_and_names:
                    previous_latents_and_names.append(x[0])
                    previous_latents_and_names.append(x[1])

            for i, data_i in enumerate(self.data):
                for k, x in enumerate(data_i):
                    pyro.observe("obs_%s_%d" % (previous_latents_and_names[i][1], k),
                                 dist.diagnormal, x, previous_latents_and_names[i][0],
                                 torch.pow(self.lambdas[-1], -0.5))
            return top_latent

        def guide(*args, **kwargs):
            latents_dict = {}

            n_nodes = len(self.q_topo_sort)
            for i, node in enumerate(self.q_topo_sort):
                deps = self.q_dag.predecessors(node)
                node_suffix = node[10:]
                log_sig_node = pyro.param("log_sig_" + node_suffix,
                                          Variable(-0.5 * torch.log(self.target_lambdas[node_suffix]).data +
                                                   difficulty * (torch.Tensor([-0.3]) -
                                                                 0.3 * (torch.randn(1) ** 2)),
                                                   requires_grad=True))
                mean_function_node = pyro.param("constant_term_" + node,
                                                Variable(self.mu0.data +
                                                         torch.Tensor([difficulty * i / n_nodes]),
                                                         requires_grad=True))
                for dep in deps:
                    kappa_dep = pyro.param("kappa_" + node_suffix + '_' + dep[10:],
                                           Variable(torch.Tensor([0.5 + difficulty * i / n_nodes]),
                                                    requires_grad=True))
                    mean_function_node = mean_function_node + kappa_dep * latents_dict[dep]
                latent_dist_node = dist.DiagNormal(mean_function_node, torch.exp(log_sig_node))
                node_flagged = True if self.which_nodes_reparam[i] == 1.0 else False
                repa = True if reparameterized else node_flagged
                latent_node = pyro.sample(node, latent_dist_node, reparameterized=repa,
                                          decaying_avg_baseline=True, baseline_beta=0.96)
                latents_dict[node] = latent_node

            return latents_dict['mu_latent_1']

        kl_optim = TraceGraph_KL_QP(model, guide, pyro.optim(torch.optim.Adam,
                                    {"lr": lr, "betas": (beta1, 0.999)}))

        for step in range(n_steps):
            t0 = time.time()
            kl_optim.step(Verbose=(step == 0))

            if (step % 5000 == 0 or step == n_steps - 1) and self.verbose:
                log_sig_errors = []
                for node in self.target_lambdas:
                    target_log_sig = -0.5 * torch.log(self.target_lambdas[node])
                    log_sig_node = pyro.param('log_sig_' + node)
                    log_sig_error = torch.sum(torch.pow(target_log_sig - log_sig_node, 2.0))
                    log_sig_errors.append(log_sig_error.data.numpy()[0])
                max_log_sig_error = np.max(log_sig_errors)
                min_log_sig_error = np.min(log_sig_errors)
                mean_log_sig_error = np.mean(log_sig_errors)
                leftmost_node = self.q_topo_sort[0]
                leftmost_constant = pyro.param('constant_term_' + leftmost_node)
                leftmost_constant_error = torch.sum(torch.pow(self.target_leftmost_constant -
                                                    leftmost_constant, 2.0)).data.numpy()[0]
                almost_leftmost_constant = pyro.param('constant_term_' + leftmost_node[:-1] + 'R')
                almost_leftmost_constant_error = torch.sum(torch.pow(self.target_almost_leftmost_constant -
                                                           almost_leftmost_constant, 2.0)).data.numpy()[0]

                print("[mean function constant errors (partial)]   %.4f  %.4f" %
                      (leftmost_constant_error, almost_leftmost_constant_error))
                print("[min/mean/max log(sigma) errors]   %.4f  %.4f   %.4f" % (min_log_sig_error,
                      mean_log_sig_error, max_log_sig_error))
                print("[step time = %.3f;  N = %d;  step = %d]\n" % (time.time() - t0, self.N, step))

        self.assertEqual(0.0, max_log_sig_error, prec=prec)
        self.assertEqual(0.0, leftmost_constant_error, prec=prec)
        self.assertEqual(0.0, almost_leftmost_constant_error, prec=prec)
