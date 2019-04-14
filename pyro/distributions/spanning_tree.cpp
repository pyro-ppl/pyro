#include <cmath>

#include <torch/torch.h>

at::Tensor make_complete_graph(long num_vertices) {
  const long V = num_vertices;
  const long K = V * (V - 1) / 2;
  auto grid = torch::empty({2, K}, at::kLong);
  int k = 0;
  for (int v2 = 0; v2 != V; ++v2) {
    for (int v1 = 0; v1 != v2; ++v1) {
      grid[0][k] = v1;
      grid[1][k] = v2;
      k += 1;
    }
  }
  return grid;
}

at::Tensor sample_tree_mcmc(at::Tensor edge_logits, at::Tensor edges) {
  return edges;  // TODO implement a sampler.
}

at::Tensor sample_tree_approx(at::Tensor edge_logits) {
  torch::NoGradGuard no_grad;

  const long K = edge_logits.size(0);
  const long V = static_cast<long>(0.5 + std::sqrt(0.25 + 2 * K));
  const long E = V - 1;
  auto grid = make_complete_graph(V);
  auto components = torch::zeros({V}, at::kByte);
  auto e2k = torch::empty({E}, at::kLong);

  // Sample the first edge at random.
  auto probs = (edge_logits - edge_logits.max()).exp();
  auto k = probs.multinomial(1)[0];
  components[grid[0][k]] = 1;
  components[grid[1][k]] = 1;
  e2k[0] = k;

  // Sample edges connecting the cumulative tree to a new leaf.
  for (int e = 1; e != E; ++e) {
    auto c1 = components.index_select(0, grid[0]);
    auto c2 = components.index_select(0, grid[1]);
    auto mask = c1.__xor__(c2);
    auto valid_logits = edge_logits.masked_select(mask);
    auto probs = (valid_logits - valid_logits.max()).exp();
    auto k = mask.nonzero().view(-1)[probs.multinomial(1)[0]];
    components[grid[0][k]] = 1;
    components[grid[1][k]] = 1;
    e2k[e] = k;
  }

  e2k.sort();
  auto edges = torch::empty({E, 2}, at::kLong);
  for (int e = 0; e != E; ++e) {
    edges[e][0] = grid[0][e2k[e]];
    edges[e][1] = grid[1][e2k[e]];
  }
  return edges;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_tree_mcmc", &sample_tree_mcmc, "Sample a random spanning tree using MCMC");
  m.def("sample_tree_approx", &sample_tree_approx, "Approximate sample a random spanning tree");
  m.def("make_complete_graph", &make_complete_graph, "Constructs a complete graph");
}
