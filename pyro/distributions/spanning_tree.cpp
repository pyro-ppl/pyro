// Copyright Uber Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <vector>
#include <unordered_set>
#include <algorithm>

#include <torch/extension.h>

at::Tensor make_complete_graph(int num_vertices) {
  const int V = num_vertices;
  const int K = V * (V - 1) / 2;
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

int _remove_edge(at::Tensor grid, at::Tensor edge_ids,
                 std::vector<std::unordered_set<int>> &neighbors,
                 std::vector<bool> &components, int e) {
  int k = edge_ids[e].item().to<int>();
  int v1 = grid[0][k].item().to<int>();
  int v2 = grid[1][k].item().to<int>();
  neighbors[v1].erase(v2);
  neighbors[v2].erase(v1);
  components[v1] = 1;
  std::vector<int> pending = {v1};
  while (!pending.empty()) {
    int v1 = pending.back();
    pending.pop_back();
    for (int v2 : neighbors[v1]) {
      if (!components[v2]) {
        components[v2] = 1;
        pending.push_back(v2);
      }
    }
  }
  return k;
}

void _add_edge(at::Tensor grid, at::Tensor edge_ids,
               std::vector<std::unordered_set<int>> &neighbors,
               std::vector<bool> &components, int e, int k) {
  edge_ids[e] = k;
  int v1 = grid[0][k].item().to<int>();
  int v2 = grid[1][k].item().to<int>();
  neighbors[v1].insert(v2);
  neighbors[v2].insert(v1);
  std::fill(components.begin(), components.end(), 0);
}

int _find_valid_edges(const std::vector<bool> &components, at::Tensor valid_edge_ids) {
  int k = 0;
  int end = 0;
  const int V = components.size();
  for (int v2 = 0; v2 != V; ++v2) {
    bool c2 = components[v2];
    for (int v1 = 0; v1 != v2; ++v1) {
      if (c2 ^ components[v1]) {
        valid_edge_ids[end] = k;
        end += 1;
      }
      k += 1;
    }
  }
  return end;
}

at::Tensor sample_tree_mcmc(at::Tensor edge_logits, at::Tensor edges) {
  torch::NoGradGuard no_grad;
  if (edges.size(0) <= 1) {
    return edges;
  }

  const int E = edges.size(0);
  const int V = E + 1;
  const int K = V * (V - 1) / 2;
  auto grid = make_complete_graph(V);

  // Each of E edges in the tree is stored as an id k in [0, K) indexing into
  // the complete graph. The id of an edge (v1,v2) is k = v1+v2*(v2-1)/2.
  auto edge_ids = torch::empty({E}, at::kLong);
  // This maps each vertex to the set of its neighboring vertices.
  std::vector<std::unordered_set<int>> neighbors(V);
  // This maps each vertex to its connected component id (0 or 1).
  std::vector<bool> components(V);
  for (int e = 0; e != E; ++e) {
    int v1 = edges[e][0].item().to<int>();
    int v2 = edges[e][1].item().to<int>();
    edge_ids[e] = v1 + v2 * (v2 - 1) / 2;
    neighbors[v1].insert(v2);
    neighbors[v2].insert(v1);
  }
  // This stores ids of edges that are valid candidates for Gibbs moves.
  auto valid_edges_buffer = torch::empty({K}, at::kLong);

  // Cycle through all edges in a random order.
  auto order = torch::randperm(E);
  for (int i = 0; i != E; ++i) {
     int e = order[i].item().to<int>();

     // Perform a single-site Gibbs update by moving this edge elsewhere.
     int k = _remove_edge(grid, edge_ids, neighbors, components, e);
     int num_valid_edges = _find_valid_edges(components, valid_edges_buffer);
     auto valid_edge_ids = valid_edges_buffer.slice(0, 0, num_valid_edges);
     auto valid_logits = edge_logits.index_select(0, valid_edge_ids);
     auto valid_probs = (valid_logits - valid_logits.max()).exp();
     double total_prob = valid_probs.sum().item().to<double>();
     if (total_prob > 0) {
       int sample = valid_probs.multinomial(1)[0].item().to<int>();
       k = valid_edge_ids[sample].item().to<int>();
     }
     _add_edge(grid, edge_ids, neighbors, components, e, k);
  }

  // Convert edge ids to a canonical list of pairs.
  edge_ids = std::get<0>(edge_ids.sort());
  edges = torch::empty({E, 2}, at::kLong);
  for (int e = 0; e != E; ++e) {
    edges[e][0] = grid[0][edge_ids[e]];
    edges[e][1] = grid[1][edge_ids[e]];
  }
  return edges;
}

at::Tensor sample_tree_approx(at::Tensor edge_logits) {
  torch::NoGradGuard no_grad;
  const int K = edge_logits.size(0);
  const int V = static_cast<int>(0.5 + std::sqrt(0.25 + 2 * K));
  const int E = V - 1;
  auto grid = make_complete_graph(V);

  // Each of E edges in the tree is stored as an id k in [0, K) indexing into
  // the complete graph. The id of an edge (v1,v2) is k = v1+v2*(v2-1)/2.
  auto edge_ids = torch::empty({E}, at::kLong);
  // This maps each vertex to whether it is a member of the cumulative tree.
  auto components = torch::zeros({V}, at::kByte);

  // Sample the first edge at random.
  auto probs = (edge_logits - edge_logits.max()).exp();
  int k = probs.multinomial(1)[0].item().to<int>();
  components[grid[0][k]] = 1;
  components[grid[1][k]] = 1;
  edge_ids[0] = k;

  // Sample edges connecting the cumulative tree to a new leaf.
  for (int e = 1; e != E; ++e) {
    auto c1 = components.index_select(0, grid[0]);
    auto c2 = components.index_select(0, grid[1]);
    auto mask = c1.__xor__(c2);
    auto valid_logits = edge_logits.masked_select(mask);
    auto probs = (valid_logits - valid_logits.max()).exp();
    int sample = probs.multinomial(1)[0].item().to<int>();
    int k = mask.nonzero().view(-1)[sample].item().to<int>();
    components[grid[0][k]] = 1;
    components[grid[1][k]] = 1;
    edge_ids[e] = k;
  }

  // Convert edge ids to a canonical list of pairs.
  edge_ids = std::get<0>(edge_ids.sort());
  auto edges = torch::empty({E, 2}, at::kLong);
  for (int e = 0; e != E; ++e) {
    edges[e][0] = grid[0][edge_ids[e]];
    edges[e][1] = grid[1][edge_ids[e]];
  }
  return edges;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_tree_mcmc", &sample_tree_mcmc, "Sample a random spanning tree using MCMC");
  m.def("sample_tree_approx", &sample_tree_approx, "Approximate sample a random spanning tree");
  m.def("make_complete_graph", &make_complete_graph, "Constructs a complete graph");
}
