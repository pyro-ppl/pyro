# Copyright Contributors to the Pyro project.
# Copyright (c) 2020, YosefLab.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""
The data preprocessing code in this script is adapted from:
https://github.com/YosefLab/scvi-tutorials/blob/50dd3269abfe0c375ec47114f2c20725a016736f/seed_labeling.ipynb
"""

import math

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse


class BatchDataLoader(object):
    """
    This custom DataLoader serves mini-batches that are either fully-observed (i.e. labeled)
    or partially-observed (i.e. unlabeled) but never mixed.
    """

    def __init__(self, data_x, data_y, batch_size, num_classes=4, missing_label=-1):
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.unlabeled = torch.where(data_y == missing_label)[0]
        self.num_unlabeled = self.unlabeled.size(0)
        self.num_unlabeled_batches = math.ceil(self.num_unlabeled / self.batch_size)

        self.labeled = torch.where(data_y != missing_label)[0]
        self.num_labeled = self.labeled.size(0)
        self.num_labeled_batches = math.ceil(self.num_labeled / self.batch_size)

        assert self.data_x.size(0) == self.data_y.size(0)
        assert len(self) > 0

    @property
    def size(self):
        return self.data_x.size(0)

    def __len__(self):
        return self.num_unlabeled_batches + self.num_labeled_batches

    def _sample_batch_indices(self):
        batch_order = torch.randperm(len(self)).tolist()
        unlabeled_idx = self.unlabeled[torch.randperm(self.num_unlabeled)]
        labeled_idx = self.labeled[torch.randperm(self.num_labeled)]

        slices = []

        for i in range(self.num_unlabeled_batches):
            _slice = unlabeled_idx[i * self.batch_size : (i + 1) * self.batch_size]
            slices.append((_slice, False))

        for i in range(self.num_labeled_batches):
            _slice = labeled_idx[i * self.batch_size : (i + 1) * self.batch_size]
            slices.append((_slice, True))

        return slices, batch_order

    def __iter__(self):
        slices, batch_order = self._sample_batch_indices()

        for i in range(len(batch_order)):
            _slice = slices[batch_order[i]]
            if _slice[1]:
                # labeled
                yield self.data_x[_slice[0]], nn.functional.one_hot(
                    self.data_y[_slice[0]], num_classes=self.num_classes
                )
            else:
                # unlabeled
                yield self.data_x[_slice[0]], None


def _get_score(normalized_adata, gene_set):
    """
    Returns the score per cell given a dictionary of + and - genes
    """
    score = np.zeros(normalized_adata.n_obs)
    for gene in gene_set["positive"]:
        expression = np.array(normalized_adata[:, gene].X)
        score += expression.flatten()
    for gene in gene_set["negative"]:
        expression = np.array(normalized_adata[:, gene].X)
        score -= expression.flatten()
    return score


def _get_cell_mask(normalized_adata, gene_set):
    """
    Calculates the score per cell for a list of genes, then returns a mask for
    the cells with the highest 50 scores.
    """
    score = _get_score(normalized_adata, gene_set)
    cell_idx = score.argsort()[-50:]
    mask = np.zeros(normalized_adata.n_obs)
    mask[cell_idx] = 1
    return mask.astype(bool)


def get_data(dataset="pbmc", batch_size=100, cuda=False):
    """
    Does the necessary preprocessing and returns a BatchDataLoader for the PBMC dataset.
    """
    assert dataset in ["pbmc", "mock"]

    # create mock dataset for CI
    if dataset == "mock":
        num_genes = 17
        num_data = 200
        X = torch.distributions.Poisson(rate=10.0).sample(
            sample_shape=(num_data, num_genes)
        )
        Y = torch.zeros(num_data, dtype=torch.long)
        Y[50:100] = 1
        Y[100:] = -1

        if cuda:
            X, Y = X.cuda(), Y.cuda()

        return BatchDataLoader(X, Y, batch_size), num_genes, 2.0, 1.0, None

    import scanpy as sc
    import scvi

    adata = scvi.data.purified_pbmc_dataset(
        subset_datasets=["regulatory_t", "naive_t", "memory_t", "naive_cytotoxic"]
    )

    gene_subset = [
        "CD4",
        "FOXP3",
        "TNFRSF18",
        "IL2RA",
        "CTLA4",
        "CD44",
        "TCF7",
        "CD8B",
        "CCR7",
        "CD69",
        "PTPRC",
        "S100A4",
    ]

    normalized = adata.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)

    normalized = normalized[:, gene_subset].copy()
    sc.pp.scale(normalized)

    # hand curated list of genes for identifying ground truth
    cd4_reg_geneset = {
        "positive": ["TNFRSF18", "CTLA4", "FOXP3", "IL2RA"],
        "negative": ["S100A4", "PTPRC", "CD8B"],
    }
    cd8_naive_geneset = {"positive": ["CD8B", "CCR7"], "negative": ["CD4"]}
    cd4_naive_geneset = {
        "positive": ["CCR7", "CD4"],
        "negative": ["S100A4", "PTPRC", "FOXP3", "IL2RA", "CD69"],
    }
    cd4_mem_geneset = {
        "positive": ["S100A4"],
        "negative": ["IL2RA", "FOXP3", "TNFRSF18", "CCR7"],
    }

    cd4_reg_mask = _get_cell_mask(normalized, cd4_reg_geneset)
    cd8_naive_mask = _get_cell_mask(normalized, cd8_naive_geneset)
    cd4_naive_mask = _get_cell_mask(normalized, cd4_naive_geneset)
    cd4_mem_mask = _get_cell_mask(normalized, cd4_mem_geneset)

    # these will be our seed labels
    seed_labels = -np.ones(cd4_mem_mask.shape[0])
    seed_labels[cd8_naive_mask] = 0  # "CD8 Naive T cell"
    seed_labels[cd4_naive_mask] = 1  # "CD4 Naive T cell"
    seed_labels[cd4_mem_mask] = 2  # "CD4 Memory T cell"
    seed_labels[cd4_reg_mask] = 3  # "CD4 Regulatory T cell"

    # this metadata will be used for plotting
    seed_colors = ["lightgray"] * seed_labels.shape[0]
    seed_sizes = [0.05] * seed_labels.shape[0]
    for i in range(len(seed_colors)):
        if seed_labels[i] == 0:
            seed_colors[i] = "lightcoral"
        elif seed_labels[i] == 1:
            seed_colors[i] = "limegreen"
        elif seed_labels[i] == 2:
            seed_colors[i] = "deepskyblue"
        elif seed_labels[i] == 3:
            seed_colors[i] = "mediumorchid"
        if seed_labels[i] != -1:
            seed_sizes[i] = 25

    adata.obs["seed_labels"] = seed_labels
    adata.obs["seed_colors"] = seed_colors
    adata.obs["seed_marker_sizes"] = seed_sizes

    Y = torch.from_numpy(seed_labels).long()
    X = torch.from_numpy(sparse.csr_matrix.todense(adata.X)).float()

    # the prior mean and scale for the log count latent variable `l`
    # is set using the empirical mean and variance of the observed log counts
    log_counts = X.sum(-1).log()
    l_mean, l_scale = log_counts.mean().item(), log_counts.std().item()

    if cuda:
        X, Y = X.cuda(), Y.cuda()

    # subsample and remove ~50% of the unlabeled cells
    torch.manual_seed(0)
    labeled = torch.where(Y != -1)[0]
    unlabeled = torch.where(Y == -1)[0]
    unlabeled = unlabeled[torch.randperm(unlabeled.size(0))[:19800]]
    idx = torch.cat([labeled, unlabeled])

    num_genes = X.size(-1)

    adata = adata[idx.data.cpu().numpy()]
    adata.raw = adata

    return (
        BatchDataLoader(X[idx], Y[idx], batch_size),
        num_genes,
        l_mean,
        l_scale,
        adata,
    )
