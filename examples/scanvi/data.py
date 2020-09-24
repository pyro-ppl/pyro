import math
import numpy as np
from scipy import sparse

import scanpy as sc
import torch
import torch.nn as nn
import scvi


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
            _slice = unlabeled_idx[i * self.batch_size: (i + 1) * self.batch_size]
            slices.append((_slice, False))

        for i in range(self.num_labeled_batches):
            _slice = labeled_idx[i * self.batch_size: (i + 1) * self.batch_size]
            slices.append((_slice, True))

        return slices, batch_order

    def __iter__(self):
        slices, batch_order = self._sample_batch_indices()

        for i in range(len(batch_order)):
            _slice = slices[batch_order[i]]
            if _slice[1]:
                # labeled
                yield self.data_x[_slice[0]], \
                      nn.functional.one_hot(self.data_y[_slice[0]], num_classes=self.num_classes)
            else:
                # unlabeled
                yield self.data_x[_slice[0]], None


def _get_score(normalized_adata, gene_set):
    """
    Returns the score per cell given a dictionary of + and - genes
    """
    score = np.zeros(normalized_adata.n_obs)
    for gene in gene_set['positive']:
        expression = np.array(normalized_adata[:, gene].X)
        score += expression.flatten()
    for gene in gene_set['negative']:
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


def get_data_loader(batch_size=100, cuda=False):
    """
    Does the necessary preprocessing and returns a BatchDataLoader for the PBMC dataset.
    """
    adata = scvi.data.purified_pbmc_dataset(subset_datasets=["regulatory_t", "naive_t",
                                                             "memory_t", "naive_cytotoxic"])

    gene_subset = ["CD4", "FOXP3", "TNFRSF18", "IL2RA", "CTLA4", "CD44", "TCF7",
                   "CD8B", "CCR7", "CD69", "PTPRC", "S100A4"]

    normalized = adata.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)

    normalized = normalized[:, gene_subset].copy()
    sc.pp.scale(normalized)

    # hand curated list of genes for identifying ground truth
    cd4_reg_geneset = {"positive": ["TNFRSF18", "CTLA4", "FOXP3", "IL2RA"], "negative": ["S100A4", "PTPRC", "CD8B"]}
    cd8_naive_geneset = {"positive": ["CD8B", "CCR7"], "negative": ["CD4"]}
    cd4_naive_geneset = {"positive": ["CCR7", "CD4"], "negative": ["S100A4", "PTPRC", "FOXP3", "IL2RA", "CD69"]}
    cd4_mem_geneset = {"positive": ["S100A4"], "negative": ["IL2RA", "FOXP3", "TNFRSF18", "CCR7"]}

    cd4_reg_mask = _get_cell_mask(normalized, cd4_reg_geneset)
    cd8_naive_mask = _get_cell_mask(normalized, cd8_naive_geneset)
    cd4_naive_mask = _get_cell_mask(normalized, cd4_naive_geneset)
    cd4_mem_mask = _get_cell_mask(normalized, cd4_mem_geneset)

    seed_labels = -np.ones(cd4_mem_mask.shape[0])
    seed_labels[cd8_naive_mask] = 0  # "CD8 Naive T cell"
    seed_labels[cd4_naive_mask] = 1  # "CD4 Naive T cell"
    seed_labels[cd4_mem_mask] = 2    # "CD4 Memory T cell"
    seed_labels[cd4_reg_mask] = 3    # "CD4 Regulatory T cell"

    Y = torch.from_numpy(seed_labels).long()
    X = torch.from_numpy(sparse.csr_matrix.todense(adata.X)).float()

    if cuda:
        X, Y = X.cuda(), Y.cuda()

    # subsample to a subset of high frequency genes using an arbitrary cutoff
    high_freq_genes = X.sum(0) > 2000
    X = X[:, high_freq_genes]

    # subsample and remove ~90% of the unlabeled cells
    labeled = torch.where(Y != -1)[0]
    unlabeled = torch.where(Y == -1)[0]
    unlabeled = unlabeled[torch.randperm(unlabeled.size(0))[:4800]]
    idx = torch.cat([labeled, unlabeled])

    return BatchDataLoader(X[idx], Y[idx], batch_size)
