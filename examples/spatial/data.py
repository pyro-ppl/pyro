# Copyright Contributors to the Pyro project.

import math
from os.path import exists

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn

import scvi
import scanpy as sc


class BatchDataLoader(object):
    def __init__(self, X_ref, Y_ref, X_ss, R_ss, batch_size, num_classes=76):
        super().__init__()
        self.X_ref = X_ref
        self.Y_ref = Y_ref
        self.X_ss = X_ss
        self.Y_ss = Y_ss
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


def get_data(batch_size=100, data_dir="/home/mjankowi/spatial/"):
    if exists(data_dir + "slideseq_MOp_cropped.common_genes.h5ad") and \
       exists(data_dir + "scRefSubsampled4000_filtered.RDS.common_genes.h5ad"):
        adata_ss = scvi.data.read_h5ad(data_dir + "slideseq_MOp_cropped.common_genes.h5ad")
        adata_ref = scvi.data.read_h5ad(data_dir + "scRefSubsampled4000_filtered.RDS.common_genes.h5ad")
    else:
        adata_ss = scvi.data.read_h5ad(data_dir + "slideseq_MOp_cropped.h5ad")
        adata_ref = scvi.data.read_h5ad(data_dir + "scRefSubsampled4000_filtered.RDS.h5ad")

        # reduce to common gene set
        common_genes = list(set(adata_ref.var.index.values).intersection(set(adata_ss.var.index.values)))

        adata_ss = adata_ss[:, common_genes]
        adata_ss.raw = adata_ss
        adata_ref = adata_ref[:, common_genes]
        adata_ref.raw = adata_ref

        adata_ss.write(data_dir + "slideseq_MOp_cropped.common_genes.h5ad", compression='gzip')
        adata_ref.write(data_dir + "scRefSubsampled4000_filtered.RDS.common_genes.h5ad", compression='gzip')

    # filter out non-variable genes using reference data
    adata_filter = adata_ref.copy()
    sc.pp.normalize_per_cell(adata_filter, counts_per_cell_after=1e4)
    sc.pp.log1p(adata_filter)
    sc.pp.highly_variable_genes(adata_filter, min_mean=0.0125, max_mean=3.0, min_disp=0.1)
    highly_variable_genes = adata_filter.var["highly_variable"]

    # convert to dense torch tensors
    X_ref = torch.from_numpy(sparse.csr_matrix.todense(adata_ref.X[:, highly_variable_genes])).float().cuda()
    Y_ref = torch.from_numpy(adata_ref.obs["liger_ident_coarse"].values).long().cuda()

    X_ss = torch.from_numpy(sparse.csr_matrix.todense(adata_ss.X[:, highly_variable_genes])).float().cuda()
    R_ss = np.stack([adata_ss.obs.x.values, adata_ss.obs.y.values], axis=-1)
    R_ss = torch.from_numpy(R_ss).float().cuda()

    #log_counts = X.sum(-1).log()
    #l_mean, l_scale = log_counts.mean().item(), log_counts.std().item()

    return BatchDataLoader(X_ref, Y_ref, X_ss, R_ss, batch_size)
