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
        self.R_ss = R_ss
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.num_ref_data = self.X_ref.size(0)
        self.num_ss_data = self.X_ss.size(0)

        self.num_ref_batches = math.floor(self.num_ref_data / self.batch_size)
        self.num_ss_batches = math.floor(self.num_ss_data / self.batch_size)

    def __len__(self):
        return self.num_ref_batches + self.num_ss_batches

    def _sample_batch_indices(self):
        batch_order = torch.randperm(len(self)).tolist()
        ref_idx = torch.randperm(self.num_ref_data)
        ss_idx = torch.randperm(self.num_ss_data)

        slices = []

        for i in range(self.num_ref_batches):
            _slice = ref_idx[i * self.batch_size: (i + 1) * self.batch_size]
            if _slice.size(0) == self.batch_size:
                slices.append((_slice, 'ref'))

        for i in range(self.num_ss_batches):
            _slice = ss_idx[i * self.batch_size: (i + 1) * self.batch_size]
            if _slice.size(0) == self.batch_size:
                slices.append((_slice, 'ss'))

        return slices, batch_order

    def __iter__(self):
        slices, batch_order = self._sample_batch_indices()

        for i in range(len(batch_order)):
            _slice = slices[batch_order[i]]
            if _slice[1] == 'ref':
                yield self.X_ref[_slice[0]], \
                      nn.functional.one_hot(self.Y_ref[_slice[0]], num_classes=self.num_classes), "ref"
            elif _slice[1] == 'ss':
                yield self.X_ss[_slice[0]], self.R_ss[_slice[0]], "ss"


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
