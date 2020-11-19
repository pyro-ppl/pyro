# Copyright Contributors to the Pyro project.

import math
from os.path import exists
import pickle

import numpy as np
import pandas as pd
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

        log_counts_ref = X_ref.sum(-1).log()
        self.l_mean_ref = log_counts_ref.mean().item()
        self.l_scale_ref = log_counts_ref.std().item()

        log_counts_ss = X_ss.sum(-1).log()
        self.l_mean_ss = log_counts_ss.mean().item()
        self.l_scale_ss = log_counts_ss.std().item()

    def __len__(self):
        return self.num_ref_batches + self.num_ss_batches

    def _sample_batch_indices(self, include_unlabeled=True, include_labeled=True):
        if include_unlabeled and include_labeled:
            batch_order = torch.randperm(len(self)).tolist()
            ref_idx = torch.randperm(self.num_ref_data)
            ss_idx = torch.randperm(self.num_ss_data)
        elif include_labeled:
            batch_order = torch.randperm(self.num_ref_batches).tolist()
            ref_idx = torch.randperm(self.num_ref_data)
        elif include_unlabeled:
            batch_order = torch.randperm(self.num_ss_batches).tolist()
            ss_idx = torch.randperm(self.num_ss_data)

        slices = []

        if include_labeled:
            for i in range(self.num_ref_batches):
                _slice = ref_idx[i * self.batch_size: (i + 1) * self.batch_size]
                if _slice.size(0) == self.batch_size:
                    slices.append((_slice, 'ref'))

        if include_unlabeled:
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
                      nn.functional.one_hot(self.Y_ref[_slice[0]], num_classes=self.num_classes), \
                      self.l_mean_ref, self.l_scale_ref, "ref"
            elif _slice[1] == 'ss':
                yield self.X_ss[_slice[0]], self.R_ss[_slice[0]], \
                      self.l_mean_ss, self.l_scale_ss, "ss"

    def labeled_data(self):
        slices, batch_order = self._sample_batch_indices(include_unlabeled=False)

        for i in range(len(batch_order)):
            _slice = slices[batch_order[i]]
            assert _slice[1] == 'ref'
            yield self.X_ref[_slice[0]], \
                  self.Y_ref[_slice[0]], \
                  self.l_mean_ref, self.l_scale_ref, "ref"
                  #nn.functional.one_hot(self.Y_ref[_slice[0]], num_classes=self.num_classes), \

    def unlabeled_data(self):
        slices, batch_order = self._sample_batch_indices(include_labeled=False)

        for i in range(len(batch_order)):
            _slice = slices[batch_order[i]]
            assert _slice[1] == 'ss'
            yield self.X_ss[_slice[0]], self.R_ss[_slice[0]], \
                  self.l_mean_ss, self.l_scale_ss, "ss"


def get_data(mock=False, batch_size=100, data_dir="/home/mjankowi/spatial/"):
    if mock:
        print("Using mock dataset...")
        num_genes = 3641
        num_cells = 200
        X_ref = torch.distributions.Poisson(rate=10.0).sample(sample_shape=(num_cells, num_genes)).cuda()
        X_ss = torch.distributions.Poisson(rate=10.0).sample(sample_shape=(num_cells, num_genes)).cuda()
        Y_ref = torch.zeros(num_cells).long().cuda()
        R_ss = torch.randn(num_cells, 2).cuda()
        return BatchDataLoader(X_ref, Y_ref, X_ss, R_ss, batch_size, num_classes=3)

    ref = 'scRef_MIDDLE_LAYER_Subsampled4000.RDS'
    #ref = 'scRefSubsampled4000_filtered.RDS'

    if exists(data_dir + "slideseq_MOp_cropped.common_genes.h5ad") and \
       exists(data_dir + ref + '.common_genes.h5ad'):
        adata_ss = scvi.data.read_h5ad(data_dir + "slideseq_MOp_cropped.common_genes.h5ad")
        adata_ref = scvi.data.read_h5ad(data_dir + ref + '.common_genes.h5ad')
    else:
        adata_ss = scvi.data.read_h5ad(data_dir + "slideseq_MOp_cropped.h5ad")
        adata_ref = scvi.data.read_h5ad(data_dir + ref + '.h5ad')

        # reduce to common gene set
        common_genes = list(set(adata_ref.var.index.values).intersection(set(adata_ss.var.index.values)))

        adata_ss = adata_ss[:, common_genes]
        adata_ss.raw = adata_ss
        adata_ref = adata_ref[:, common_genes]
        adata_ref.raw = adata_ref

        adata_ss.write(data_dir + "slideseq_MOp_cropped.common_genes.h5ad", compression='gzip')
        adata_ref.write(data_dir + ref + '.common_genes.h5ad', compression='gzip')

    #return adata_ref, adata_ss

    # filter out non-variable genes using reference data
    #adata_filter = adata_ref.copy()
    #sc.pp.normalize_per_cell(adata_filter, counts_per_cell_after=1e4)
    #sc.pp.log1p(adata_filter)
    #sc.pp.highly_variable_genes(adata_filter, min_mean=0.0125, max_mean=3.0, min_disp=0.5)
    #highly_variable_genes = adata_filter.var["highly_variable"]
    #print("adata_ref.obs.index",adata_ref.obs.index.shape)
    #print("highly_variable_genes",highly_variable_genes.shape)
    #print("highly_variable_genes",np.sum(highly_variable_genes))
    #hvgenes = adata_ref.var.features[highly_variable_genes].tolist()

    #highly_variable_genes = set(pd.read_csv('middle_diffexp_genes.txt').values[:, 0].tolist())
    #highly_variable_genes = np.array([adata_ss.var.features[i] in highly_variable_genes for i in range(adata_ss.var.features.shape[0])])
    #print("highly_variable_genes.shape", highly_variable_genes.shape, np.sum(highly_variable_genes))

    barcodes = pd.read_csv('MIDDLE_LAYER_filtered_barcodes.txt').values[:, 0].tolist()
    #print("len(barcodes)", len(barcodes))

    #print("adata_ref.X.shape before", adata_ref.X.shape)
    #adata_ref = adata_ref[:, highly_variable_genes]
    #adata_ref.raw = adata_ref
    #print("adata_ref.X.shape after", adata_ref.X.shape)

    # convert to dense torch tensors
    X_ref = torch.from_numpy(sparse.csr_matrix.todense(adata_ref.X)).float().cuda()
    #X_ref = X_ref[:, highly_variable_genes]
    Y_ref = torch.from_numpy(adata_ref.obs["liger_ident_coarse"].values).long().cuda()

    #torch.manual_seed(0)
    #idx = torch.randperm(Y_ref.size(0))
    #train, test = idx[:-1000], idx[-1000:]
    #X_ref_train = X_ref[train]
    #Y_ref_train = Y_ref[train]
    #X_ref_test = X_ref[test]
    #Y_ref_test = Y_ref[test]

    #adata_ref = adata_ref[train.numpy(), :]
    #adata_ref = adata_ref[:, highly_variable_genes]
    #adata_ref.raw = adata_ref

    #print("adata_ss.X.shape before", adata_ss.X.shape)
    adata_ss = adata_ss[barcodes, :]
    #adata_ss = adata_ss[:, highly_variable_genes]
    adata_ss.raw = adata_ss
    #print("adata_ss.X.shape after", adata_ss.X.shape)

    X_ss = torch.from_numpy(sparse.csr_matrix.todense(adata_ss.X)).float().cuda()
    R_ss = np.stack([adata_ss.obs.x.values, adata_ss.obs.y.values], axis=-1)
    R_ss = torch.from_numpy(R_ss).float().cuda()
    #nonvariable = X_ss[:, ~highly_variable_genes]
    #X_ss = X_ss[:, highly_variable_genes]

    #torch.manual_seed(0)
    #idx = torch.randperm(X_ss.size(0))
    #train, test = idx[:-1000], idx[-1000:]
    #X_ss_train = X_ss[train]
    #R_ss_train = R_ss[train]
    ##X_ss_test = X_ss[test]
    #R_ss_test = R_ss[test]

    #adata_ss = adata_ss[train.numpy(), :]
    #adata_ss = adata_ss[:, highly_variable_genes]
    #adata_ss.raw = adata_ss

    #Xr = X_ref.sum(0).topk(50)[1].data.cpu().tolist()
    #Xs = X_ss.sum(0).topk(50)[1].data.cpu().tolist()
    #common_genes = list(set(Xr).intersection(set(Xs)))
    #print("common_genes", len(common_genes), common_genes)

    #good_ss_cells = (X_ss.sum(-1) >= 50)
    #X_ss, R_ss = X_ss[good_ss_cells], R_ss[good_ss_cells]

    #retained_ss_cells = adata_ss.obs.index[good_ss_cells.data.cpu().numpy()].tolist()
    #f = open('retained_ss_cells.pkl', 'wb')
    #pickle.dump(retained_ss_cells, f)
    #f.close()

    num_classes = np.unique(adata_ref.obs["liger_ident_coarse"].values).shape[0]
    print("num_classes", num_classes)

    print("X_ref, Y_ref", X_ref.shape, Y_ref.shape)
    print("X_ss, R_ss", X_ss.shape, R_ss.shape)
    #print("X_ss_sum", X_ss.sum(-1).mean().item(), X_ss.std(-1).mean().item())
    #counts = X_ss.sum(-1)
    #print("counts mean median min max", counts.mean(), counts.median(), counts.min(), counts.max(), counts.std())
    #return X_ref, Y_ref, X_ss, R_ss
    #print("nonvariable", nonvariable.shape)
    #counts = nonvariable.sum(-1)
    #print("nonvariable counts mean median min max", counts.mean(), counts.median(), counts.min(), counts.max(), counts.std())

    return BatchDataLoader(X_ref, Y_ref, X_ss, R_ss, batch_size, num_classes=num_classes), adata_ss, adata_ref
    #return BatchDataLoader(X_ref_train, Y_ref_train, X_ss_train, R_ss_train, batch_size, num_classes=num_classes), \
    #       BatchDataLoader(X_ref_test, Y_ref_test, X_ss_test, R_ss_test, batch_size, num_classes=num_classes), adata_ss, adata_ref
