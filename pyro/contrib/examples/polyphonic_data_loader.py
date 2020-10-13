# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Data loader logic with two main responsibilities:
(i)  download raw data and process; this logic is initiated upon import
(ii) helper functions for dealing with mini-batches, sequence packing, etc.

Data are taken from

Boulanger-Lewandowski, N., Bengio, Y. and Vincent, P.,
"Modeling Temporal Dependencies in High-Dimensional Sequences: Application to
Polyphonic Music Generation and Transcription"

however, the original source of the data seems to be the Institut fuer Algorithmen
und Kognitive Systeme at Universitaet Karlsruhe.
"""

import os
from collections import namedtuple
from urllib.request import urlopen
import pickle

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from pyro.contrib.examples.util import get_data_directory


dset = namedtuple("dset", ["name", "url", "filename"])

JSB_CHORALES = dset("jsb_chorales",
                    "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/jsb_chorales.pickle",
                    "jsb_chorales.pkl")

PIANO_MIDI = dset("piano_midi",
                  "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/piano_midi.pickle",
                  "piano_midi.pkl")

MUSE_DATA = dset("muse_data",
                 "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/muse_data.pickle",
                 "muse_data.pkl")

NOTTINGHAM = dset("nottingham",
                  "https://d2hg8soec8ck9v.cloudfront.net/datasets/polyphonic/nottingham.pickle",
                  "nottingham.pkl")


# this function processes the raw data; in particular it unsparsifies it
def process_data(base_path, dataset, min_note=21, note_range=88):
    output = os.path.join(base_path, dataset.filename)
    if os.path.exists(output):
        try:
            with open(output, "rb") as f:
                return pickle.load(f)
        except (ValueError, UnicodeDecodeError):
            # Assume python env has changed.
            # Recreate pickle file in this env's format.
            os.remove(output)

    print("processing raw data - {} ...".format(dataset.name))
    data = pickle.load(urlopen(dataset.url))
    processed_dataset = {}
    for split, data_split in data.items():
        processed_dataset[split] = {}
        n_seqs = len(data_split)
        processed_dataset[split]['sequence_lengths'] = torch.zeros(n_seqs, dtype=torch.long)
        processed_dataset[split]['sequences'] = []
        for seq in range(n_seqs):
            seq_length = len(data_split[seq])
            processed_dataset[split]['sequence_lengths'][seq] = seq_length
            processed_sequence = torch.zeros((seq_length, note_range))
            for t in range(seq_length):
                note_slice = torch.tensor(list(data_split[seq][t])) - min_note
                slice_length = len(note_slice)
                if slice_length > 0:
                    processed_sequence[t, note_slice] = torch.ones(slice_length)
            processed_dataset[split]['sequences'].append(processed_sequence)
    pickle.dump(processed_dataset, open(output, "wb"), pickle.HIGHEST_PROTOCOL)
    print("dumped processed data to %s" % output)


# this logic will be initiated upon import
base_path = get_data_directory(__file__)
if not os.path.exists(base_path):
    os.mkdir(base_path)


# ingest training/validation/test data from disk
def load_data(dataset):
    # download and process dataset if it does not exist
    process_data(base_path, dataset)
    file_loc = os.path.join(base_path, dataset.filename)
    with open(file_loc, "rb") as f:
        dset = pickle.load(f)
        for k, v in dset.items():
            sequences = v["sequences"]
            dset[k]["sequences"] = pad_sequence(sequences, batch_first=True).type(torch.Tensor)
            dset[k]["sequence_lengths"] = v["sequence_lengths"].to(device=torch.Tensor().device)
    return dset


# this function takes a torch mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1).
def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch


# this function takes the hidden state as output by the PyTorch rnn and
# unpacks it it; it also reverses each sequence temporally
def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output


# this function returns a 0/1 mask that can be used to mask out a mini-batch
# composed of sequences of length `seq_lengths`
def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0:seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask


# this function prepares a mini-batch for training or evaluation.
# it returns a mini-batch in forward temporal order (`mini_batch`) as
# well as a mini-batch in reverse temporal order (`mini_batch_reversed`).
# it also deals with the fact that packed sequences (which are what what we
# feed to the PyTorch rnn) need to be sorted by sequence length.
def get_mini_batch(mini_batch_indices, sequences, seq_lengths, cuda=False):
    # get the sequence lengths of the mini-batch
    seq_lengths = seq_lengths[mini_batch_indices]
    # sort the sequence lengths
    _, sorted_seq_length_indices = torch.sort(seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

    # compute the length of the longest sequence in the mini-batch
    T_max = torch.max(seq_lengths)
    # this is the sorted mini-batch
    mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
    # this is the sorted mini-batch in reverse temporal order
    mini_batch_reversed = reverse_sequences(mini_batch, sorted_seq_lengths)
    # get mask for mini-batch
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)

    # cuda() here because need to cuda() before packing
    if cuda:
        mini_batch = mini_batch.cuda()
        mini_batch_mask = mini_batch_mask.cuda()
        mini_batch_reversed = mini_batch_reversed.cuda()

    # do sequence packing
    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(mini_batch_reversed,
                                                            sorted_seq_lengths,
                                                            batch_first=True)

    return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths
