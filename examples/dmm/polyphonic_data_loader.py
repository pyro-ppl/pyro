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

from os.path import exists, join

import numpy as np
import six.moves.cPickle as pickle
import torch
import torch.nn as nn
from observations import jsb_chorales


# this function processes the raw data; in particular it unsparsifies it
def process_data(base_path, filename, T_max=160, min_note=21, note_range=88):
    output = join(base_path, filename)
    if exists(output):
        return

    print("processing raw polyphonic music data...")
    data = jsb_chorales(base_path)
    processed_dataset = {}
    for split, data_split in zip(['train', 'test', 'valid'], data):
        processed_dataset[split] = {}
        n_seqs = len(data_split)
        processed_dataset[split]['sequence_lengths'] = np.zeros((n_seqs), dtype=np.int32)
        processed_dataset[split]['sequences'] = np.zeros((n_seqs, T_max, note_range))
        for seq in range(n_seqs):
            seq_length = len(data_split[seq])
            processed_dataset[split]['sequence_lengths'][seq] = seq_length
            for t in range(seq_length):
                note_slice = np.array(list(data_split[seq][t])) - min_note
                slice_length = len(note_slice)
                if slice_length > 0:
                    processed_dataset[split]['sequences'][seq, t, note_slice] = np.ones((slice_length))
    pickle.dump(processed_dataset, open(output, "wb"))
    print("dumped processed data to %s" % output)


# this logic will be initiated upon import
base_path = './data'
process_data(base_path, "jsb_processed.pkl")


# this function takes a numpy mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1)
def reverse_sequences_numpy(mini_batch, seq_lengths):
    reversed_mini_batch = mini_batch.copy()
    for b in range(mini_batch.shape[0]):
        T = seq_lengths[b]
        reversed_mini_batch[b, 0:T, :] = mini_batch[b, (T - 1)::-1, :]
    return reversed_mini_batch


# this function takes a torch mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1)
# in contrast to `reverse_sequences_numpy`, this function plays
# nice with torch autograd
def reverse_sequences_torch(mini_batch, seq_lengths):
    reversed_mini_batch = mini_batch.new_zeros(mini_batch.size())
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = np.arange(T - 1, -1, -1)
        time_slice = torch.cuda.LongTensor(time_slice) if 'cuda' in mini_batch.data.type() \
            else torch.LongTensor(time_slice)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch


# this function takes the hidden state as output by the PyTorch rnn and
# unpacks it it; it also reverses each sequence temporally
def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences_torch(rnn_output, seq_lengths)
    return reversed_output


# this function returns a 0/1 mask that can be used to mask out a mini-batch
# composed of sequences of length `seq_lengths`
def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = np.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0:seq_lengths[b]] = np.ones(seq_lengths[b])
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
    sorted_seq_length_indices = np.argsort(seq_lengths)[::-1]
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

    # compute the length of the longest sequence in the mini-batch
    T_max = np.max(seq_lengths)
    # this is the sorted mini-batch
    mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
    # this is the sorted mini-batch in reverse temporal order
    mini_batch_reversed = reverse_sequences_numpy(mini_batch, sorted_seq_lengths)
    # get mask for mini-batch
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)

    # wrap in PyTorch Tensors, using default tensor type
    mini_batch = torch.tensor(mini_batch).type(torch.Tensor)
    mini_batch_reversed = torch.tensor(mini_batch_reversed).type(torch.Tensor)
    mini_batch_mask = torch.tensor(mini_batch_mask).type(torch.Tensor)

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
