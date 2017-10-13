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

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import os.path
from os.path import join, dirname
from six.moves.urllib.request import urlretrieve
import six.moves.cPickle as pickle


# this function downloads the raw data if it hasn't been already
def download_if_absent(saveas, url):

    if not os.path.exists(saveas):
        print("Couldn't find polyphonic music data at {}".format(saveas))
        print("downloading polyphonic music data from %s..." % url)
        urlretrieve(url, saveas)


# this function processes the raw data; in particular it unsparsifies it
def process_data(output="jsb_processed.pkl", rawdata="jsb_raw.pkl",
                 T_max=160, min_note=21, note_range=88):

    if os.path.exists(output):
        return

    print("processing raw polyphonic music data...")
    data = pickle.load(open(rawdata, "rb"))
    processed_dataset = {}
    for split in ['train', 'valid', 'test']:
        processed_dataset[split] = {}
        data_split = data[split]
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
base_loc = dirname(__file__)
raw_file = join(base_loc, "jsb_raw.pkl")
out_file = join(base_loc, "jsb_processed.pkl")
download_if_absent(raw_file, "http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle")
process_data(output=out_file, rawdata=raw_file)


# this function takes a mini-batch and reverses each sequence
# (w.r.t the temporal axis, i.e. axis=1)
def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = mini_batch.copy()
    for b in range(mini_batch.shape[0]):
        T = seq_lengths[b]
        reversed_mini_batch[b, 0:T, :] = mini_batch[b, (T - 1)::-1, :]
    return reversed_mini_batch


# this function takes the hidden state as output by the pytorch rnn and
# unpacks it it; it also reverses each sequence temporally
def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    if 'cuda' in rnn_output.data.type():
        rev_output = reverse_sequences(rnn_output.cpu().data.numpy(), seq_lengths)
    else:
        rev_output = reverse_sequences(rnn_output.data.numpy(), seq_lengths)

    return Variable(torch.Tensor(rev_output).type_as(rnn_output.data))


# this function returns a 0/1 mask that can be used to mask out a mini-batch
# composed of sequences of length `seq_lengths`
def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = np.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0:seq_lengths[b]] = np.ones(seq_lengths[b])
    return mask


# this function prepares a mini-batch for training or evaluation
# it returns a mini-batch in forward temporal order (`mini_batch`) as
# as a mini-batch in reverse temporal order (`mini_batch_reversed`).
# it also deals with the fact that packed sequences (which are what what we
# feed to the pytorch rnn) need to be sorted by sequence length.
def get_mini_batch(mini_batch_indices, sequences, seq_lengths, volatile=False, cuda=False):
    seq_lengths = seq_lengths[mini_batch_indices]
    sorted_seq_length_indices = np.argsort(seq_lengths)[::-1]
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]
    mini_batch = sequences[sorted_mini_batch_indices, :, :]
    mini_batch_reversed = Variable(torch.Tensor(reverse_sequences(mini_batch, sorted_seq_lengths)),
                                   volatile=volatile)

    # need to cuda before it's packed
    if cuda:
        mini_batch_reversed = mini_batch_reversed.cuda()

    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(mini_batch_reversed,
                                                            sorted_seq_lengths,
                                                            batch_first=True)
    mini_batch_mask = Variable(torch.Tensor(get_mini_batch_mask(mini_batch, sorted_seq_lengths)),
                               volatile=volatile)
    if cuda:
        return Variable(torch.Tensor(mini_batch), volatile=volatile).cuda(), mini_batch_reversed, \
            mini_batch_mask.cuda(), sorted_seq_lengths
    else:
        return Variable(torch.Tensor(mini_batch), volatile=volatile), mini_batch_reversed, \
            mini_batch_mask, sorted_seq_lengths
