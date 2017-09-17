#####################################################################################
#  download data and process;  initiated upon import
#
#  data are taken from Boulanger-Lewandowski, N., Bengio, Y. and Vincent, P.,
#  "Modeling Temporal Dependencies in High-Dimensional Sequences: Application to
#  Polyphonic Music Generation and Transcription"
#
#  however, the original source of the data seems to be the Institut fuer Algorithmen
#  und Kognitive Systeme at Universitaet Karlsruhe
#####################################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cPickle
import os.path
import urllib


def download_if_absent(saveas, url):
    if not os.path.exists(saveas):
        print("downloading polyphonic music data from %s..." % url)
        urllib.URLopener().retrieve(url, saveas)


def process_data(output="jsb_processed.pkl", rawdata="jsb_raw.pkl",
                 T_max=160, min_note=21, note_range=88):

    if os.path.exists(output):
        return

    print("processing raw polyphonic music data...")
    data = cPickle.load(open(rawdata, "rb"))
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
    cPickle.dump(processed_dataset, open(output, "wb"))
    print("dumped processed data to %s" % output)


download_if_absent("jsb_raw.pkl", "http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle")
process_data()


def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = mini_batch.copy()
    for b in range(mini_batch.shape[0]):
        T = seq_lengths[b]
        reversed_mini_batch[b, 0:T, :] = mini_batch[b, (T - 1)::-1, :]
    return reversed_mini_batch


def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    rnn_output = reverse_sequences(rnn_output.data.numpy(), seq_lengths)
    return Variable(torch.Tensor(rnn_output))


def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = np.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0:seq_lengths[b]] = np.ones(seq_lengths[b])
    return mask


def get_mini_batch(mini_batch_indices, sequences, seq_lengths):
    seq_lengths = seq_lengths[mini_batch_indices]
    sorted_seq_length_indices = np.argsort(seq_lengths)[::-1]
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]
    mini_batch = sequences[sorted_mini_batch_indices, :, :]
    mini_batch_reversed = Variable(torch.Tensor(reverse_sequences(mini_batch, sorted_seq_lengths)))
    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(mini_batch_reversed,
                                                            sorted_seq_lengths,
                                                            batch_first=True)
    mini_batch_mask = Variable(torch.Tensor(get_mini_batch_mask(mini_batch, sorted_seq_lengths)))

    return Variable(torch.Tensor(mini_batch)), mini_batch_reversed, mini_batch_mask,\
        sorted_seq_lengths
