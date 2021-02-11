# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import torch
from torch.utils.data import Dataset


alphabets = {'amino-acid': np.array(
                ['R', 'H', 'K', 'D', 'E',
                 'S', 'T', 'N', 'Q', 'C',
                 'G', 'P', 'A', 'V', 'I',
                 'L', 'M', 'F', 'Y', 'W']),
             'dna': np.array(['A', 'C', 'G', 'T'])}


class BiosequenceDataset(Dataset):
    """Load biological sequence data."""

    def __init__(self, source, source_type='list', alphabet='amino-acid'):

        # Get sequences.
        if source_type == 'list':
            seqs = source
        elif source_type == 'fasta':
            seqs = self._load_fasta(source)

        # Get lengths.
        self.L_data = torch.tensor([len(seq) for seq in seqs])
        self.max_length = int(torch.max(self.L_data))
        self.data_size = len(self.L_data)

        # Get alphabet.
        if type(alphabet) is list:
            alphabet = np.array(alphabet)
        elif alphabet in alphabets:
            alphabet = alphabets[alphabet]
        else:
            assert 'Alphabet unavailable, please provide a list of letters.'
        self.alphabet_length = len(alphabet)

        # Build dataset.
        self.seq_data = torch.cat([self._one_hot(
                seq, alphabet, self.max_length).unsqueeze(0) for seq in seqs])

    def _load_fasta(self, source):
        """A basic multiline fasta parser."""
        seqs = []
        seq = ''
        with open(source, 'r') as fr:
            for line in fr:
                if line[0] == '>':
                    if seq != '':
                        seqs.append(seq)
                        seq = ''
                else:
                    seq += line.strip('\n')
        if seq != '':
            seqs.append(seq)
        return seqs

    def _one_hot(self, seq, alphabet, length):
        """One hot encode and pad with zeros to max length."""
        # One hot encode.
        oh = torch.tensor((np.array(list(seq))[:, None] == alphabet[None, :]
                           ).astype(np.float64))
        # Pad.
        x = torch.cat([oh, torch.zeros([length - len(seq), len(alphabet)])])

        return x

    def __len__(self):

        return self.data_size

    def __getitem__(self, ind):

        return (self.seq_data[ind], self.L_data[ind])
