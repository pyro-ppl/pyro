# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch.utils.data import Dataset

alphabets = {
    "amino-acid": np.array(
        [
            "R",
            "H",
            "K",
            "D",
            "E",
            "S",
            "T",
            "N",
            "Q",
            "C",
            "G",
            "P",
            "A",
            "V",
            "I",
            "L",
            "M",
            "F",
            "Y",
            "W",
        ]
    ),
    "dna": np.array(["A", "C", "G", "T"]),
}


class BiosequenceDataset(Dataset):
    """
    Load biological sequence data, either from a fasta file or a python list.

    :param source: Either the input fasta file path (str) or the input list
        of sequences (list of str).
    :param str source_type: Type of input, either 'list' or 'fasta'.
    :param str alphabet: Alphabet to use. Alphabets 'amino-acid' and 'dna' are
        preset; any other input will be interpreted as the alphabet itself,
        i.e. you can use 'ACGU' for RNA.
    :param int max_length: Total length of the one-hot representation of the
        sequences, including zero padding. Defaults to the maximum sequence
        length in the dataset.
    :param bool include_stop: Append stop symbol to the end of each sequence
        and add the stop symbol to the alphabet.
    :param torch.device device: Device on which data should be stored in
        memory.
    """

    def __init__(
        self,
        source,
        source_type="list",
        alphabet="amino-acid",
        max_length=None,
        include_stop=False,
        device=None,
    ):

        super().__init__()

        # Determine device
        if device is None:
            device = torch.tensor(0.0).device
        self.device = device

        # Get sequences.
        self.include_stop = include_stop
        if source_type == "list":
            seqs = [seq + include_stop * "*" for seq in source]
        elif source_type == "fasta":
            seqs = self._load_fasta(source)

        # Get lengths.
        self.L_data = torch.tensor([float(len(seq)) for seq in seqs], device=device)
        if max_length is None:
            self.max_length = int(torch.max(self.L_data))
        else:
            self.max_length = max_length
        self.data_size = len(self.L_data)

        # Get alphabet.
        if alphabet in alphabets:
            alphabet = alphabets[alphabet]
        else:
            alphabet = np.array(list(alphabet))
        if self.include_stop:
            alphabet = np.array(list(alphabet) + ["*"])
        self.alphabet = alphabet
        self.alphabet_length = len(alphabet)

        # Build dataset.
        self.seq_data = torch.cat(
            [self._one_hot(seq, alphabet, self.max_length).unsqueeze(0) for seq in seqs]
        )

    def _load_fasta(self, source):
        """A basic multiline fasta parser."""
        seqs = []
        seq = ""
        with open(source, "r") as fr:
            for line in fr:
                if line[0] == ">":
                    if seq != "":
                        if self.include_stop:
                            seq += "*"
                        seqs.append(seq)
                        seq = ""
                else:
                    seq += line.strip("\n")
        if seq != "":
            if self.include_stop:
                seq += "*"
            seqs.append(seq)
        return seqs

    def _one_hot(self, seq, alphabet, length):
        """One hot encode and pad with zeros to max length."""
        # One hot encode.
        oh = torch.tensor(
            (np.array(list(seq))[:, None] == alphabet[None, :]).astype(np.float64),
            device=self.device,
        )
        # Pad.
        x = torch.cat(
            [oh, torch.zeros([length - len(seq), len(alphabet)], device=self.device)]
        )

        return x

    def __len__(self):

        return self.data_size

    def __getitem__(self, ind):

        return (self.seq_data[ind], self.L_data[ind])
