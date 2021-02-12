# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.contrib.mue.dataloaders import alphabets, BiosequenceDataset


@pytest.mark.parametrize('source_type', ['list', 'fasta'])
@pytest.mark.parametrize('alphabet', ['amino-acid', 'dna', ['A', 'T', 'C']])
def test_biosequencedataset(source_type, alphabet):

    # Define dataset.
    seqs = ['AATC', 'CA', 'T']

    # Encode dataset, alternate approach.
    if type(alphabet) is list:
        alphabet_list = alphabet
    elif alphabet in alphabets:
        alphabet_list = list(alphabets[alphabet])
    L_data_check = [len(seq) for seq in seqs]
    max_length_check = max(L_data_check)
    data_size_check = len(seqs)
    seq_data_check = torch.zeros([len(seqs), max_length_check,
                                  len(alphabet_list)])
    for i in range(len(seqs)):
        for j, s in enumerate(seqs[i]):
            seq_data_check[i, j, list(alphabet_list).index(s)] = 1

    # Setup data source.
    if source_type == 'fasta':
        # Save as external file.
        source = 'test_seqs.fasta'
        with open(source, 'w') as fw:
            text = """>one
AAT
C
>two
CA
>three
T
"""
            fw.write(text)
    elif source_type == 'list':
        source = seqs

    # Load dataset.
    dataset = BiosequenceDataset(source, source_type, alphabet)

    # Check.
    assert torch.allclose(dataset.L_data,
                          torch.tensor(L_data_check, dtype=torch.float64))
    assert dataset.max_length == max_length_check
    assert len(dataset) == data_size_check
    assert dataset.data_size == data_size_check
    assert dataset.alphabet_length == len(alphabet_list)
    assert torch.allclose(dataset.seq_data, seq_data_check)
    ind = torch.tensor([0, 2])
    assert torch.allclose(dataset[ind][0],
                          torch.cat([seq_data_check[0, None, :, :],
                                     seq_data_check[2, None, :, :]]))
    assert torch.allclose(dataset[ind][1], torch.tensor([4., 1.]))
