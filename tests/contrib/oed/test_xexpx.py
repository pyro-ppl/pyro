#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:52:27 2018

@author: charlinelelan
"""
import pytest
import torch

from pyro.contrib.oed.eig import xexpx


@pytest.mark.parametrize("argument,output", [
    (torch.tensor([float('-inf')]), torch.tensor([0.])),
    (torch.tensor([0.]), torch.tensor([0.])),
    (torch.tensor([1.]), torch.exp(torch.tensor([1.])))
])
def test_xexpx(argument, output):
    assert xexpx(argument) == output
