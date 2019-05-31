from __future__ import absolute_import, division, print_function

import os
import re

import torch
from six.moves import urllib


def load(root_path):
    """
    Loads the boston housing dataset.

    References:

    - http://lib.stat.cmu.edu/datasets/boston
    - Harrison, D. and Rubinfeld, D.L.
      'Hedonic prices and the demand for clean air'
      J. Environ. Economics & Management, vol.5, 81-102, 1978.
    - Belsley, Kuh & Welsch, 'Regression diagnostics', Wiley, 1980.
    """
    pkl_path = os.path.join(root_path, "housing.pkl")
    if not os.path.exists(pkl_path):
        # Download.
        csv_path = os.path.join(root_path, "housing.data")
        if not os.path.exists(csv_path):
            urllib.request.urlretrieve(
               "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
               csv_path)

        # Convert to tensor.
        with open(csv_path) as f:
            text = re.sub(r" +", " ", f.read())
        table = [[float(cell) for cell in line.strip().split()]
                 for line in text.strip().split("\n")]
        tensor = torch.tensor(table)
        torch.save(tensor, pkl_path)

    data = torch.load(pkl_path)
    header = "CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV".split(",")
    return data, header
