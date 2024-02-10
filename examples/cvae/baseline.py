# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class BaselineNet(nn.Module):
    def __init__(self, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        y = torch.sigmoid(self.fc3(hidden))
        return y


class MaskedBCELoss(nn.Module):
    def __init__(self, masked_with=-1):
        super().__init__()
        self.masked_with = masked_with

    def forward(self, input, target):
        target = target.view(input.shape)
        # only calculate loss on target pixels (value = -1)
        loss = F.binary_cross_entropy(
            input[target != self.masked_with],
            target[target != self.masked_with],
            reduction="none",
        )
        return loss.sum()


def train(
    device,
    dataloaders,
    dataset_sizes,
    learning_rate,
    num_epochs,
    early_stop_patience,
    model_path,
):
    # Train baseline
    baseline_net = BaselineNet(500, 500)
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    criterion = MaskedBCELoss()
    best_loss = np.inf
    early_stop_count = 0

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                baseline_net.train()
            else:
                baseline_net.eval()

            running_loss = 0.0
            num_preds = 0

            bar = tqdm(
                dataloaders[phase], desc="NN Epoch {} {}".format(epoch, phase).ljust(20)
            )
            for i, batch in enumerate(bar):
                inputs = batch["input"].to(device)
                outputs = batch["output"].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    preds = baseline_net(inputs)
                    loss = criterion(preds, outputs) / inputs.size(0)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(
                        loss="{:.2f}".format(running_loss / num_preds),
                        early_stop_count=early_stop_count,
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            # deep copy the model
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(baseline_net.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    baseline_net.load_state_dict(best_model_wts)
    baseline_net.eval()

    # Save model weights
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_net.state_dict(), model_path)

    return baseline_net
