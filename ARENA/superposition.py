# I struggled a lot working with the skeleton code, and am annoyed that the
# canonical solution is using einops magic for what should be very
# straightforward linear algebra. What happens if I try to write a more
# from-scratch version that doesn't do this garbage thing with n-instances as a
# pseudo-batch dimension, so that my matrix multiplications work right?

# (Although at this point it's clear that bouncing off einops has been terrible
# for my development and I need to go back and do those intro exercises
# thoroughly, but whatever, stubbornly doing things my own way is learning
# practicum)

import torch
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        feature_count=5,
        hidden_count=2,
        correlated_pairs=0,
        anticorrelated_pairs=0,
        feature_probability=None,
        importance=None,
    ):
        super().__init__()

        self.feature_count = feature_count
        self.hidden_count = hidden_count
        self.correlated_pairs = correlated_pairs
        self.anticorrelated_pairs = anticorrelated_pairs

        if feature_probability is None:
            feature_probability = torch.ones(())
        else:
            feature_probability = torch.tensor(feature_probability)
        self.feature_probability = feature_probability.to("cuda")

        if importance is None:
            importance = torch.ones(())
        else:
            torch.tensor(importance)
        self.importance = importance.to("cuda").broadcast_to(feature_count)

        # canoncial skeleton code also did some broadcasting—we'll probably see about adapting that later

        self.W = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_count, feature_count))
        )
        self.b = nn.Parameter(torch.zeros(feature_count))

        self.relu = nn.ReLU()

        self.to("cuda")

    def forward(self, x):
        # sorry about all the transposes
        h = (self.W @ x.T).T  # shape (batch, hidden_count)
        return self.relu((self.W.T @ h.T).T + self.b)  # shape (batch, feature_count)

    def generate_batch(self, batch_size):
        features = torch.rand(batch_size, self.feature_count, device=self.W.device)

        feature_rolls = torch.rand(batch_size, self.feature_count, device=self.W.device)
        feature_present = feature_rolls <= self.feature_probability

        return torch.where(feature_present, features, 0.0)


    def loss(self, x, out):
        ...
