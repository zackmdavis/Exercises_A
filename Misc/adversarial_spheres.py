import torch
from torch import nn

import numpy as np

def spherical_sample(n, r, size):  # thanks, Claude
    z = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=size)
    x = r * z / np.linalg.norm(z, axis=1)[:, np.newaxis]
    return torch.tensor(x, dtype=torch.float32)


class AdversarialSphereNetwork(nn.Module):
    def __init__(self, n, h):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n, h),
            nn.ReLU(),
            nn.Linear(h, 1)
        )

    def forward(self, x):
        return self.layers(x)


def train_basic_model(minibatch_size, steps, learning_rate=0.0001):
    training_data = sample_spherical(500, (minibatch_size, steps))
    model = AdversarialSphereNetwork(500, 1000)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for minibatch in training_data:
        output = model(minibatch)
        # TODO: need to generate target classes in conjunction with samples
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model
