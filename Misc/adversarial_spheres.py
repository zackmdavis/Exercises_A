import torch
from torch import nn

import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spherical_sample(n, r, size):  # thanks, Claude
    z = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=size)
    x = r * z / np.linalg.norm(z, axis=1)[:, np.newaxis]
    return torch.tensor(x, dtype=torch.float32).to(device)


class AdversarialSphereNetwork(nn.Module):
    def __init__(self, n, h):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, 1)
        )

    def forward(self, x):
        return self.layers(x)


def train_basic_model(dimensionality, hidden_layer_size, minibatch_size, steps, learning_rate=0.0001):
    r = 1.3
    model = AdversarialSphereNetwork(dimensionality, hidden_layer_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(steps))
    for step in progress_bar:
        # XXX TODO: this should be properly randomized, and composing one
        # minibatch at a time is probably not GPU-efficient, but that's
        # probably not important for this quick experiment
        inner_data = spherical_sample(500, 1, (steps, minibatch_size // 2))
        outer_data = spherical_sample(500, r, (steps, minibatch_size // 2))
        minibatch = torch.cat((inner_data, outer_data))
        targets = torch.cat(
            (
                torch.zeros(minibatch_size // 2, dtype=torch.float32, device=device),
                torch.ones(minibatch_size // 2, dtype=torch.float32, device=device)
            )
        )
        outputs = model(minibatch)
        # "CUDA out of memory"?! WTF
        loss = criterion(outputs, targets.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.set_postfix(loss=loss.item())

    return model
