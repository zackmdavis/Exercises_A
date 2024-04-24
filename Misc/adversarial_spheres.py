import torch
from torch import nn

import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spherical_sample(n, r, size):  # thanks, Claude
    z = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=size)
    x = r * z / np.linalg.norm(z, axis=1)[:, np.newaxis]
    return torch.tensor(x, dtype=torch.float32).to(device)


def generate_training_data(r, steps, minibatch_size):
    inner_data = spherical_sample(500, 1, (steps, minibatch_size // 2))
    outer_data = spherical_sample(500, r, (steps, minibatch_size // 2))

    # thanks, Claude, for help shuffling the labeled data (the code supplied
    # didn't quite work, but fixing it is less effort than it would have taken
    # to do it all with just the docs)

    raw_data = torch.cat((inner_data, outer_data), dim=1)

    raw_targets = torch.cat(
        (
            torch.zeros(steps, minibatch_size // 2, dtype=torch.float32, device=device),
            torch.ones(steps, minibatch_size // 2, dtype=torch.float32, device=device),
        ),
        dim=1,
    )

    shuffled_indices = torch.stack(
        [torch.randperm(minibatch_size, device=device) for _ in range(steps)]
    )

    data = torch.gather(
        raw_data, 1, shuffled_indices.unsqueeze(-1).expand(-1, -1, raw_data.size(-1)),
    )
    targets = torch.gather(raw_targets, 1, shuffled_indices)

    return data, targets


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
    data, targets = generate_training_data(1.3, steps, minibatch_size)
    model = AdversarialSphereNetwork(dimensionality, hidden_layer_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(steps))
    for _, (minibatch, targets) in zip(progress_bar, zip(data, targets)):
        outputs = model(minibatch)
        loss = criterion(outputs, targets.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.set_postfix(loss=loss.item())

    return model

# After training with 10,000 steps, my results are not very consistent at all?!
# Even if that's just not a lot of training, I would expect to see more signal
# than this.

# inner_test = spherical_sample(500, 1, 10000)
# outer_test = spherical_sample(500, 1.3, 10000)
# inner_result = model(inner_test)
# outer_result = model(outer_test)
# inner_probs = torch.sigmoid(inner_result)
# outer_probs = torch.sigmoid(outer_result)
# inner_yeses = (inner_probs > 0.5).float().sum()
# outer_yeses = (outer_probs > 0.5).float().sum()

# In [24]: inner_yeses
# Out[24]: tensor(5501., device='cuda:0')

# In [25]: outer_yeses
# Out[25]: tensor(5418., device='cuda:0')

# I think my sphere data is just wrong? I'd expect `torch.sqrt(sum(x**2 for x
# in data[0][0]))` to be 1 or 1.3, in accordance with `targets[0][0]`, but I'm
# seeing numbers in the teens ...

# But the `spherical_sample` function is behaving correctly.

# In [56]: torch.sqrt(sum(xi**2 for xi in x[0]))
# Out[56]: tensor(1.3000, device='cuda:0')

# In [57]: y = spherical_sample(500, 1, 1)

# In [58]: torch.sqrt(sum(xi**2 for xi in y[0]))
# Out[58]: tensor(1.0000, device='cuda:0')
