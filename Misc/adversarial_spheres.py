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
        inner_data = spherical_sample(500, 1, minibatch_size // 2)
        outer_data = spherical_sample(500, r, minibatch_size // 2)
        minibatch = torch.cat((inner_data, outer_data))
        targets = torch.cat(
            (
                torch.zeros(minibatch_size // 2, dtype=torch.float32, device=device),
                torch.ones(minibatch_size // 2, dtype=torch.float32, device=device)
            )
        )
        outputs = model(minibatch)
        loss = criterion(outputs, targets.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.set_postfix(loss=loss.item())

    return model

# So, now this actually works for random inputs. (Probabilities in the e-06
# range for r=1, but 0.9992-1 for r=1.3.)

# The next step is to find on-distribution adversarial examples with PGD.

def spherical_projection(x, r):
    # underscore-suffix operations are in-place
    with torch.no_grad():
        x.div_(torch.norm(x))
        x.mul_(r)

def find_adversarial_example(model, initial_input, target_label, learning_rate=0.0001):
    model.eval()
    x = initial_input.clone()
    x.requires_grad_(True)
    r = torch.norm(initial_input)
    target = torch.tensor([target_label], dtype=torch.float32, device=device).unsqueeze(0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([x], lr=learning_rate)
    step_count = 0
    keep_going = True
    while keep_going:
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        spherical_projection(x, r)

        optimizer.zero_grad()

        step_count += 1
        if step_count % 100 == 0:
            print("{} steps, loss {}".format(step_count, loss))

        # threshold 7 semi-arbitrarily chosen (as largest integer that all
        # logits had greater magnitude than over 10 random points from each
        # class)
        if target_label:
            if output > 7:
                keep_going = False
        else:
            if output < -7:
                keep_going = False

    return x


if __name__ == "__main__":
    model = AdversarialSphereNetwork(500, 1000).to(device)
    state_dict = torch.load('sphere_weights.pth')
    model.load_state_dict(state_dict)
    initial_input = spherical_sample(500, 1, 1)
    adversarial_example = find_adversarial_example(model, initial_input, 1.)
    diff = initial_input - adversarial_example
