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

import numpy as np

from torch import nn
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


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
        elif isinstance(feature_probability, float):
            feature_probability = torch.tensor(feature_probability)
        self.feature_probability = feature_probability.to("cuda")

        if importance is None:
            importance = torch.ones(())
        elif isinstance(importance, float):
            torch.tensor(importance)
        self.importance = importance.to("cuda")

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
        batch_size = x.shape[0]
        return (1 / (self.feature_count * batch_size)) * torch.sum(
            self.importance * (x - out) ** 2
        )

    # skeleton code
    def optimize(
        self,
        batch_size=1024,
        steps=10_000,
        log_freq=100,
        lr=1e-3,
        lr_scale=constant_lr,
    ):
        optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.loss(out, batch)
            loss.backward()
            optimizer.step()

            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item(), lr=step_lr)


def experiment(instance_count=8, feature_count=5):
    # features vary in importance within each instance
    importance = (0.9 ** torch.arange(feature_count)).unsqueeze(0)
    # sparsity varies over instances
    feature_probability = 50 ** -torch.linspace(0, 1, instance_count)
    instances = [
        Model(importance=importance, feature_probability=p)
        for _i, p in enumerate(feature_probability)
    ]
    for instance in instances:
        instance.optimize(steps=4000)
    return instances


# The plotting code doesn't work in the terminal, which is disappointing. (I
# guess ML culture really believes in Colabs?! Gross.)


class NeuronModel(Model):
    def __init__(self, feature_probability=None, importance=None):
        super().__init__(feature_probability=feature_probability, importance=importance)

    def forward(self, x):
        h = self.relu((self.W @ x.T).T)
        return self.relu((self.W.T @ h.T).T + self.b)


# § Sparse Autoencoders in Toy Models

# We are asked, "in the formulas above (in the "Problem setup" section), what
# are the shapes of x, x', z, h, and h' ?"
#
# I'm inclined reply, x and x′ are (batch, feature_count), h and h′ are (batch,
# hidden_count), and z is (n_hidden_ae,)?
#
# Self-followup question: no batch dimension for z because it's tracking
# activations, not data? I'm not sure that makes sense.
#
# Instructor's solution says I was otherwise right, but that these all include
# batch and instance dimensions.


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dimensionality,
        latent_dimensionality,
        l1_coëfficient=0.5,
        tied=False,
        weight_normalizer_ε=1e-8,
    ):
        super().__init__()

        self.l1_coëfficient = l1_coëfficient
        self.tied = tied
        self.weight_normalizer_ε = weight_normalizer_ε

        self.encoder_weights = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(input_dimensionality, latent_dimensionality)
            )
        )
        if not tied:
            self.decoder_weights = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(latent_dimensionality, input_dimensionality)
                )
            )

        self.encoder_biases = nn.Parameter(torch.zeros(latent_dimensionality))
        self.decoder_biases = nn.Parameter(torch.zeros(input_dimensionality))

        self.to(device)

    def normalize_and_return_decoder_weights(self):
        # Docstring says "Normalization should be over the `n_input_ae`
        # dimension, i.e. each feature should have a normalized decoder
        # weight."
        #
        # "Normalizing over a dimension" means making them add up to one. I can
        # do that one column at a time with `m[:, i] /= m[:, i].sum()` but
        # presumably there's a more systematic way to do it (surely with
        # einops). GPT-4 points out that the non-one-at-a-time method is just
        # dividing by the sum-with-keepdims.
        #
        # But if the tied weights get transposed, then don't I have to
        # normalize with respect to the other dimension (or do the
        # normalization after the transpose)?
        #
        # if self.tied:
        #     normalized = self.encoder_weights / (self.encoder_weights.sum(dim=1, keepdim=True)
        #     return normalized.T
        # else:
        #     self.decoder_weights[:] = self.decoder_weights / self.decoder_weights.sum(dim=1, keepdim=True)
        #     return self.decoder_weights
        #
        # Let's go with the instructor's version. ... except the instructor's
        # version (adapted for my variable names) seems to have an error in the
        # untied branch (saying `dim=2` when there is no dim 2). Changing to
        # dim=1 resolves the error, but now summing along dim=0 doesn't give me
        # 1—which is nuts, because we are, in fact, dividing by the norm.
        #
        # `dim=2` could be because of me ditching `n_instances`
        if self.tied:
            return self.encoder_weights.transpose(-1, -2) / (
                self.encoder_weights.transpose(-1, -2).norm(dim=1, keepdim=True)
                + self.weight_normalizer_ε
            )
        else:
            self.decoder_weights.data = self.decoder_weights.data / (
                self.decoder_weights.data.norm(dim=1, keepdim=True)
                + self.weight_normalizer_ε
            )
        return self.decoder_weights
