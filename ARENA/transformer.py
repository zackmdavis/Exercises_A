import torch
from torch import nn

@dataclass
class Configuration:
    embedding_dimensionality: int = 768
    debug: bool = True
    layer_norm_ε: float = 0.00001
    vocabulary_size: int = 50257
    initialization_range: float = 0.02
    context_size: int = 1024
    attention_dimensionality: int = 64
    mlp_dimensionality: int = 3072
    headcount: int = 12
    layercount: int = 12


class LayerNorm(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.weight = nn.Parameter(torch.ones(configuration.embedding_dimensionality))
        self.bias = nn.Parameter(torch.zeros(configuration.embedding_dimensionality))

    def forward(self, residual):
        # (batch, context_size, embedding_dimensionality)
        mean = residual.mean(dim=-1, keepdim=True)
        variance = residual.var(dim=-1, keepdim=True, unbiased=False)
        x = (residual - mean) / (variance + self.configuration.layer_norm_ε).sqrt()
        return x * self.weight + self.bias


class Embedding(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.embedding_weights = nn.Parameter(
            torch.empty(configuration.vocabulary_size, configuration.embedding_dimensionality)
        )
        nn.init.normal_(self.embedding_weights, std=self.configuration.initialization_range)

    def forward(self, tokens):
        # (batch, token-sequence)
        # This is using indexing magic: you can index into a tensor with another tensor.
        #
        # For every token sequences in the batch, and for every token in the
        # sequence, index in to the corresponding embedding.
        return self.embedding_weights[tokens]


class PositionalEmbedding(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.positional_embedding_weights = nn.Parameter(
            torch.empty(configuration.vocabulary_size, configuration.embedding_dimensionality)
        )
        nn.init.normal_(self.positional_embedding_weights, std=self.configuration.initialization_range)

    def forward(self, tokens):
        batch_size, sequence_length = tokens.shape
        # unsqueeze inserts a dimension of size 1
        # repeat repeats along each dimension in the args
        # So we're taking the first `sequence_length` positional-embedding
        # vectors and copying them along the batch dimension, to get
        # shape (batch_size, sequence_length, embedding_dimensionality)
        return self.positional_embedding_weights[:sequence_length].unsqueeze(0).repeat(batch_size, 1, 1)
