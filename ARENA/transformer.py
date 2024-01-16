import torch
from torch import nn


@dataclass
class Configuration:
    embedding_dimensionality: int = 768  # d_model
    debug: bool = True
    layer_norm_ε: float = 0.00001
    vocabulary_size: int = 50257
    initialization_range: float = 0.02
    context_size: int = 1024
    attention_dimensionality: int = 64  # d_head
    mlp_dimensionality: int = 3072  # d_mlp
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
            torch.empty(
                configuration.vocabulary_size, configuration.embedding_dimensionality
            )
        )
        nn.init.normal_(
            self.embedding_weights, std=self.configuration.initialization_range
        )

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
            torch.empty(
                configuration.vocabulary_size, configuration.embedding_dimensionality
            )
        )
        nn.init.normal_(
            self.positional_embedding_weights,
            std=self.configuration.initialization_range,
        )

    def forward(self, tokens):
        batch_size, sequence_length = tokens.shape
        # unsqueeze inserts a dimension of size 1
        # repeat repeats along each dimension in the args
        # So we're taking the first `sequence_length` positional-embedding
        # vectors and copying them along the batch dimension, to get
        # shape (batch_size, sequence_length, embedding_dimensionality)
        return (
            self.positional_embedding_weights[:sequence_length]
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )


class Attention(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.query_weights = nn.Parameter(
            torch.empty(
                configuration.headcount,
                configuration.embedding_dimensionality,
                configuration.attention_dimensionality,
            )
        )
        self.key_weights = nn.Parameter(
            torch.empty(
                configuration.headcount,
                configuration.embedding_dimensionality,
                configuration.attention_dimensionality,
            )
        )
        self.value_weights = nn.Parameter(
            torch.empty(
                configuration.headcount,
                configuration.embedding_dimensionality,
                configuration.attention_dimensionality,
            )
        )
        self.output_weights = nn.Parameter(
            torch.empty(
                configuration.headcount,
                configuration.attention_dimensionality,
                configuration.embedding_dimensionality,
            )
        )
        self.query_biases = nn.Parameter(
            torch.zeros(configuration.headcount, configuration.attention_dimensionality)
        )
        self.key_biases = nn.Parameter(
            torch.zeros(configuration.headcount, configuration.attention_dimensionality)
        )
        self.value_biases = nn.Parameter(
            torch.zeros(configuration.headcount, configuration.attention_dimensionality)
        )
        self.output_biases = nn.Parameter(
            torch.zeros(configuration.embedding_dimensionality)
        )
        nn.init.normal_(self.query_weights, std=self.configuration.initialization_range)
        nn.init.normal_(self.key_weights, std=self.configuration.initialization_range)
        nn.init.normal_(self.value_weights, std=self.configuration.initialization_range)
        nn.init.normal_(
            self.output_weights, std=self.configuration.initialization_range
        )
        self.register_buffer(
            "IGNORE", torch.tensor(-1e5, dtype=torch.float32, device=device)
        )

    def forward(self, residual):
        # (batch, context_size, embedding_dimensionality)
        #
        # Trying to figure this out from the diagram ... I'm imagining it would be
        # `keys = residual @ self.key_weights.mT + self.key_biases`
        # But the dimensions for that don't match up correctly.
        #
        # After chatting with GPT-4, I decide to look at the instructor's
        # solution. It uses `einops.einsum`. (I've been preferring PyTorch
        # fundamentals to einops for my own obscure æsthetic reasons.)
        #
        # After a lot of back-and-forth with GPT-4 and using concrete numbers
        # in the terminal, I think it's going to be like—
        #
        # batch = residual.size(0)
        # key_product = residual @ self.key_weights.view(
        #     configuration.embedding_dimensionality,
        #     configuration.headcount * configuration.attention_dimensionality,
        # )
        # keys = key_product.view(
        #     batch,
        #     configuration.context_size,
        #     configuration.headcount,
        #     configuration.attention_dimensionality,
        # )
        # But I should scrupulously check that this gives the same result as
        # the instructor's einops magic ... it doesn't.
        #
        # OK, the einsum expression should amount to—
        # "batch seq embed_dim, headcount emed_dim attn_dim -> batch seq headcount attn_dim"
        # = \sum{emed_dim} residual * key_weights
        # But the dimensions don't match! WTF?!?
        #
        # ... I feel sad about how useful GPT-4 is being for me, despite
        # hallucinations: I'm using it legitimately as a tutor to help me
        # understand, not mindlessly outsourcing, but it implies that pre-2023
        # autodidacticism was less viable
        batch = residual.size(0)
        attention_components = {}
        for label, weights, biases in [
            ("query", self.query_weights, self.query_biases),
            ("key", self.key_weights, self.key_biases),
            ("value", self.value_weights, self.value_biases),
        ]:
            # reshape to get the head-batching into the tail dimension while we multiply
            w = weights.transpose(0, 1).reshape(
                self.configuration.embedding_dimensionality,
                self.configuration.headcount
                * self.configuration.attention_dimensionality,
            )
            # do the muliplication and re-batch the heads
            attention_components[label] = (residual @ w).reshape(
                batch,
                self.configuration.context_size,
                self.configuration.headcount,
                self.configuration.attention_dimensionality,
            ) + biases

    def apply_causal_mask(self, attention_scores):
        # (batch, headcount, query_pos, key_pos)
        pass
