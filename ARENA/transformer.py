from dataclasses import dataclass

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
                configuration.context_size, configuration.embedding_dimensionality
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
        sequence_length = residual.size(1)
        attention_components = {}
        for label, weights, biases in [
            ('query', self.query_weights, self.query_biases),
            ('key', self.key_weights, self.key_biases),
            ('value', self.value_weights, self.value_biases),
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
                sequence_length,
                self.configuration.headcount,
                self.configuration.attention_dimensionality,
            ) + biases

        # (batch, headcount, context_size (Q), embedding_dimensionality)
        q = attention_components['query'].permute(0, 2, 1, 3)
        # (batch, headcount, embedding_dimensionality, context_size (K))
        k = attention_components['key'].permute(0, 2, 3, 1)
        attention_scores = q @ k
        masked = self.apply_causal_mask(
            attention_scores / self.configuration.attention_dimensionality**0.5
        )
        attention_pattern = masked.softmax(-1)

        # I can translate the instructor's einsum expressions into permuted
        # batched matrix multiplications (that check out when I compare to the
        # einsum version in a separate script) by matching up the dimensions,
        # but blindly matching up dimensions from the instructor's solution is
        # different from understanding what's going on
        #
        # value is (batch, context_size, headcount, attention_dimensionality)
        # attention_pattern is (batch, headcount, context_size, context_size)
        # z is going to be (batch, context_size, headcount, attention_dimensionality)
        #
        # so we permute to get batch and headcount as the batch dimensions,
        # and then the matrix multiplication is (attention_dimensionality ×
        # context_size(K)) × (context_size(K) × context_size(Q)) =
        # attention_dimensionality × context_size(Q)
        z = (
            attention_components['value'].permute(0, 2, 3, 1)
            @ attention_pattern.permute(0, 1, 3, 2)
        ).permute(0, 3, 1, 2)

        # more GPT-4 assistance here 😦
        out = (
            z.reshape(batch, sequence_length, -1)
            @ self.output_weights.reshape(
                -1, self.configuration.embedding_dimensionality
            )
            + self.output_biases
        )
        return out

    def apply_causal_mask(self, attention_scores):
        # (batch, headcount, query_pos, key_pos)
        mask = torch.triu(
            torch.ones(
                attention_scores.size(-2),
                attention_scores.size(-1),
                device=attention_scores.device,
            ),
            diagonal=1,
        ).bool()
        return attention_scores.masked_fill_(mask, -100000.0)


class MultilayerPerceptron(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.weights_in = nn.Parameter(
            torch.empty(
                configuration.embedding_dimensionality, configuration.mlp_dimensionality
            )
        )
        self.weights_out = nn.Parameter(
            torch.empty(
                configuration.mlp_dimensionality, configuration.embedding_dimensionality
            )
        )
        self.bias_in = nn.Parameter(torch.zeros(configuration.mlp_dimensionality))
        self.bias_out = nn.Parameter(
            torch.zeros(configuration.embedding_dimensionality)
        )
        nn.init.normal_(self.weights_in, std=self.configuration.initialization_range)
        nn.init.normal_(self.weights_out, std=self.configuration.initialization_range)
        self.gelu = nn.GELU()

    def forward(self, residual):
        # (batch, context_size, embedding_dimensionality)
        return (
            self.gelu(residual @ self.weights_in + self.bias_in) @ self.weights_out
            + self.bias_out
        )


class TransformerBlock(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.layer_norm_1 = LayerNorm(configuration)
        self.attention = Attention(configuration)
        self.layer_norm_2 = LayerNorm(configuration)
        self.multilayer_perceptron = MultilayerPerceptron(configuration)

    def forward(self, residual):
        attended = self.attention(self.layer_norm_1(residual)) + residual
        return self.multilayer_perceptron(self.layer_norm_2(attended)) + attended


class Unembedding(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.unembedding_weights = nn.Parameter(
            torch.empty(
                configuration.embedding_dimensionality, configuration.vocabulary_size
            )
        )
        nn.init.normal_(
            self.unembedding_weights, std=self.configuration.embedding_dimensionality
        )
        self.unembedding_biases = nn.Parameter(
            torch.zeros(configuration.vocabulary_size, requires_grad=False)
        )

    def forward(self, residual):
        return residual @ self.unembedding_weights + self.unembedding_biases


class Transformer(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.embedding = Embedding(configuration)
        self.positional_embedding = PositionalEmbedding(configuration)
        self.blocks = nn.ModuleList(
            [TransformerBlock(configuration) for _ in range(configuration.layercount)]
        )
        self.layer_norm_final = LayerNorm(configuration)
        self.unembedding = Unembedding(configuration)

    def forward(self, tokens):
        residual = self.embedding(tokens) + self.positional_embedding(tokens)
        # Could this just as well have been an nn.Sequential?
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembedding(self.layer_norm_final(residual))
        return logits


# So, that's the implementation. I've been following the given solution, but I
# haven't been running the tests. The fact that I've been using my own variable
# names will probably present an obstacle to `load_state_dict` ...

# GPT-4 suggests renaming the keys in the GPT-2-small's state dict. I could
# have thought of that! (Given that I have to compile the key-name mapping
# either way, it's not any harder than `load_state_dict` itself having the
# option.)

# There are 222 keys, but a lot of them are formulaic. GPT-4 further suggests
# that just zipping up the keys should work. (I had assumed I would need to
# manually match up the names, but it is an ordered dict!)
#
# ... nah, I am going to have to do a name match, it's fine.

def construct_key_mapping():
    mapping = {
        'embed.W_E': 'embedding.embedding_weights',
        'pos_embed.W_pos': 'positional_embedding.positional_embedding_weights',
        'ln_final.w': 'layer_norm_final.weight',
        'ln_final.b': 'layer_norm_final.bias',
        'unembed.W_U': 'unembedding.unembedding_weights',
        'unembed.b_U': 'unembedding.unembedding_biases',
    }
    for i in range(12):
        mapping['blocks.{}.ln1.w'.format(i)] = 'blocks.{}.layer_norm_1.weight'.format(i)
        mapping['blocks.{}.ln1.b'.format(i)] = 'blocks.{}.layer_norm_1.bias'.format(i)
        mapping['blocks.{}.ln2.w'.format(i)] = 'blocks.{}.layer_norm_2.weight'.format(i)
        mapping['blocks.{}.ln2.b'.format(i)] = 'blocks.{}.layer_norm_2.bias'.format(i)
        mapping['blocks.{}.attn.W_Q'.format(i)] = 'blocks.{}.attention.query_weights'.format(i)
        mapping['blocks.{}.attn.W_K'.format(i)] = 'blocks.{}.attention.key_weights'.format(i)
        mapping['blocks.{}.attn.W_V'.format(i)] = 'blocks.{}.attention.value_weights'.format(i)
        mapping['blocks.{}.attn.W_O'.format(i)] = 'blocks.{}.attention.output_weights'.format(i)
        mapping['blocks.{}.attn.b_Q'.format(i)] = 'blocks.{}.attention.query_biases'.format(i)
        mapping['blocks.{}.attn.b_K'.format(i)] = 'blocks.{}.attention.key_biases'.format(i)
        mapping['blocks.{}.attn.b_V'.format(i)] = 'blocks.{}.attention.value_biases'.format(i)
        mapping['blocks.{}.attn.b_O'.format(i)] = 'blocks.{}.attention.output_biases'.format(i)
        mapping['blocks.{}.mlp.W_in'.format(i)] = 'blocks.{}.multilayer_perceptron.weights_in'.format(i)
        mapping['blocks.{}.mlp.b_in'.format(i)] = 'blocks.{}.multilayer_perceptron.bias_in'.format(i)
        mapping['blocks.{}.mlp.W_out'.format(i)] = 'blocks.{}.multilayer_perceptron.weights_out'.format(i)
        mapping['blocks.{}.mlp.b_out'.format(i)] = 'blocks.{}.multilayer_perceptron.bias_out'.format(i)
        # reference implementation seems to have a 'blocks.{}.attn.mask' that wasn't in the tutorial?
    return mapping


# Load it up—
#
# from transformer import Configuration, Transformer, construct_key_mapping
# from transformer_lens import HookedTransformer
# from tqdm import tqdm
# reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
# my_lm = Transformer(Configuration())
# key_mapping = construct_key_mapping()
# weights = {key_mapping.get(k, 'N/A'): v for k, v in reference_gpt2.state_dict().items()}
# weights.pop('N/A')
# my_lm.load_state_dict(weights, strict=False)
# my_lm = my_lm.to("cuda")

# ... successfully—
#
# In [10]: my_lm.load_state_dict(weights, strict=False)
# Out[10]: <All keys matched successfully>

# Try it out—
#
# test_string = '''"I want you, Jake," said the woman in the video as she took off her shirt. "Those negative comments on your pull requests were just a smokescreen—because I was afraid to confront the inevitability of'''
# continuation = ""
# for i in tqdm(range(100)):
#     test_tokens = reference_gpt2.to_tokens(test_string).to("cuda")
#     my_logits = my_lm(test_tokens)
#     continuation += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())
# print(continuation)

# It doesn't work.
#
# ~/Code/Exercises_A/ARENA/transformer.py in forward(self, residual)
#     186             )
#     187             # do the muliplication and re-batch the heads
# --> 188             attention_components[label] = (residual @ w).reshape(
#     189                 batch,
#     190                 self.configuration.context_size,
#
# RuntimeError: shape '[1, 1024, 12, 64]' is invalid for input of size 34560

# I drop into PuDB ...
#
# `residual @ w` has shape [1, 45, 768], which is indeed of size 34560
#
# >>> residual.shape
# torch.Size([1, 45, 768])
# >>> w.shape
# torch.Size([768, 768])

# 768 is the embedding dimensionality; where is the 45 coming from?—it was passed directly to `forward`.
# Looking up the stack, it appears that `tokens` is of length 45. What have we
# done greviously wrong such that our internals can't deal with predicting a
# 45-token context?? Am I assuming the context is always full, or what?

# I am explicitly using `self.configuration.context_size`. Maybe that needs to
# be the actual sequence length (dim 1 of the residual). (Yes, I see how einops
# avoids this.)

# Some more dumb bugs (forgotten `return`), and misuse of `nn.GELU`, and
# `demo_logits` copied from example code not existing ...

# Now it works.

# In [11]: test_string = '''A sense of life is a pre-conceptual equivalent of metaphysics, an
#     ...:  emotional, subconsciously integrated appraisal of'''
#     ...: for i in tqdm(range(100)):
#     ...:     test_tokens = reference_gpt2.to_tokens(test_string).to("cuda")
#     ...:     my_logits = my_lm(test_tokens)
#     ...:     test_string += reference_gpt2.tokenizer.decode(my_logits[-1, -1].argmax())
#     ...: print(test_string)
# 100%|████████████████████████████████████████████████████| 100/100 [00:01<00:00, 81.97it/s]
# A sense of life is a pre-conceptual equivalent of metaphysics, an emotional, subconsciously integrated appraisal of the world. It is a way of understanding the world that is not only a way of understanding the world, but also a way of understanding the world that is not only a way of understanding the world, but also a way of understanding the world that is not only a way of understanding the world, but also a way of understanding the world that is not only a way of understanding the world, but also a way of understanding the world that is not only a way of understanding the world, but also
