import json
import sys
from pathlib import Path

import einops
import torch

from transformer_lens import (
    utils,
    ActivationCache,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter1_transformer_interp/exercises/")
section_dir = exercises_dir / "part7_balanced_bracket_classifier"

if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part7_balanced_bracket_classifier.brackets_datasets import (
    SimpleTokenizer,
    BracketsDataset,
)
from part7_balanced_bracket_classifier import tests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB = "()"

config = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=2,
    d_mlp=56,
    n_layers=3,
    attention_dir="bidirectional",  # defaults to "causal"
    act_fn="relu",
    d_vocab=len(VOCAB) + 3,  # plus 3 because of end and pad and start token
    d_vocab_out=2,  # 2 because we're doing binary classification
    use_attn_result=True,
    device=device,
    use_hook_tokens=True,
)

model = HookedTransformer(config).eval()
state_dict = torch.load(section_dir / "brackets_model_state_dict.pt")
model.load_state_dict(state_dict)

tokenizer = SimpleTokenizer("()")


def add_perma_hooks_to_mask_pad_tokens(model, pad_token):
    def cache_padding_tokens_mask(tokens, hook):
        hook.ctx["padding_tokens_mask"] = einops.rearrange(
            tokens == pad_token, "b sK -> b 1 1 sK"
        )

    def apply_padding_tokens_mask(
        attn_scores,  # (batch, head, seq_Q, seq_K)
        hook,
    ):
        attn_scores.masked_fill_(
            model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5
        )
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)

    return model


model.reset_hooks(including_permanent=True)
model = add_perma_hooks_to_mask_pad_tokens(model, tokenizer.PAD_TOKEN)

N_SAMPLES = 2000
with open(section_dir / "brackets_data.json") as f:
    data_tuples = json.load(f)
    print(f"loaded {len(data_tuples)} examples")
assert isinstance(data_tuples, list)
data_tuples = data_tuples[:N_SAMPLES]
data = BracketsDataset(data_tuples).to(device)
data_mini = BracketsDataset(data_tuples[:100]).to(device)

# We are asked to plot a histogram of sequence lengths. I think a
# `collections.Counter` is just as good. All lengths are even. In our sample of
# 5K, we go from 862 length-2s down to 172 length-40s.


# We're asked to solve it ourselves with conventional code—which ought to be routine.
def is_balanced_classic(parens):
    open_groups = 0
    for char in parens:
        match char:
            case '(':
                open_groups += 1
            case ')':
                open_groups -= 1
        if open_groups < 0:
            return False
    return open_groups == 0


# But procedural code isn't necessarily cheap for a transformer (where for a
# given sequence position, the same weights execute).
#
# I'm still not used to breaking out of for-loop thinking. Surely we need a
# sequential scan in order to do the `open_groups < 0` check and thus rule out
# sequences like ')(' which have an equal number of open- close-parens, but
# don't respect the order?
#
# The hint does mention "cumulative sum", so I guess that's right (and not in
# contradiction of vectorization as I had imagined)? GPT-4 points out that
# there is in fact a `torch.cumsum`.
def is_balanced_vectorized(parens):
    just_parens = torch.where(parens < 3, 0, parens)
    group_deltas = torch.where(just_parens == 3, 1, just_parens)
    group_deltas = torch.where(group_deltas == 4, -1, group_deltas)
    cumulative = group_deltas.cumsum(dim=-1)
    return (torch.all(cumulative >= 0) and cumulative[..., -1] == 0).item()


# Now we're trying to reason about the network's output, reducing questions to
# answers about the previous layer.

# The classification probabilities depend on the differences in logits before
# the softmax. (Actually, our model itself doesn't seem to have a softmax
# layer, but if it did, we'd want to reason backwards to what fed into it.)

# "Since the logits are each a linear function of the output of the final
# LayerNorm, their difference will be some linear function as well." That is,
# if L₁(x) is linear and L₂(x) is linear, then (L₁ − L₂)(x) is linear.

# "In other words, we can find a vector in the space of LayerNorm outputs such
# that the logit difference will be the dot product of the LayerNorm's output
# with that vector."

# The logits have shape seq_len × 2 (well, after the batch dimension).
# That's because it's the product of—
#   • the unembedding matrix having shape embedding_dimensionality × 2, and
#   • the final LayerNorm output having shape seq_len × embedding_dimensionality.


def get_post_final_ln_dir(model):
    return model.W_U[:, 0] - model.W_U[:, 1]


# LayerNorm isn't linear, but we're going to approximate it as linear, but that
# means taking measurements.

from part7_balanced_bracket_classifier.solutions import (
    get_activations,
    LN_hook_names,
    get_out_by_components,
    get_pre_20_dir,
    get_out_by_neuron,
)
from sklearn.linear_model import LinearRegression


def get_ln_fit(model, data, layernorm, seq_pos=None):
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    cache = get_activations(model, data.toks, [input_hook_name, output_hook_name])

    before = cache[input_hook_name]
    after = cache[output_hook_name]
    batch, seq_len, dimensionality = before.shape

    # The hints clarify: if we have a sequence position, we're selecting just
    # that dimension, but if not, then we reshape so that the regression is
    # going over the Cartesian product batch × position
    if seq_pos is None:
        before = before.reshape(batch * seq_len, dimensionality)
        after = after.reshape(batch * seq_len, dimensionality)
    else:
        before = before[:, seq_pos, :]
        after = after[:, seq_pos, :]

    assert len(before.shape) == len(after.shape) == 2
    assert before.shape[1] == after.shape[1] == dimensionality

    b = before.cpu()
    a = after.cpu()
    fit = LinearRegression().fit(b, a)
    return fit, fit.score(b, a)


# 98% fit for position 0, 97% fit for all positions—nice
# (um, wait, the second one seems to have been looking at the next layer back?)

# Next, we're supposed to us that fit to find the direction in the residual
# stream which most points towards unbalanced classifications. But ... how? The
# regression coëfficients seem to be a dimensionality × dimensionality
# matrix—what do I do with it?
#
# The obvious guess is to multiply its inverse with the `post_final_ln_dir`
# vector? ... perhaps, on the right??
#
# Because if `pre · A = post` then `pre · AA⁻¹ = post · A⁻¹ = pre`?

# def get_pre_final_ln_dir(model, data):
#     fitted, _score = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
#     fit = torch.tensor(fitted.coef_).to(device)
#     return get_post_final_ln_dir(model) @ fit**-1

# Dimensionality checks out, but tests say no.
#
# ````
# AssertionError: Tensor-likes are not close!
#
# Mismatched elements: 56 / 56 (100.0%)
# Greatest absolute difference: 3642.8369140625 at index (34,) (up to 1e-05 allowed)
# Greatest relative difference: 653.4279174804688 at index (11,) (up to 1.3e-06 allowed)
# ```

# The instructor's solution says we're multiplying by the transpose.


def get_pre_final_ln_dir(model, data):
    fitted, _score = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
    fit = torch.tensor(fitted.coef_).to(device)
    return fit.T @ get_post_final_ln_dir(model)


# Why the transpose, though? GPT-4 and Gemini use confident language while
# trying to explain, but I'm suspicious that they're hallucinating.

# ... moving on for now? The residual stream is a sum of terms representing
# paths through the network.

# I'm confused at how the instructions claim that head 0.0, 0.1, 1.0, 1.1,
# &c. are going to be separate terms, but when I look at the model object, I
# don't see separate numbered heads per layer?

# ['hook_embed', 'hook_pos_embed', 'blocks.0.hook_attn_out',
# 'blocks.0.hook_mlp_out', 'blocks.1.hook_attn_out', 'blocks.1.hook_mlp_out',
# 'blocks.2.hook_attn_out', 'blocks.2.hook_mlp_out'] is 8, not 10

# Looking at the solution, we wanted 'attn.hook_result' rather than
# 'hook_attn_out', but other than that, it looks like I had the right idea? (8
# elements in the `all_hook_names` list, despite the text saying 10 ... but
# then the function itself eventually outputs 10.)


# We are instructed to "compute a (10, batch)-size tensor called
# out_by_component_in_unbalanced_dir. The [i, j]th element of this tensor
# should be the dot product of the ith component's output with the unbalanced
# direction, for the jth sequence in your dataset."

# `get_pre_final_ln_dir` gives us "the unbalanced direction"

unbalanced_direction = get_pre_final_ln_dir(model, data)
out_by_components = get_out_by_components(
    model, data
)  # (10, batch, seq_len, dimensionality)

# I'm so retarded—all these weeks, and I still can't think in tensors. How would I do it with for loops?
# Or am I just having trouble parsing the English of "for the jth sequence"?

# for i in range(10):
#     for j in range(len(batch)):
#         [i][j] = out_by_components[i][j] @ unbalanced_direction

# In terms of dimensions that drop out (for einsum purposes), I'm probably
# going to want to say something like "component, batch, seq_len,
# dimensionality -> component batch"?

# Looking at the solution, I was on the wrong track: "they" (the immortal
# "they") want you to pluck out the 0th sequence position `out_by_components[:,
# :, 0, :]` (10, batch, dimensionality)

# (Oh, there was also a fold-out hint.)

out0 = out_by_components[..., 0, :]
out_by_component_in_unbalanced_direction = out0 @ unbalanced_direction

# And then we subtract the mean. `[:, data.isbal]` from the solutions is
# selecting just the balanced rows.
out_by_component_in_unbalanced_direction -= (
    out_by_component_in_unbalanced_direction[:, data.isbal].mean(dim=1).unsqueeze(1)
)

# (And now that I've got my virtualenv/path stuff sorted out, I can call the
# instructor's `plotly_utils`—which is not the PyPI package of the same
# name—and see the graphs.)


def is_balanced_vectorized_return_both(toks):
    just_parens = torch.where(toks < 3, 0, toks)
    group_deltas = torch.where(just_parens == 3, 1, just_parens)
    group_deltas = torch.where(group_deltas == 4, -1, group_deltas)

    # I also forgot about the right-to-left business.
    group_deltas = group_deltas.flip(-1)

    totals = group_deltas.sum(dim=-1)
    cumulative = group_deltas.cumsum(dim=-1)

    # Still so bad at tensor manipulation! This fails tests!!
    # negative_failures = (cumulative < 0).any(dim=1)
    # elevation_failures = totals != 0

    # Instructor's version
    elevation_failures = cumulative[:, -1] != 0
    negative_failures = cumulative.max(-1).values > 0

    return negative_failures, elevation_failures


negative_failure, total_elevation_failure = is_balanced_vectorized_return_both(
    data.toks
)
h20_in_unbalanced_dir = out_by_component_in_unbalanced_direction[7]
h21_in_unbalanced_dir = out_by_component_in_unbalanced_direction[8]

# We get a nice scatterplot. It seems like an algorithm for detecting total
# elevation failures must live in Head 2.0, and one for detecting ever-negative
# failures must live in Head 2.1?

# It makes sense that "only-ever-negative" is rarer (sparser cloud of dots),
# because it's more of a coincidence to have exactly equal numbers of
# parentheses that just don't close each other.

# We're supposed to write a function that extracts the attention patterns for a
# given layer and head, but ... I don't see different heads in the activation
# cache?

# blocks.0.attn.hook_q torch.Size([2000, 42, 2, 28])
# blocks.0.attn.hook_k torch.Size([2000, 42, 2, 28])
# blocks.0.attn.hook_v torch.Size([2000, 42, 2, 28])
# blocks.0.attn.hook_attn_scores torch.Size([2000, 2, 42, 42])
# blocks.0.attn.hook_pattern torch.Size([2000, 2, 42, 42])
# blocks.0.attn.hook_z torch.Size([2000, 42, 2, 28])
# blocks.0.attn.hook_result torch.Size([2000, 42, 2, 56])
# blocks.0.hook_attn_out torch.Size([2000, 42, 56])

# Oh, I get it—the different heads are the second dimension.

def get_attention_probabilities(model, data, layer, head):
    attention_scores = get_activations(model, data.toks, "blocks.{}.attn.hook_pattern".format(layer))
    return attention_scores[:, head, ...]

# It turns out that in head 2.0, position 0 is mostly paying attention to
# position 1, which means information about unbalancedness has already been
# copied to position 1 from the previous layer. So now we're going to keep
# digging backwards—

def get_WOV(model, layer, head):
    # return model.W_O[layer][head] @ model.W_V[layer][head]
    # Despite the name, the instructor's solution has the multiplication in the
    # other direction (which also helps the dimensional analysis later)
    return model.W_V[layer][head] @ model.W_O[layer][head]

# I'm not sure how to write this part ... we're told that it should be
# conceptually similar to `get_pre_final_ln_dir` from above.

# `get_pre_final_ln_dir` was finding a linear approximation to the layernorm,
# and multiplying that by `get_post_final_ln_dir`, which in turn was
# `model.W_U[:, 0] - model.W_U[:, 1]`, the difference in the ... classification
# dimension?—in the unembedding matrix.

# Dimensions aren't lining up on my first attempt: `fit` is 56×56, but
# post-20-direction is 28.

# def get_pre_20_direction(model, data):
#     WOV = get_WOV(model, 2, 0)
#     post_20_direction = WOV[:, 0] - WOV[:, 1]
#     fitted, _score = get_ln_fit(model, data, layernorm=model.blocks[2].ln1, seq_pos=1)
#     fit = torch.tensor(fitted.coef_).to(device)
#     return fit.T @ post_20_direction

# Instructor's solution was
#
# def get_pre_20_dir(model, data) -> Float[Tensor, "d_model"]:
#     # SOLUTION
#     W_OV = get_WOV(model, 2, 0)
#     layer2_ln_fit, r2 = get_ln_fit(model, data, layernorm=model.blocks[2].ln1, seq_pos=1)
#     layer2_ln_coefs = t.from_numpy(layer2_ln_fit.coef_).to(device)
#     pre_final_ln_dir = get_pre_final_ln_dir(model, data)
#     return layer2_ln_coefs.T @ W_OV @ pre_final_ln_dir


# Next we're doing component magnitudes again. I get a dimension mismatch (10
# vs. 7); a brief glance at the instructor's solution indicates that `:-3` is
# in fact indicated. (It makes sense; the layers are in order.)

out1 = out_by_components[:-3, :, 1, :]
out_by_component_in_pre_20_unbalanced_direction = out1 @ get_pre_20_dir(model, data)
out_by_component_in_pre_20_unbalanced_direction -= (
    out_by_component_in_pre_20_unbalanced_direction[:, data.isbal].mean(dim=1).unsqueeze(1)
)


# I'm very confused about what code I'm supposed to write for
# `get_out_by_neuron`. We get pseudocode—that mentions f, for example. The
# activation function? It was a GeLU in our demo transformer from last week—how
# do I access it here? Or is the psuedocode "just conceptual" and I'm supposed
# to be accessing an activation from the cache?

# Instructor's solution—
# W_out = model.W_out[layer] # [neuron d_model]
# # Get activations of the layer just after the activation function, i.e. this is f(x.T @ W_in)
# f_x_W_in = get_activations(model, data.toks, utils.get_act_name('post', layer)) # [batch seq neuron]
# # f_x_W_in are activations, so they have batch and seq dimensions - this is where we index by seq if necessary
# if seq is not None:
# f_x_W_in = f_x_W_in[:, seq, :] # [batch neuron]
# # Calculate the output by neuron (i.e. so summing over the `neurons` dimension gives the output of the MLP)
# out = einops.einsum(
#     f_x_W_in,
#     W_out,
#     "... neuron, neuron d_model -> ... neuron d_model",
# )

def get_out_by_neuron_in_20_direction(model, data, layer):
    # instructor's solution clarifies that this is `get_pre_20_dir` (I was
    # imagining it being the `out_by_component_in_pre_20_unbalanced_direction`
    # that we computed above)
    return get_out_by_neuron(model, data, layer, seq=1) @ get_pre_20_dir(model, data)


def get_q_and_k_for_given_input(model, tokenizer, parens, layer):
    hook_names = [key.format(layer) for key in ['blocks.{}.attn.hook_q', 'blocks.{}.attn.hook_k']]
    cache = get_activations(model, tokenizer.tokenize(parens), hook_names)
    return [cache[name].squeeze(0) for name in hook_names]

# Next we learn about "activation patching"/"causal tracing" ...

# The embedding vectors are being used to tally up the number of open and close parens!

def embedding(model, tokenizer, char):
    assert char in ("(", ")")
    idx = tokenizer.t_to_i[char]
    return model.W_E[idx]

fitted, _score = get_ln_fit(model, data, layernorm=model.blocks[0].ln1)
L = torch.tensor(fitted.coef_).to(device)
v_L = embedding(model, tokenizer, "(").T @ L.T @ get_WOV(model, 0, 0)
v_R = embedding(model, tokenizer, ")").T @ L.T @ get_WOV(model, 0, 0)
print("Cosine similarity: ", torch.cosine_similarity(v_L, v_R, dim=0).item()) # −0.997
