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
h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

# We get a nice scatterplot. It seems like an algorithm for detecting total
# elevation failures must live in Head 2.0, and one for detecting ever-negative
# failures must live in Head 2.1?

# It makes sense that "only-ever-negative" is rarer (sparser cloud of dots),
# because it's more of a coincidence to have exactly equal numbers of
# parentheses that just don't close each other.

# TODO: continue ...

def get_attn_probs(model, data, layer, head):
    activations = get_activations(model, data.toks, ...)
    pass
