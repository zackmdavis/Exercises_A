import sys
from pathlib import Path

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter1_transformer_interp/exercises/")
section_dir = exercises_dir / "part3_indirect_object_identification"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from pathlib import Path
import torch
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

from plotly_utils import imshow, line, scatter, bar

import part3_indirect_object_identification.tests as tests
import part3_indirect_object_identification.solutions as solutions

torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# "Centering" weights means subtracting their mean. Centering weights that
# write to the residual stream is OK because of LayerNorm. Centering the
# unembedding weights is OK because softmax is translation-invariant. (It
# changes the logits, but not the probabilities.)

# Folding LayerNorm weights into the subsequent layer is reputedly OK, but I'm
# not sure what that means, and Claude isn't good at explaining it to me. (If I
# have two linear transformations A and B, I know that doing A then B is the
# same as doing the "folded" AB, but LayerNorm is dividing by a variance inside
# its forward method—how do we "fold" that?) The course links to an explanation.

# `refactor_factored_attn_matrices` is making W_O orthogonal, and W_Q and W_K
# "nice" in a way that shouldn't affect the computation?

# Anyway, for the indirect-object-identification task we want to measure the
# difference in logits between the correct and incorrect names.

# For the first prompt in the batch, this is
# `original_logits[0][-1][answer_tokens[0][0]] -
# original_logits[0][-1][answer_tokens[0][1]]`.  but as grown-up ML engineers,
# we want to do this in parallel rather than looping through
# `original_logits[i]...`.

# I thought the answer was going to be `logits[:, -1, answer_tokens[:, 0]] -
# logits[:, -1, answer_tokens[:, 1]]`, but this somehow turns out to have the
# wrong shape?! (expected (4,), got (4, 4))

# In the test, logits.shape is (4, 5, 6)
# logits[:, -1] shape is (4, 6). (Selecting the last sequence position.)
# answer_tokens[:, 0].shape is (4,)
# So logits[:, -1, answer_tokens[:, 0]] is, for each batch, giving me 4 token
# logits from the four entries of `answer_tokens[:, 0]`, but that's not what I
# want. What I want is, for each batch, to select the corresponding index from
# `answer_tokens[:, 0]`.

# I think what I actually want is
# [logits[:, -1][i][answer_tokens[:, 0][i]].item() for i in range(4)] ?!?!
#
# ... which passes the tests, but the fact that I wrote it that way is
# shameful, disgusting.

# XXX—this is apparently wrong
def my_attempted_logit_diff(logits, answer_tokens, per_prompt=False):
    # logits is (batch, seq, vocab_size)
    # answer_tokens is (batch, 2)
    a = torch.tensor([logits[:, -1][i][answer_tokens[:, 0][i]].item() for i in range(4)])
    b = torch.tensor([logits[:, -1][i][answer_tokens[:, 1][i]].item() for i in range(4)])
    logit_diffs = a - b
    if per_prompt:
        return logit_diffs
    else:
        return logit_diffs.mean()

# What does the non-disgusting version look like?

# Instructor's solution—
# final_logits = logits[:, -1, :]
# answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
# correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
# answer_logit_diff = correct_logits - incorrect_logits
# return answer_logit_diff if per_prompt else answer_logit_diff.mean()
#
# So I need to learn and internalize what `gather` and its `index` kwarg mean.

# Per the documentation (https://pytorch.org/docs/stable/generated/torch.gather.html), `gather` is
# out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
# out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
# out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

# We're prompted to brainstorm how the IOI behavior could be implemented in a
# transformer. Presumably names have to "attend to" previous names, and verbs
# or prepositions indicating an object also have to attend to previous names?
# And probably the previous attention "writes to the residual steam", such that
# a later layer can use that to decide which name to write? (This is still
# pretty vague.)

# We are advised: "Getting an output logit is equivalent to projecting onto a
# direction in the residual stream, and the same is true for getting the logit
# diff."

# Well, the residual stream is going to get converted into output logits. And
# ... it has to be a "direction" in the residual stream, because unembedding is
# a linear transformation?

# It's probably important to go slow (like they say about reading mathematics,
# and no one ever does). I had convinced myself that the columns of the
# unembedding matrix are what you dot a residual stream vector with in order to
# get an output logit for a particular token. That is, a correspondence
# betw—oh. Maybe that's right! I was about to say that I falsified my
# understanding because `model.tokens_to_residual_directions(0).shape` is
# (768,) but model.W_U[0].shape is (50257,), but [0] on a tensor is going to
# pull out a row, not a column. And indeed,
# `model.tokens_to_residual_directions(0) == model.W_U[:, 0]` is a vector of
# `True`s.

# Now we're going to use the "logit lens" technique—looking at the residual
# stream after each layer, and checking the logit diff predictions (what we
# would output if the remaining layers were deleted and we just did the
# enembedding here).

def my_attempted_residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions):
    stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return stack.mul(logit_diff_directions).sum() / residual_stack.shape[0]

# The test isn't passing here, and I don't think it's because of my einops
# disability. (The einops version gives the same answer.) When I paste in the
# instructor's version (equivalent by inspection), it does the same ... and I
# don't even think I'm returning the right type!! (I'm giving a scalar, but
# we're doing a layer plot next; it's supposed to return the average over the
# batch, but not aggregate over the layers/components. And in fact, the
# instructor's einsum expression does this, whole confusingly calling the layer
# dimension "...".

# Even running the instructor's code in the Colab notebook (before the
# exercise) has the mismatch?!
# Calculated average logit diff: 3.5518774986
# Original logit difference:     3.2613105774
# ------------------------------------------------------
# AssertionError

# I can't just shrug and say "Whatever" and keep going (and get lost) I need to
# understand this—

# Okay, I think the difference between 3.55 and 3.26 is a different issue from
# the shape mismatch. When I use the instructor's `... batch d_model, batch
# d_model -> ...` einops expression, I do get a graph that looks like the one
# in the solutions (unlike the `.mul([...]).sum()` code).
#
# Or, it looks like it shape-wise; the actual figures don't match! Probably
# scaling by the wrong amount?

# The solutions notebook says that 3.55 is the correct value ... and that is
# what the instructor's code verbatim says. So my `logit_diff` implementation
# (with the list comprehensions) was also wrong despite passing tests?!

# This is unacceptable, crazy—I have to make it make sense! And it shouldn't
# take interminable hours of sidereal time!! There are two functions in
# question! I have implementations from the instructor. I'm going to annotate
# them with my own comments so it's very clear that I understand every line!
# And then I'm going to continue with the notebook.

prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
name_pairs = [
    (" John", " Mary"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]

prompts = [
    prompt.format(name)
    for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1]
]
answers = [names[::i] for names in name_pairs for i in (1, -1)]
answer_tokens = torch.concat([
    model.to_tokens(names, prepend_bos=False).T for names in answers
])

tokens = model.to_tokens(prompts, prepend_bos=True).to(device)
original_logits, cache = model.run_with_cache(tokens)

# Instructor's solution, annotated with my explanation.
def logit_diff(logits, answer_tokens, per_prompt = False):
    # `logits` is (batch, seq_len, vocab_size): for every batch, and for every
    # sequence position, we have a function of all the vocabulary tokens that
    # becomes a probability distribution when you shove it through a softmax.
    #
    # Then we take the logits for just the last sequence position, to get (batch, vocab_size)
    final_logits = logits[:, -1, :]
    # Then we `gather` by the index of answer tokens—
    # answer_logits[batch][answer] = final_logits[batch][answer_tokens[batch][answer]]
    # which is (batch, 2)
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    # Unpack the correct and incorrect answers
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    # And subtract.
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


original_average_logit_diff = logit_diff(original_logits, answer_tokens)

# `answer_tokens` is (8, 2): eight prompts, by two correct and incorrect tokens
#
# `answer_residual_directions` (8, 2, 768), collecting the unembedding matrix
# column for each of those sixteen tokens, respectively
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)

# Then we splat out the correct and incorrect directions.
correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
# And subtract.
logit_diff_directions = correct_residual_directions - incorrect_residual_directions

# And then we're going to apply those directions to the residual stream.
final_residual_stream = cache["resid_post", -1]
# Specifically, the last position thereof.
final_token_residual_stream = final_residual_stream[:, -1, :]
# After correcting for LayerNorm weirdness.
scaled_final_token_residual_stream = cache.apply_ln_to_stack(final_token_residual_stream, layer=-1, pos_slice=-1)

# Claude is helping me rewrite the instructor's solution without einops. (This is
# educational, because even with LLM assistance, to adapt it, I still have to
# think about what the code is saying, what it means.)

# It turns out that the ellipsis is actually significant in einsum (for a
# variable number of dimensions, possibly zero); it's not just a quirky naming
# choice.

# I'm in this absurd situation where the einsum code works, but I don't want to
# just accept that and move on; I want to ensure that I understand I know what
# it's doing, as proven by having plain linear-algebra operations that do the
# same thing, and neither me nor Claude can figure it out. I need to go back to
# basics and re-review the Ch. 0 stuff rather than delusionally assuming I
# could get by without it.

# ------

# I got sidetracked from this for a while, but one of the things I did while
# being sidetracked was in fact implementing `einsum` in Rust
# (https://github.com/zackmdavis/meinsum), which will hopefully help with that
# being a blocker? Where was I??

# Let's review our story so far: we want to understand how the model knows
# which name to predict in prompts like "When John and Jane went to the park,
# John gave the ball to". This is quantified as the difference in the logits
# for ` John` and ` Jane` (where logits are the model outputs that you shove
# through a softmax to get a probability distribution).

# And that difference amounts to ... the dot product of the residual steam with
# a particular direction?

# The unembedding matrix has shape (dimensionality, vocab_size).
# The residual stream vector has shape (batch, sequence, dimensionality).

# If we ignore the batch direction and just take the residual stream at a
# particular position, that's a vector of shape (dimensionality,).

# The prediction for that sequence position comes from multiplying that vector
# by the unembedding matrix (on the left with a transpose).

# x^T · W_U is (1 × dimensionality) · (dimensionality × vocab_size) = (1 × vocab_size).

# Instructor's solution
def residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions=logit_diff_directions):
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size


# Basically, for every layer of our transformer, we can look at the residual
# stream, and see the logit difference "if the network ended there". As it
# happens, performance on this this task "if the network ended there" only
# perks up around layer 9, so there's probably a relevant circuit there?

# You can do this at the level of transformer blocks or more specifically
# MLP/attention layers.

# Attention heads move information between residual stream positions, which may
# not correspond to what token is at that position.

# So, that was "logit attribution". Another technique is "activation patching",
# introduced by Bau and Meng (of ROME fame) as "causal tracing".
