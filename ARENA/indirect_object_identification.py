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

def logit_diff(logits, answer_tokens, per_prompt=False):
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
answer_tokens = t.concat([
    model.to_tokens(names, prepend_bos=False).T for names in answers
])

tokens = model.to_tokens(prompts, prepend_bos=True).to(device)
original_logits, cache = model.run_with_cache(tokens)

original_average_logit_diff = logit_diff(original_logits, answer_tokens)

# Now we're going to use the "logit lens" technique—looking at the residual
# stream after each layer, and checking the logit diff predictions (what we
# would output if the remaining layers were deleted and we just did the
# enembedding here).

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

def residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions):
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
