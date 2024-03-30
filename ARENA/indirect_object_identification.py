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
