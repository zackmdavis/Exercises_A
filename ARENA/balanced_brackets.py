import json
import sys
from pathlib import Path

import einops
import torch

from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter1_transformer_interp/exercises/")
section_dir = exercises_dir / "part7_balanced_bracket_classifier"

if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part7_balanced_bracket_classifier.brackets_datasets import SimpleTokenizer, BracketsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB = "()"

config = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=2,
    d_mlp=56,
    n_layers=3,
    attention_dir="bidirectional", # defaults to "causal"
    act_fn="relu",
    d_vocab=len(VOCAB)+3, # plus 3 because of end and pad and start token
    d_vocab_out=2, # 2 because we're doing binary classification
    use_attn_result=True,
    device=device,
    use_hook_tokens=True
)

model = HookedTransformer(config).eval()
state_dict = torch.load(section_dir / "brackets_model_state_dict.pt")
model.load_state_dict(state_dict)

tokenizer = SimpleTokenizer("()")

def add_perma_hooks_to_mask_pad_tokens(model, pad_token):
    def cache_padding_tokens_mask(tokens, hook):
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_token, "b sK -> b 1 1 sK")

    def apply_padding_tokens_mask(
        attn_scores, # (batch, head, seq_Q, seq_K)
        hook,
    ):
        attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
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
