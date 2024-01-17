import torch

from transformer_lens import HookedTransformer

gpt2_small = HookedTransformer.from_pretrained("gpt2-small")

# We are asked, "Can you see evidence of any induction heads at work, on this text?"
#
# My response: the fact that it knows to predict `former` after seeing `Trans`
# seems like in-context learning. (When I give GPT-3 davinci an input starting
# with `Trans`, it continues with `lation` or `flagration` or `it`.)
#
# (Instructor's answer is slightly more specific, pointing to it predicting
# `ookedTransformer` the second time but not the first.)


# There's an exercise to verify that the layer-0 attention pattern is the same
# as what you get from the Q and K activations. Given how confused I was by the
# attention implementation the past couple days, I definitely need to do this
# one.

# In [54]: q = cache["q", 0]

# In [55]: k = cache["k", 0]

# In [56]: q.shape, k.shape
# Out[56]: (torch.Size([33, 12, 64]), torch.Size([33, 12, 64]))

# attention_scores = q @ k.mT
# attention_scores /= 8  # √d_head
# masked = attention_scores.masked_fill_(
#     torch.triu(torch.ones(12, 12, device=attention_scores.device), diagonal=1).bool(),
#     -10000.0,
# )
# attention_pattern = masked.softmax(-1)

# And I think that's supposed to be equal to
# `cache["blocks.0.attn.hook_pattern"].shape`? But I'm not even getting the
# right shape.

# In [78]: cache["blocks.0.attn.hook_pattern"].shape
# Out[78]: torch.Size([12, 33, 33])

# In [79]: attention_pattern.shape
# Out[79]: torch.Size([33, 12, 12])

# The instructor's solution is doing `layer0_attn_scores = einops.einsum(q, k,
# "seqQ n h, seqK n h -> n seqQ seqK")` and getting a shape [12, 33, 33].

# We get some visualizations (run in Colab) ...

# We are asked: why couldn't an induction head form in a one-layer model?
# I answer: because the head is part of a two layer circuit. In order to detect
# the "token after a copy of this-and-such token" you need to have already been
# paying to this-and-such token in a previous layer.

# There's an exercise to see the model do better on predicting the second half of a random sequence ...

def generate_repeated_tokens(model, seq_len, batch=1):
    sequence = [random.randint(1, 40000) for _ in range(seq_len)]
    return torch.tensor([[50256] + sequence * 2] * batch).to(next(model.parameters()).device)

# You can still see me thinking like a normie Python programmer in the
# above. The instructor's solution uses more local jargon—
#
# prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
# rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
# rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
# return rep_tokens

# In [98]: gpt2_small.to_string(generate_repeated_tokens(gpt2_small, 10))
# Out[98]: ['<|endoftext|> Gener alarm Fun precipitation upsetMouseLegend concedReports Chen Gener alarm Fun precipitation upsetMouseLegend concedReports Chen']

# In [99]: gpt2_small.to_string(generate_repeated_tokens(gpt2_small, 10))
# Out[99]: ['<|endoftext|> TueurgicalEvery thing merchantsextra embracedle squirrel identifiable TueurgicalEvery thing merchantsextra embracedle squirrel identifiable']

def run_and_cache_model_repeated_tokens(model, seq_len, batch=1):
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens, remove_batch_dim=True)
    return rep_tokens, rep_logits, rep_cache

# We are asked to look back at the visualization earlier in the Colab and look
# for which heads might be serving as induction heads—or rather, we can adapt
# that code to get another Colab visualization for this particular string.

# attention_pattern = rep_cache["pattern", 1, "attn"]
# display(cv.attention.attention_patterns(
#     tokens=rep_str,
#     attention=attention_pattern,
#     attention_head_names=[f"L1H{i}" for i in range(12)],
# ))

# Heads L1H4 and L1H10 have a very distinct dark diagonal line. (Not the main
# diagonal, but about halfway down.) Those are probably the induction heads!
# The diagonal line indicating attention halfway back through the sequence?
# (This feels intuitive, but I would struggle to write out the proof.)

# Instructor's answer confirms (and notes that head 6 has a much weaker
# induction stripe).

# TODO continue—"calculate induction scores with hooks" and "find induction heads in GPT2-small"
