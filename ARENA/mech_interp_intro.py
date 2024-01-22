import torch

from transformer_lens import HookedTransformer, FactoredMatrix

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
    return torch.tensor([[50256] + sequence * 2] * batch).to(
        next(model.parameters()).device
    )


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

# How do I take average of a diagonal over a batch? I think `dim1=2, dim2=3`
# (these kwargs having been mentioned by the text as potentially useful) should
# give me the diagonal over just the last two dimensions ... and it gives me a
# batch of those, I guess?

# After some thought, I got it!
# You can still see my Python normie tendencies (using a for loop instead of
# tensor index magic), but getting the job done is paramount.
def induction_score_hook(pattern, hook):
    (batch, headcount, dest_posns, source_posns) = pattern.shape

    diagonals = pattern.diagonal(offset=-(seq_len - 1), dim1=2, dim2=3)
    assert diagonals.shape[:2] == (batch, headcount)
    for head in range(headcount):
      induction_score_store[hook.layer(), head] = diagonals[:, head, ...].mean()
# Instructor's solution was
    # induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # induction_score_store[hook.layer(), :] = induction_score


# We are asked why second-layer heads would have higher logit contributions
# than first-layer, and hinted to think about what the attributions mean in
# terms of paths.
#
# I reply ... um, because the second layer of heads incorporates the inputs of
# the first layer?? (Instructor's answer is broadly similar?—"the attributions
# for layer-1 heads will involve not only the single-head paths through those
# attention heads, but also the 2-layer compositional paths through heads in
# layer 0 and layer 1".)

# We're asked to summarize how the induction circuit in our simple model must
# work, given our logit-attribution and ablation findings.
#
# I reply: 0.7 is "attending to the previous token", and 1.4 and 1.10 and using
# that to look up the token that came after that previous token earlier? This
# is a bit vague, though.

# I think I need to do some slow, line-by-line focus to really get this (and
# get the model to play with locally rather than being chained the Colab).

# "Head 0.7 is a previous token head (the QK-circuit ensures it always attends
# to the previous token)." What does that mean, in detail?

# It means that the attention pattern has a diagonal streak "one below" the
# main diagonal. So when "position A attends to position B", then A is the row
# and B is the column: that's the convention that would make a "previous token
# head" be the below-main diagonal (with e.g., the zero-index 1st row matching
# with the 0th column).

# GPT-4 elaborates: "In these matrices, each row corresponds to a query (the
# token considering others) and each column corresponds to a key (the token
# being considered)."

# Next step: "The OV circuit of head 0.7 writes a copy of the previous token in
# a different subspace to the one used by the embedding." What does that mean?
# That Q, K, and V happen to be such that, when we take V · Softmax(K^⊤ · Q),
# the position for the nth token ends up encoding information about what the
# n–1th token was.

# Next step: "The output of head 0.7 is used by the key input of head 1.10 via
# K-Composition to attend to 'the source token whose previous token is the
# destination token'."

# Suppose (following the diagram) that the n-1th token was `D`, and the nth
# token was `urs`. So the nth position in the output of the first layer ends up
# encoding the information "the previous token was D" somehow (in some
# subspace; the embedding space is high-dimensional and can be assumed to have
# plenty of room for whatever we want to store).

# So then when `D` appears again in the input at a later position (call it k),
# the attention pattern for head 1.10 can focus on positions where "the
# previous token was D" has been stored, and look up what the actual token was
# at that time `urs`, and predict that for the k+1th position?

# This still feels handwavy! How does it "focus on positions where 'the
# previous token was D' has been stored, and how can it looks up what the token
# at that time was? Where is that information actually stored?

# OK, we get some quiz questions: "what is the interpretation of each of the
# following matrices? [...]  If you're confused, you might want to read the
# answers to the first few questions and then try the later ones. [...]  you
# should describe the type of input it takes, and what the outputs represent."

# W^h_OV
# I reply: the OV circuit for the hth head. Which means ...?
# Instructor's answer (my paraphase): it has dimensions d_model, d_model (the
# emedding dimensionality). It represents the flow of info from source to
# destination. If x is a "source" vector in the residual stream (which has
# sequence-length vectors), then xW^h_OV is the destination.

# W_E W^h_{OV} W^U
# I reply: this is the embedding, composed with the OV circuit, composed with
# the unembedding. The input and output both have the embedding
# dimensionality. I'd say that this is what the model outputs, but I'm not sure
# how the QK circuit fits in?

# Exercise—compute OV circuit for 1.4
#
# My guess was `FactoredMatrix(model.W_E, model.W_V[layer][head_index]) @
# model.W_U`, but that doesn't even shape-check, and anyway it was a retarded
# guess because of course the OV circuit is also going to use W_O


# Getting a CUDA error running this in Colab—but it also occurs when I run the
# instructor's solution!—guess I need to gear up locally? (The Colab was nice
# for having the tests configured.)
def top_1_acc(full_OV_circuit, batch_size=1000):
    indices = torch.randint(0, model.cfg.d_vocab, (batch_size,))
    hits = 0
    # I'm still being a Python normie with these loops, I guess?!
    argmaxes = full_OV_circuit[indices, indices].AB.argmax(dim=0)
    for i, argmax in enumerate(argmaxes):
        if i == argmax:
            hits += 1
    return hits / len(argmaxes)


# But when I run locally, I'm getting much better results?! (Course text says
# "This should return about 30.79%"; I'm getting around 75%.)

# Summing `FactoredMatrix`es doesn't work, but the diagram suggests that we
# should be concatting them?

def effective_circuit():
    return (
        model.W_E @
        FactoredMatrix(
            torch.concat((model.W_V[1, 4], model.W_V[1, 10]), dim=1),
            torch.concat((model.W_O[1, 4], model.W_O[1, 10]), dim=0)
        ) @
        model.W_U
    )


def positional_pattern():
    return model.W_pos @ model.W_Q[0, 7] @ model.W_K[0, 7].T @ model.W_pos.T
