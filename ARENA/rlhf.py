import sys
from pathlib import Path

import torch
from torch import nn

from transformer_lens import HookedTransformer

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter2_rl/exercises/")
section_dir = exercises_dir / "part4_rlhf"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part4_rlhf.tests as tests
import part4_rlhf.solutions as solutions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (The solutions are trying to import `eindex`, which doesn't work even after
# `pip install`ing it, but there's an alternate implementation of the function
# in question, so I should be fine to comment it out.)
#
# Mystery solved—library is written by the author and their page
# https://www.perfectlynormal.co.uk/blog-eindex (linked later in the notebook)
# recommends installing from Git.


# So, how does the state–action–reward paradigm apply to language modeling?
# The state is the sequence so far.
# The actions are our token vocabulary.
# There'll be a reward at the end of the sequence.

# The LM is the actor, and we'll train a value head as the critic.
# We'll have PPO-like rollout and learning phases, plus a KL penalty to keep
# the policy from diverging too much from the base LM.

# It's D_KL(π_PPO || π_base), because we want the penalty to be prohibitive for
# behavior that would have a low probability on the base policy. That's the
# correct order, because D_KL(P || Q) is how surprised you'll be on observing
# data from P when you expected Q. So the base LM should be Q, because that's
# what sets the expectations that we want to penalize being surprising with
# respect to.

# The value head will get tacked onto the residual stream before unembedding.

# We're going to set a hook to call the value head.

# ... I do not remember how TransformerLens hooks work. Specifically, how do I
# find the name of that final layernorm?—I can `run_with_cache` and look at the
# cache?


class TransformerWithValueHead(nn.Module):
    def __init__(self, base_model="gpt2-small"):
        super().__init__()
        self.base_model = HookedTransformer.from_pretrained(base_model)
        self.value_head = nn.Sequential(
            nn.Linear(self.base_model.cfg.d_model, 4 * self.base_model.cfg.d_model),
            nn.ReLU(),
            nn.Linear(4 * self.base_model.cfg.d_model, 1),
        )

    def value_hook(self, activation, hook):
        self.value = self.value_head(activation).squeeze(-1)

    def forward(self, tokens):
        logits = self.base_model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=[('ln_final.hook_normalized', self.value_hook)],
        )
        return logits, self.value


get_samples = solutions.get_samples

model = TransformerWithValueHead().to(device)

# We are asked to ponder: if we reward periods in the output, what will the
# result be, given that the KL penalty ties us to GPT-2-plausible text? Obvious
# answer: initialisms and short sentences. Any more specific predictions?
# Ellipses!

# We're going to normalize our rewards.


def normalize_reward(reward, eps=1e-5):
    return (reward - reward.mean()) / (reward.std() + eps)


RLHFTrainingArgs = solutions.RLHFTrainingArgs

# We're going to use a simpler calculation of the advantage than in the PPO. (I
# might say, "Which is good, because I didn't understand the PPO version", but
# that's school thinking; not understanding things is bad, whether or not
# you're being graded.)

# A(s_t, a_t) = Q(s_t, a_t) − V(s_t)

# But for the language modeling setting, a_t just concatenates to the existing
# sequence.


@torch.no_grad()
def compute_advantages(values, rewards, prefix_len):
    batch_size, seq_len = values.shape
    return (
        torch.concat((values[:, prefix_len:seq_len-1], rewards.unsqueeze(1)), dim=1)
        - values[:, prefix_len-1 : seq_len-1]
    )


ReplayMinibatch = solutions.ReplayMinibatch
ReplayMemory = solutions.ReplayMemory

# I ended up getting some help from Claude for this one, which is embarrassing
# because it's simple and I should have been able to get it fully on my own. I
# think I tripped over the instructor's comment about using `log_softmax` for
# numerical stability. In the formula Σ P * log(P/Q), the latter actually has a
# log in it for real, which is distinct from using exp(log_softmax(logit_p))
# for numerical stability (when what you really want is P).

# I think the slice is `prefix_len-1:-1` because the logits at the nth position
# represent predictions about the n+1th token (so you want to start looking at
# the position for the last token in the prefix, for predictions about what
# comes after the prefix).

def calculate_kl_penalty(logits, ref_logits, kl_coef, prefix_len):
    # minibatch, seq_len, model_dimensionality
    logit_p = logits[:, prefix_len-1:-1]
    logit_q = ref_logits[:, prefix_len-1:-1]

    log_p = torch.log_softmax(logit_p, dim=2)
    log_q = torch.log_softmax(logit_q, dim=2)

    p = torch.exp(log_p)

    return kl_coef * (p * (log_p - log_q)).sum(dim=2).mean()

def calculate_entropy_bonus(logits, ent_coef, prefix_len):
    log_p = torch.log_softmax(logits[:, prefix_len-1:-1], dim=2)
    p = torch.exp(log_p)
    return -ent_coef * (p * log_p).sum(dim=2).mean()


calculate_value_function_loss = solutions.calc_value_function_loss
calculate_clipped_surrogate_objective = solutions.calc_clipped_surrogate_objective

# Is it just me, or is it weird that we're still using the PPO clipped
# surrogate objective in conjunction with the KL penalty here? (Because the
# clipped surrogate is already supposed to be an easier-to-compute alternative
# to a KL penalty in TRPO.)


def get_logprobs(logits, tokens, prefix_len=1):
    generated_tokens = tokens[:, prefix_len:]
    logit_distribution = logits[:]
    # TODO: finish
