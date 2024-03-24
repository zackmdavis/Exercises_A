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
