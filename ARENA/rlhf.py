import sys
from pathlib import Path

import torch
from torch import nn

from tqdm import tqdm
from transformer_lens import HookedTransformer

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter2_rl/exercises/")
section_dir = exercises_dir / "part4_rlhf"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part4_rlhf.tests as tests
import part4_rlhf.solutions as solutions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            fwd_hooks=[("ln_final.hook_normalized", self.value_hook)],
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
        torch.concat((values[:, prefix_len : seq_len - 1], rewards.unsqueeze(1)), dim=1)
        - values[:, prefix_len - 1 : seq_len - 1]
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
    logit_p = logits[:, prefix_len - 1 : -1]
    logit_q = ref_logits[:, prefix_len - 1 : -1]

    log_p = torch.log_softmax(logit_p, dim=2)
    log_q = torch.log_softmax(logit_q, dim=2)

    p = torch.exp(log_p)

    return kl_coef * (p * (log_p - log_q)).sum(dim=2).mean()


def calculate_entropy_bonus(logits, ent_coef, prefix_len):
    log_p = torch.log_softmax(logits[:, prefix_len - 1 : -1], dim=2)
    p = torch.exp(log_p)
    return -ent_coef * (p * log_p).sum(dim=2).mean()


calculate_value_function_loss = solutions.calc_value_function_loss
calculate_clipped_surrogate_objective = solutions.calc_clipped_surrogate_objective

# Is it just me, or is it weird that we're still using the PPO clipped
# surrogate objective in conjunction with the KL penalty here? (Because the
# clipped surrogate is already supposed to be an easier-to-compute alternative
# to a KL penalty in TRPO.)

# I still have this awful mental block where I just can't wrap my head around
# tensor manipulations—even when I can describe the thing I want in English, I
# don't know how to write it as PyTorch advanced indexes (let alone einsum),
# even though I could do it as a list comprehension (because I have that
# drilled and locked)
#
# What happens if I just write it as a list comprehension (or for loop) in
# order to get unblocked?
#
# While looking at the test (not the solution), some info about the solution
# leaked anyway because the solution was inline rather than imported—I was
# treating logits as the answer, when it needs to be log-softmaxed

# TODO (sometime?): use this as a case study for how to do index manipulations
# correctly


def get_logprobs(logits, tokens, prefix_len=1):
    # The tests actually set `prexfix_len` to `None`!?
    if prefix_len is None:
        prefix_len = 1

    batch_size, seq_len = tokens.shape
    gen_len = seq_len - prefix_len

    generated_tokens = tokens[:, prefix_len:]
    log_distribution = torch.log_softmax(logits, dim=-1)

    # import pudb; pudb.set_trace()
    result = [[None for j in range(gen_len)] for i in range(batch_size)]
    for i in range(batch_size):
        for j in range(gen_len):
            result[i][j] = log_distribution[i][prefix_len - 1 + j][
                generated_tokens[i][j]
            ]

    return torch.tensor(result)


# On to the optimizer/scheduler. The value head and base model get different learning rates.


def get_optimizer(args, model):
    parameter_groups = [
        {"params": model.base_model.parameters(), "lr": args.base_learning_rate},
        {"params": model.value_head.parameters(), "lr": args.head_learning_rate},
    ]
    return torch.optim.Adam(parameter_groups, maximize=True)


get_lr_scheduler = solutions.get_lr_scheduler
get_optimizer_and_scheduler = solutions.get_optimizer_and_scheduler

reward_fn_char_count = solutions.reward_fn_char_count

# In addition to tapering down the learning rate later, there's also going to
# be a "warmup" period to account for our initial gradients and adaptive
# moments being large or unstable.

# Now we write the trainer class ...

# In the PPO implementation, we invoked the actor—and the critic—inside of
# `compute_ppo_objective`. The analogue here would be generating completions
# inside of `compute_rlhf_objective`. (The model-with-value-head will return
# both a completion and a value.) For some reason, this feels like a
# counterintuitive place to do it (though I don't remember having this
# objection to the PPO exercise).

# But the minibatch itself already has `logprobs` and `ref_logits`. "Ref" makes
# sense, because we want to know what the reference policy did. But what are
# `logprobs` in the minibatch from, then?

# And actually, the docstring for the rollout phase talks about getting "logits
# of those generated samples (from model & reference model)". This is making me
# confused about what's supposed to happen where!

# I think `logprobs` in this ReplayMemory is supposed to be analogous to the
# one in the PPO implementation (in contrast to `ref_logits` being a new
# addition)—which implies that `logprobs` should be those of the actions (new
# policy).

# I think there's something I'm not getting about the policy being used during
# both rollout and learning? In the PPO implementation, the `logprobs` that
# went into replay memory were the log-probabilities of actions taken (which
# were determined by sampling from the actor network).


class RLHFTrainer:
    def __init__(self, args):  # course-provided
        torch.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.exp_name}__{args.seed}"
        self.model = TransformerWithValueHead(args.base_model).to(device).train()
        self.ref_model = TransformerWithValueHead(args.base_model).to(device).eval()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            self.args, self.model
        )
        self.prefix_len = len(
            self.model.base_model.to_str_tokens(self.args.prefix, prepend_bos=False)
        )

    def compute_rlhf_objective(self, minibatch):
        self.model.train()
        prefix_len = minibatch.sample_ids.shape[1] - minibatch.advantages.shape[1]
        logits, values = self.model(self.model.base_model.tokenizer.batch_decode(minibatch.sample_ids))
        logprobs = get_logprobs(logits, sample_ids, prefix_len=prefix_len)

        clipped_surrogate_objective = calculate_clipped_surrogate_objective(
            logprobs, minibatch.logprobs, minibatch.advantages, self.args.clip_coef
        )
        value_function_loss = calculate_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calculate_entropy_bonus(logits, self.args.ent_coef, prefix_len)
        kl_penalty = calculate_kl_penalty(logits, minibatch.ref_logits, self.args.kl_coef, prefix_len)
        return (
            clipped_surrogate_objective
            - value_function_loss
            + entropy_bonus
            - kl_penalty
        )

    def rollout_phase(self):
        sample_ids, samples = get_samples(
            self.model.base_model,
            prompt=self.args.prefix,
            batch_size=self.args.batch_size,
            gen_len=self.args.gen_len,
            temperature=self.args.temperature,
        )
        self.model.eval()
        logits, values = self.model(samples)
        ref_logits = self.ref_model.base_model(samples)

        _batch_size, seq_len = sample_ids.shape
        prefix_len = seq_len - self.args.gen_len

        logprobs = get_logprobs(logits, sample_ids, prefix_len=prefix_len)

        # We need `rewards` to compute advantages, but where do we get them
        # from? I guess this is where we call the reward function on the
        # generated samples?—and normalize, per the docstring? And then
        # presumably the rewards will get distilled into the value estimates
        rewards = normalize_reward(reward_fn_char_count(samples))

        advantages = compute_advantages(values, rewards, prefix_len)

        return ReplayMemory(
            self.args,
            sample_ids,
            logprobs,
            advantages,
            values,
            ref_logits,
        )

    def learning_phase(self, memory):
        for minibatch in memory.get_minibatches():
            objective = self.compute_rlhf_objective(minibatch)
            objective.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()
        self.step += 1

    def train(self):  # course-provided
        self.step = 0

        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                config=self.args,
            )

        for self.phase in tqdm(range(self.args.total_phases)):
            memory = self.rollout_phase()
            self.learning_phase(memory)

        if self.args.use_wandb:
            wandb.finish()

# I may not have the hardware for this?

# `OutOfMemoryError: CUDA out of memory. Tried to allocate 102.00 MiB. GPU 0
# has a total capacity of 3.80 GiB of which 93.12 MiB is free. Including
# non-PyTorch memory, this process has 3.70 GiB memory in use. Of the allocated
# memory 3.55 GiB is allocated by PyTorch, and 62.26 MiB is reserved by PyTorch
# but unallocated. If reserved but unallocated memory is large try setting
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See
# documentation for Memory Management
# (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)`

# The instructions say, "these exercises assume you're running on an A100 [...]
# If you're running on a less powerful machine e.g. A10," whereas I have an RTX
# 3050, which is even worse.

# I think this merits peeking at the solution (if I can't trivially just run
# the code first) ...

# Actually, batch-size=4 might work?!—it's hitting an error in my code
# ("IndexError: index 29 is out of bounds for dimension 0 with size 29") before
# running out of memory

# It does pass the tests, though?! ...

# >>> gen_len
# 30
# >>> tokens.shape
# torch.Size([4, 32])
# >>> generated_tokens.shape
# torch.Size([4, 29])
# >>> prefix_len
# 3

# I think prefix_len == 3 is a lie; the actual prefix, "This is", tokenizes as
# `['This', ' is']`.

# It was calculated as `seq_len - self.args.gen_len`, where `seq_len` was the
# second (1th) dimension of logits (33) ... but presumably it should actually
# be the second (1th) dimension of `sample_ids` (32).

# But apparenlty that wasn't the entire problem, because now we get
# `IndexError: index 30 is out of bounds for dimension 0 with size 30`!!
# ... but I had a similar `seq_len` calculation inside of `get_logprobs`.

# The off-by-one propagates—
# `RuntimeError: The size of tensor a (31) must match the size of tensor b (30)
# at non-singleton dimension 1`
