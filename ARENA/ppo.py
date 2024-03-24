import itertools
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch import Tensor
from tqdm import tqdm

import gym
import wandb
import numpy as np
from gym.envs.classic_control.cartpole import CartPoleEnv

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter2_rl/exercises/")
section_dir = exercises_dir / "part3_ppo"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part2_q_learning_and_dqn.utils import set_global_seeds
from part2_q_learning_and_dqn.solutions import Probe1, Probe2, Probe3, Probe4, Probe5
from part3_ppo.utils import make_env
import part3_ppo.utils as utils
import part3_ppo.tests as tests
import part3_ppo.solutions as solutions
from plotly_utils import plot_cartpole_obs_and_dones

for idx, probe in enumerate([Probe1, Probe2, Probe3, Probe4, Probe5]):
    gym.envs.registration.register(id=f"Probe{idx+1}-v0", entry_point=probe)

import warnings

# indefinitely repeated Gym deprecation warnings are super-annoying
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PPOArgs = solutions.PPOArgs

layer_init = solutions.layer_init
get_actor_and_critic = solutions.get_actor_and_critic


class Actor(nn.Module):
    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions),
        )
        layer_init(self.layers[0])
        layer_init(self.layers[2])
        layer_init(self.layers[4], std=0.01)

    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    def __init__(self, num_observations):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        layer_init(self.layers[0])
        layer_init(self.layers[2])
        layer_init(self.layers[4], std=0.01)

    def forward(self, x):
        return self.layers(x)


def get_actor_and_critic_classic(num_observations, num_actions):
    return Actor(num_observations, num_actions), Critic(num_observations)


# We are asked: "what do you think is the benefit of using a small standard
# deviation for the last actor layer?"
#
# I reply: um, the obvious guess is that you don't want too much variance in
# your choice of action? But that seems like a shallow, fake answer without a
# better sense of how the variance in the parameters in a single layer relate
# to the variance of the output ... but actually, for the last layer in
# particular, there is going to be more of a dependence.
#
# Instructor's answer specifically says the risk is getting locked into a
# deterministic policy and not being able to update away, and links to
# Andrychowicz et al. 2021 claiming that this is an important architectural
# deet.

# I find the exposition of Generalized Advantage Estimation deeply confusing
# and painful!! I'm trying to have Claude explain it to me ...

compute_advantages = solutions.compute_advantages

# I'm going to proceed, because spinning my wheels forever in the Generalized
# Advantage Estimation mud isn't working. Next, we're building a replay memory,
# which is not quite the same as the DQN replay buffer.

# During rollout, we'll fill the buffer. During learning, we're going to learn
# from the whole buffer (but shuffled). In addition to storing state, action,
# and done, we'll be storing logprobs (action logits), advantages, and returns.


def minibatch_indexes(rng, batch_size, minibatch_size):
    assert batch_size % minibatch_size == 0
    indices = list(range(batch_size))
    random.shuffle(indices)
    return [
        np.array(indices[i * minibatch_size : (i + 1) * minibatch_size])
        for i in range(batch_size // minibatch_size)
    ]


ReplayMinibatch = solutions.ReplayMinibatch
ReplayMemory = solutions.ReplayMemory


def play_step(self_):
    observations = self_.next_obs
    dones = self_.next_done

    # get a distribution over actions
    values = self_.critic(observations).squeeze(-1)
    action_logits = self_.actor(observations)
    action_probabilities = torch.softmax(action_logits, dim=-1)
    distribution = torch.distributions.categorical.Categorical(action_probabilities)
    actions = distribution.sample()
    logprobs = distribution.log_prob(actions)

    # The `next_` vs. not asymmetry makes me nervous that I've misunderstood
    # something—but Claude's commentary seemed reässuring
    next_observations, rewards, next_dones, infos = self_.envs.step(
        actions.cpu().numpy()
    )

    self_.memory.add(observations, actions, logprobs, values, rewards, dones)

    self_.next_obs = torch.from_numpy(next_observations).to(device, dtype=torch.float)
    self_.next_done = torch.from_numpy(next_dones).to(device, dtype=torch.float)
    self_.step += self_.envs.num_envs

    return infos


PPOAgent = solutions.PPOAgent
PPOAgent.play_step = play_step

# That only specifies how our agent acts (taking the actor and critic nets as a
# given) and stores memories. We need more code to learn from those memories.

# Our PPO objective function will have three terms. First, the "clipped
# surrogate objective", eq. 7 in the paper.

# The min and clip parts look straightforward, but I'm not sure how to
# calculate r_t(θ) or A_t from the supplied arguments!

# r_t(θ) is the ratio of π_θ(a_t | s_t) for the new and old policies. I guess
# in terms of the arguments given, that would be `probs`/`mb_logprobs`? But
# `probs` has the wrong shape: it has a `num_actions` dimension. So maybe we
# need to index into it with the actual actions: `probs[mb_action]`?

# But then there's another argument that the docstring says to add to the
# standard deviation of `mb_advantages` to prevent division-by-zero when
# normalizing ... but I'm not seeing where the advantages would get normalized?

# Also, `probs[mb_action]` isn't legal (`Categorical` object not subscriptable)
# ... but the underlying logits are a 3×4 tensor.

# probs.logits[mb_action] is also 3×4 ... talking to Claude, it may be that I
# want `gather`?

# Now `clip` is failing because my `policy_ratio` is a tensor, not a float (the
# way you would expect a ratio to be), and therefore cannot be compared to a bound.


def calculate_clipped_surrogate_objective(
    probs, mb_action, mb_advantages, mb_logprobs, clip_coef, eps=1e-8
):
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape
    # new_policy_logits = probs.logits.gather(1, mb_action.unsqueeze(0))
    # corrected?—
    d = torch.distributions.categorical.Categorical(logits=probs)
    new_policy_logits = d.log_prob(mb_action)
    # To get the ratio π_θ(a_t | s_t) / π_θ_old(a_t | s_t), we subtract the
    # logits and exponentiate (to turn the difference into a quotient)
    policy_ratio = torch.exp(new_policy_logits - mb_logprobs)

    # The notebook mentions in passing that we should normalize as in #7 of
    # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    unclipped = policy_ratio * advantages
    # thanks to Claude for suggesting `.clamp` (I had written the obvious
    # `clip` function, which doesn't work with tensors)
    clipped = policy_ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantages

    # corrected: `minimum`, not `min`
    surrogate_objective = torch.minimum(unclipped, clipped)

    # average over the minibatch
    return surrogate_objective.mean()


# After all that struggle to finally get code that makes sense ... it fails the
# tests numerically. I think I've earned the right to peek at the instructor's
# solution.

# Instructor's version—
#     logits_diff = probs.log_prob(mb_action) - mb_logprobs
#     r_theta = t.exp(logits_diff)
#     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)
#     non_clipped = r_theta * mb_advantages
#     clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages
#     return t.minimum(non_clipped, clipped).mean()

# which looks pretty sim—oh. It turns out that `min` and `minimum` are not
# aliases (as contrasted to `clip` and `clamp`) and the difference is critical.

# But—wait. Under what circumstances are `torch.min(a, b)` and
# `torch.minimum(a, b)` going to be different, exactly? And now the test is
# failing again, even though I thought I saw it working with minimum?? Was the
# problem actually with `gather`?

# ... I'm going to move on. The clipped surrogate objective helps improve our
# actor. A separate value term will help improve our critic.


def calculate_value_function_loss(values, mb_returns, vf_coef):
    assert values.shape == mb_returns.shape, "{} ≠ {}".format(values.shape, mb_returns.shape)
    return (vf_coef * (values - mb_returns) ** 2).mean()


# ... that was much easier.

# There's also an entropy bonus term, which incentivizes exploration.

# We are asked: "in CartPole, what are the minimum and maximum values that
# entropy can take? What behaviors correspond to each of these cases?"
#
# I reply: max entropy is pressing left and right with equal probability. (A
# good policy would approach this because you're just as likely to be
# unbalanced one way as the other, but a bad policy could just as well, by
# randomizing rather than being sensitive to the pole.) Minimum entropy is
# all-left or all-right, which will quickly fail.


def calculate_entropy_bonus(probs, ent_coef):
    return ent_coef * probs.entropy().mean()


# Adam is already adaptive (ADAptive Momentum!), but we're separately going to
# decay the learning rate.


class PPOScheduler:
    def __init__(self, optimizer, initial_lr, end_lr, total_training_steps):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_training_steps = total_training_steps
        self.n_step_calls = 0

    def step(self):
        self.n_step_calls += 1
        decrement = (self.initial_lr - self.end_lr) / self.total_training_steps
        for param_group in self.optimizer.param_groups:
            param_group["lr"] -= decrement


def make_optimizer(agent, total_training_steps, initial_lr, end_lr):
    optimizer = torch.optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
    return optimizer, scheduler


# And now we write the training loop's rollout and learning phases.


class PPOTrainer:
    def __init__(self, args):  # course-provided
        set_global_seeds(args.seed)
        self.args = args
        self.run_name = (
            f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        )
        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    args.seed + i,
                    i,
                    args.capture_video,
                    self.run_name,
                    args.mode,
                )
                for i in range(args.num_envs)
            ]
        )
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.optimizer, self.scheduler = make_optimizer(
            self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0
        )

    def rollout_phase(self):
        for step in range(self.args.num_steps):
            infos = self.agent.play_step()
        return infos[0].get('episode', {}).get('episode_length', {})

    def learning_phase(self):
        minibatches = self.agent.get_minibatches()
        for minibatch in minibatches:
            objective = self.compute_ppo_objective(minibatch)
            objective.backward()
            # The instructor's notebook has the gradient-clipping bullet after
            # the update bullet, but that doesn't make logical sense
            nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.agent.actor.parameters(), self.agent.critic.parameters()
                ),
                0.5,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def compute_ppo_objective(self, minibatch):
        # The clipped surrogate objective requires "probs" in addition to the
        # args gotten from the minibatch—which presumably comes from the actor
        # net that we're training—the minibatch data was kept in order to hit
        # the net with a gradient
        probs = self.agent.actor(minibatch.observations)
        clipped_surrogate_objective = calculate_clipped_surrogate_objective(
            probs,
            minibatch.actions,
            minibatch.advantages,
            minibatch.logprobs,
            clip_coef=0.2,
        )

        values = self.agent.critic(minibatch.observations).squeeze(1)
        # Values come from the critic, but what are `returns`? Is that the same
        # thing as `rewards`? No, the notebook says it's `advantages +
        # values`—no, the minibatch does have precomputed `returns` defined on
        # it.
        value_function_loss = calculate_value_function_loss(
            values, minibatch.returns, vf_coef=1
        )
        # XXX SUSPICIOUS: if we're converting a tensor to a distribution here,
        # doesn't it seem likely that we should also do so for the surrogate
        # objective?
        d = torch.distributions.categorical.Categorical(logits=probs)
        entropy_bonus = calculate_entropy_bonus(d, ent_coef=0.01)
        # Negative loss because we're doing gradient ascent (maximizing
        # negative loss is the same as minimizing loss)
        return clipped_surrogate_objective - value_function_loss + entropy_bonus

    def train(self):  # course-provided
        if args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.capture_video,
            )

        progress_bar = tqdm(range(self.args.total_phases))

        for epoch in progress_bar:
            last_episode_len = self.rollout_phase()
            if last_episode_len is not None:
                progress_bar.set_description(
                    f"Epoch {epoch:02}, Episode length: {last_episode_len}"
                )

            self.learning_phase()

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()


# I'm skeptical I got that right on the first try—let's look at our probe environments.

test_probe = solutions.test_probe

# All five probe environments pass!!  But trying to train for real immediately
# fails on a missing `range` in `rollout_phase`, which makes me question
# exactly what the probe environments were testing? (But it calls `train` which
# should call `rollout_phase` ...)
#
# There are several other "dumb" bugs like that (empty info dicts, `.log_prob`
# is a method on `Categorical`, not `Tensor`, I accepted the course's `coef`
# variable spelling and then spelled it `coeff` later because that's how I
# would have written it, torch.Size([128, 1]) ≠ torch.Size([128]), &c. ), which
# bodes ill for this endeavor

# And different parts of the code disagree about whether to expect a
# categorical distribution or a tensor ... 😰

# It's training now, but I'm less confident about it training correctly.

def cartpole_demo(policy=None, task="balance"):
    match task:
        case 'balance':
            env_id = 'CartPole-v1'
        case 'spin':
            env_id = 'SpinCart-v0'

    env = gym.make(env_id)
    env.render_mode = 'human'

    if policy is None:
        policy = Actor(4, 2)
        policy.load_state_dict(torch.load("ppo_cartpole_{}_actor.pth".format(task)))

    policy.eval()

    observation = env.reset()

    done = False
    while not done:
        observation_tensor = torch.tensor(observation, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = policy(observation_tensor).argmax().item()

        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        env.render()

# ... actually, it works!

# There's an exercise to implement reward shaping to make the task easier to
# learn, but below that, there's the suggestion of learning a "spin" policy
# instead of a "balance" policy!

class SpinCart(CartPoleEnv):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        x, v, theta, omega = observation

        # reward angular velocity
        reward = abs(omega) - 1.7 * abs(x)
        # don't stop
        done = False

        return observation, reward, done, info

gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)

# If I just reward angular velocity, the cart goes off the screen. To make a
# pretty demo, I also need to penalize x-displacement. (And since this is just a
# cartpole policy and not a superintelligence, I do get to tweak the reward
# function and try again.)
#
# ... and the displacement penalty was too small on the third attempt.
# ... maybe we should also be taking the absolute value of angular velocity?

# Should I do the Atari exercise, or was this enough to get a taste of what RL/PPO is like?

# The human-playable play_breakout.py demo doesn't work for me. (I think an
# earlier exercise required an older version of gym that only returned four
# args from `env.step(action)`, but the Breakout implementation assumes a newer gym ...
