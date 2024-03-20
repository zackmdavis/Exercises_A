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
    observations = self.next_obs
    dones = self.next_done

    # get a distribution over actions
    action_logits = self.actor(observations)
    distribution = torch.distributions.categorical.Categorical(action_logits)
    action = distribution.sample()

    # TODO: continue/finish

    # The `next_` vs. not asymmetry makes me nervous that I've misunderstood something
    next_observations, rewards, next_dones, infos = self.envs.step()

    # calculate logprobs and values

    self.memory.add(observations, actions, logprobs, values, rewards, dones)

    self.next_obs = torch.from_numpy(next_observations).to(device, dtype=t.float)
    self.next_done = torch.from_numpy(next_dones).to(device, dtype=t.float)
    self.step += self.envs.num_envs

    return infos


PPOAgent = solutions.PPOAgent
PPOAgent.play_step = play_step

tests.test_ppo_agent(PPOAgent)
