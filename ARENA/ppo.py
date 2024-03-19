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

@torch.inference_mode()
def compute_advantages(next_value, next_done, rewards, values, dones, gamma, gae_lambda):
    '''Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (buffer_size, env)
    values: shape (buffer_size, env)
    dones: shape (buffer_size, env)
    Return: shape (buffer_size, env)
    '''
    ...
