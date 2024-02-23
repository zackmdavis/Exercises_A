import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch import Tensor

import gym
import numpy as np

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter2_rl/exercises/")
section_dir = exercises_dir / "part2_q_learning_and_dqn"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import line, cliffwalk_imshow

import part2_q_learning_and_dqn.utils as utils
import part2_q_learning_and_dqn.tests as tests
import part2_q_learning_and_dqn.solutions as solutions

from rl_intro import find_optimal_policy

DiscreteEnviroGym = solutions.DiscreteEnviroGym
Norvig = solutions.Norvig
Experience = solutions.Experience
AgentConfig = solutions.AgentConfig
Agent = solutions.Agent
RandomAgent = solutions.Random
Toy = solutions.Toy

gym.envs.registration.register(
    id="NorvigGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=100,
    nondeterministic=True,
    kwargs={"env": Norvig(penalty=-0.04)},
)

gym.envs.registration.register(
    id="ToyGym-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=2,
    nondeterministic=False,
    kwargs={"env": Toy()}
)

class CheatingAgent(Agent):
    def __init__(self, env, config=AgentConfig(), gamma=0.99, seed=0):
        super().__init__(env, config, gamma, seed)
        self.policy = find_optimal_policy(env.unwrapped.env, gamma=gamma)

    def get_action(self, obs):
        return self.policy[obs]


# The Q_π(s, a) function measures the expected rewards from taking action a in
# state s, given that other actions are coming from the policy π. If we know
# the environment dynamics (rewards and next-states given action–state pairs),
# we can compute this with the Bellman equation. If not, we have to estimate
# it.

# We store state–action–reward–state–action 5-tuples.

# For an optimal policy, Q*(s_t, a_t) ≈ r_{t+1} + γQ*(s_{t+1}, a_{t+1}).

# This inspires the temporal difference error,
# r_{t+1} + γQ(s_{t+1}, a_{t+1}) − Q(s_t, a_t)

# SARSA is "on-policy" learning: we estimate Q values for the policy that we're executing.
# Q-learning is "off-policy"

# Why aren't dataclasses iterable?!
def unpack(experience):
    return experience.obs, experience.act, experience.reward, experience.new_obs, experience.new_act

class EpsilonGreedy(Agent):
    def __init__(self, env, config, gamma=0.99, seed=0):
        super().__init__(env, config, gamma, seed)
        self.Q = np.zeros((self.num_states, self.num_actions)) + self.config.optimism

    def get_action(self, observation):
        if random.random() < self.config.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return max(range(self.num_actions), key=self.Q[observation].__getitem__)

class QLearning(EpsilonGreedy):
    def observe(self, experience):
        s, a, r, s_, a_ = unpack(experience)
        self.Q[s, a] += self.config.lr * (r + self.gamma * max(self.Q[s_]) - self.Q[s, a])


class SARSA(EpsilonGreedy):
    def observe(self, experience):
        s, a, r, s_, a_ = unpack(experience)
        self.Q[s, a] += self.config.lr * (r + self.gamma * self.Q[s_, a_] - self.Q[s, a])

    # An alternate `run_episode` is provided.
    # We are asked: what's different, and why?
    # I reply: instead of just tracking `action`, we track both `action` and
    # `new_action`. This is because the SARSA TD update looks ahead to the next
    # action that was taken, in contrast to how Q-learning does its update best
    # on the estimated best action.
    def run_episode(self, seed):
        rewards = []
        obs = self.env.reset(seed=seed)
        act = self.get_action(obs)
        self.reset(seed=seed)
        done = False
        while not done:
            (new_obs, reward, done, info) = self.env.step(act)
            new_act = self.get_action(new_obs)
            exp = Experience(obs, act, reward, new_obs, new_act)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
            act = new_act
        return rewards

# If our reward function was arbitrary, there's no way to do better than the
# table-driven approach. The real world is regular enough that we hope to generalize.

# Thus—deep Q-learning!!

# We are asked: why not include a ReLU at the end?
# I answer: um ... I'm not totally sure. I guess, the point of a ReLU is to be
# a nonlinearity between MLP layers. If you put it at the end, you're just
# clamping negative numbers to zero for no particular benefit?
#
# Instructor's answer confirms: an ending ReLU would be inappropriate because
# we want to be able to predict negative Q values for environment with negative
# rewards (or if we scale rewards).

# We are asked: the environment gives +1 for every timestep. Why won't the
# network just learn to predict +1?
# I answer: some states are more likely to lead to the end of the episode soon,
# and thus have a lower sum of discounted reward, even if they provide the same
# reward "now" (just before hitting the ground).


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dimensionality,
        num_actions,
        hidden_sizes=[120, 84]
    ):
        super().__init__()
        self.observation_dimensionality = observation_dimensionality
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(self.observation_dimensionality, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], self.num_actions)
        )

    def forward(self, x):
        return self.layers(x)

# In order to make our experiences "more i.i.d.", we record stuff that happened
# into a buffer, and sample from it.

@dataclass
class ReplayBufferSamples:
    observations: Tensor # shape [sample_size, *observation_shape]
    actions: Tensor # shape [sample_size, *action_shape]
    rewards: Tensor # shape [sample_size,]
    dones: Tensor # shape [sample_size,]
    next_observations: Tensor # shape [sample_size, observation_shape]

    def __post_init__(self):
        for experience in self.__dict__.values():
            assert isinstance(experience, Tensor), "Error: expected type tensor, found {}".format(type(exp))


# TODO: continue implementing replay buffer
