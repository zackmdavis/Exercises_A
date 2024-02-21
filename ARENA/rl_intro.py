import math
import random
import sys
from pathlib import Path

import gym
import numpy as np

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter2_rl/exercises/")
section_dir = exercises_dir / "part1_intro_to_rl"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_intro_to_rl.utils as utils
import part1_intro_to_rl.tests as tests
import part1_intro_to_rl.solutions as solutions
from plotly_utils import imshow

MultiArmedBandit = solutions.MultiArmedBandit
Agent = solutions.Agent

max_episode_steps = 1000

gym.envs.registration.register(
    id="ArmedBanditTestbed-v0",
    entry_point=MultiArmedBandit,
    max_episode_steps=max_episode_steps,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={"num_arms": 10, "stationary": True},
)

env = gym.make("ArmedBanditTestbed-v0")

class RandomAgent(Agent):
    def get_action(self):
        return random.randint(0, self.num_arms - 1)

    def __repr__(self):
        return "<RandomAgent>"


class RewardAveragingAgent(Agent):
    def __init__(self, num_arms, seed, ε, optimism):
        self.ε = ε
        self.optimism = optimism
        super().__init__(num_arms, seed)

    def get_action(self):
        if random.random() < self.ε:
            return random.randint(0, self.num_arms - 1)
        else:
            # neat argmax trick suggested by GPT-4
            return max(range(self.num_arms), key=self.quality.__getitem__)

    def observe(self, action, reward, info):
        self.plays[action] += 1
        self.quality[action] += (1/self.plays[action]) * (reward - self.quality[action])

    def reset(self, seed):
        super().reset(seed)
        self.plays = [0 for _ in range(num_arms)]
        self.quality = [self.optimism for _ in range(num_arms)]

    def __repr__(self):
        return "<RewardAveraging ε={}, optimism={}>".format(self.ε, self.optimism)


class CheatingAgent(Agent):
    def __init__(self, num_arms, seed):
        super().__init__(num_arms, seed)
        self.best_arm = 0

    def get_action(self):
        return self.best_arm

    def observe(self, action, reward, info):
        self.best_arm = info['best_arm']

    def __repr__(self):
        return "<Cheater>"


class UpperConfidenceBoundActionSelectionAgent(Agent):
    def __init__(self, num_arms, seed, c, ε=1e-6):
        self.c = c
        self.ε = ε
        self.t = 0
        super().__init__(num_arms, seed)

    def get_action(self):
        self.t += 1

        # In order to avoid division by zero, anything we've never tried takes priority
        if any(p == 0 for p in self.plays):
            return min(range(self.num_arms), key=self.plays.__getitem__)
        # Instructor's solution used the ε from the args list in the denominator

        actions = [q + self.c * math.sqrt(math.log(self.t)/self.plays[i]) for i, q in enumerate(self.quality)]
        return max(range(self.num_arms), key=actions.__getitem__)

    def observe(self, action, reward, info):
        self.plays[action] += 1
        self.quality[action] += (1/self.plays[action]) * (reward - self.quality[action])

    def reset(self, seed: int):
        super().reset(seed)
        self.plays = [0 for _ in range(num_arms)]
        self.quality = [0 for _ in range(num_arms)]

    def __repr__(self):
        return "<UpperConfidenceBound(c={})>".format(self.c)


Environment = solutions.Environment
Toy = solutions.Toy
Norvig = solutions.Norvig

# Now we're going to solve the Bellman equation numerically.
# V_π(s) is the value of following the policy π from state s.

def numerical_policy_evaluation(env, π, γ=0.99, ε=1e-8, max_iterations=10_000):
    previous_values = torch.tensor([0] * π.shape[0])
    values = torch.tensor([0] * π.shape[0])
    for i in range(max_iterations):
        # The value of the policy for state s, V(s), is the sum over next
        # states s′, Σ, of T(s′ | s, a) (R(s, a, s′) + γV(s′))
        for s in range(len(π)):
            for s_ in range(len(π)):
                values[s] = env.T[s, s_, a] * (env.R[s, a, s_] + γ * previous_values[s_])
        previous_values = values
