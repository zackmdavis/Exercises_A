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


class RandomAgent(Agent):
    def get_action(self):
        return random.randint(0, self.num_arms - 1)

    def __repr__(self):
        return "<RandomAgent>"


class RewardAveragingAgent(Agent):
        def __init__(self, num_arms, seed, ε, optimism):
        self.epsilon = epsilon
        self.optimism = optimism
        super().__init__(num_arms, seed)

    # TODO

    def get_action(self):
        pass

    def observe(self, action, reward, info):
        pass

    def reset(self, seed: int):
        pass

    def __repr__(self):
        return "<RewardAveraging ε={}, optimism={}>".format(self.epsilon, self.optimism)
