import math
import random
import sys
from pathlib import Path

import torch

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
        self.quality[action] += (1 / self.plays[action]) * (
            reward - self.quality[action]
        )

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

        actions = [
            q + self.c * math.sqrt(math.log(self.t) / self.plays[i])
            for i, q in enumerate(self.quality)
        ]
        return max(range(self.num_arms), key=actions.__getitem__)

    def observe(self, action, reward, info):
        self.plays[action] += 1
        self.quality[action] += (1 / self.plays[action]) * (
            reward - self.quality[action]
        )

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


def numerical_policy_evaluation(env, pi, gamma=0.99, eps=1e-8, max_iterations=10_000):
    # instructor's test case uses their kwarg names
    π = pi
    γ = gamma
    ε = eps

    previous_values = torch.tensor([0.0] * π.shape[0], dtype=torch.float64)
    values = torch.tensor([0.0] * π.shape[0], dtype=torch.float64)
    for i in range(max_iterations):
        # The value of the policy for state s, V(s), is the sum over next
        # states s′, Σ, of T(s′ | s, a) (R(s, a, s′) + γV(s′))
        for s in range(len(π)):
            a = π[s]
            sum_over_following = 0
            for s_ in range(len(π)):
                sum_over_following += env.T[s, a, s_] * (
                    env.R[s, a, s_] + γ * previous_values[s_]
                )
            values[s] = sum_over_following

        if (values - previous_values).abs().max().item() < ε:
            break

        previous_values = values.clone()

    return values


# I was distressed that my tests were failing, but GPT-4 pointed out a dumb
# bug—that I need to `clone()` when assigning to `previous_values`—and the
# tests pass.

# Instructor's solution goes like—
#
# transition_matrix = env.T[states, actions, :]
# reward_matrix = env.R[states, actions, :]
# V = np.zeros_like(pi)
# for i in range(max_iterations):
#     V_new = einops.einsum(transition_matrix, reward_matrix + gamma * V, "s s_prime, s s_prime -> s")
#     if np.abs(V - V_new).max() < eps:
#         print(f"Converged in {i} steps.")
#         return V_new
#     V = V_new
#
# If I continue to find `einsum` inscrutable, that's a clue that I'm not cut
# out for mechanistic interpretability!!


# Next, Policy Evaluation (exact): rather than iteratively updating an
# estimated policy value function, you can set up a system of equations to
# solve it exactly.


def exact_policy_evaluation(env, pi, gamma=0.99):
    π = pi
    γ = gamma

    n = len(π)

    # # Once again, I natively think in `for`-loops, not tensor-manipulations; I
    # # can only pray that someday it'll click (the way that Rust eventually
    # # clicked)
    # P = torch.empty(n, n, dtype=torch.float64)
    # R = torch.empty(n, n, dtype=torch.float64)

    # for i in range(n):
    #     for j in range(n):
    #         # state → next-state transition matrix
    #         P[i][j] = env.T[j, π[i], i]
    #         # state → next-state reward matrix (actions chosen from π)
    #         R[i][j] = env.R[i, π[i], j]

    # I'm not sure what was wrong with my version, which seemed to follow the
    # exposition in the notebook, but the instructor's version is
    states = np.arange(env.num_states)
    actions = pi
    P = env.T[states, actions, :]
    R = env.R[states, actions, :]

    # The notation in the Colab notebook is unclear to me—what shape is 𝐫?
    # Instructor's solution gives `r = einops.einsum(transition_matrix, reward_matrix, "i j, i j -> i")`
    r = np.diag(P @ R.T)

    return (torch.eye(n) - γ * P).inverse() @ r


# Given a value function, we can compute an equal-or better policy. (A form of
# hill climbing, not unlike gradient-descent?) We don't even need the current
# policy; the computation only relies on V_π


def policy_improvement(env, V, gamma=0.99):
    γ = gamma

    π = np.array([0] * len(V))
    for state in range(len(V)):
        best_action = None
        best_action_value = None
        for action in range(env.T.shape[1]):
            sum_over_following = 0
            for next_state in range(len(V)):
                sum_over_following += env.T[state, action, next_state] * (
                    env.R[state, action, next_state] + γ * V[next_state]
                )
            if best_action_value is None or sum_over_following > best_action_value:
                best_action = action
                best_action_value = sum_over_following
        π[state] = best_action
    return π


def find_optimal_policy(env, gamma=0.99, max_iterations=10_000):
    γ = gamma

    π = np.zeros(env.num_states, dtype=int)
    for _ in range(max_iterations):
        V = exact_policy_evaluation(env, π)
        old_π = π
        π = policy_improvement(env, V)
        if np.array_equal(old_π, π):
            break

    return π
