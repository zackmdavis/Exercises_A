import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch import Tensor

import gym
import wandb
import numpy as np

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter2_rl/exercises/")
section_dir = exercises_dir / "part2_q_learning_and_dqn"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import line, cliffwalk_imshow, plot_cartpole_obs_and_dones

import part2_q_learning_and_dqn.utils as utils
import part2_q_learning_and_dqn.tests as tests
import part2_q_learning_and_dqn.solutions as solutions

from rl_intro import find_optimal_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    kwargs={"env": Toy()},
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
    return (
        experience.obs,
        experience.act,
        experience.reward,
        experience.new_obs,
        experience.new_act,
    )


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
        self.Q[s, a] += self.config.lr * (
            r + self.gamma * max(self.Q[s_]) - self.Q[s, a]
        )


class SARSA(EpsilonGreedy):
    def observe(self, experience):
        s, a, r, s_, a_ = unpack(experience)
        self.Q[s, a] += self.config.lr * (
            r + self.gamma * self.Q[s_, a_] - self.Q[s, a]
        )

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
    def __init__(self, observation_dimensionality, num_actions, hidden_sizes=[120, 84]):
        super().__init__()
        self.observation_dimensionality = observation_dimensionality
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(self.observation_dimensionality, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], self.num_actions),
        )

    def forward(self, x):
        return self.layers(x)


# In order to make our experiences "more i.i.d.", we record stuff that happened
# into a buffer, and sample from it.


@dataclass
class ReplayBufferSamples:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    next_observations: Tensor

    def __post_init__(self):
        for experience in self.__dict__.values():
            assert isinstance(
                experience, Tensor
            ), "Error: expected type tensor, found {}".format(type(experience))


class ReplayBuffer:
    def __init__(self, num_environments, obs_shape, action_shape, buffer_size, seed):
        assert (
            num_environments == 1
        ), "This buffer only supports SyncVectorEnv with 1 environment inside."
        self.num_environments = num_environments
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)

        self.observations = np.empty((0, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty(0, dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.float32)
        self.dones = np.empty(0, dtype=bool)
        self.next_observations = np.empty((0, *self.obs_shape), dtype=np.float32)

    # Something about the way the instructions are written is confusing me
    # about what "they" want me to do here. (First dimension of 0??) Is it a
    # concatenation?

    def add(self, observations, actions, rewards, dones, next_observations):
        assert observations.shape == (self.num_environments, *self.obs_shape)
        assert actions.shape == (self.num_environments, *self.action_shape)
        assert rewards.shape == (self.num_environments,)
        assert dones.shape == (self.num_environments,)
        assert next_observations.shape == (self.num_environments, *self.obs_shape)

        if self.observations.shape[0] + 1 > self.buffer_size:
            self.observations = self.observations[1:, ...]
        if self.actions.shape[0] + 1 > self.buffer_size:
            self.actions = self.actions[1:, ...]
        if self.rewards.shape[0] + 1 > self.buffer_size:
            self.rewards = self.rewards[1:, ...]
        if self.dones.shape[0] + 1 > self.buffer_size:
            self.dones = self.dones[1:, ...]
        if self.next_observations.shape[0] + 1 > self.buffer_size:
            self.next_observations = self.next_observations[1:, ...]

        self.observations = np.concatenate((self.observations, observations))
        self.actions = np.concatenate((self.actions, actions))
        self.rewards = np.concatenate((self.rewards, rewards))
        self.dones = np.concatenate((self.dones, dones))
        self.next_observations = np.concatenate(
            (self.next_observations, next_observations)
        )

    def sample(self, sample_size, device):
        indices = [
            self.rng.integers(low=0, high=self.observations.shape[0])
            for _ in range(sample_size)
        ]
        return ReplayBufferSamples(
            observations=torch.tensor(self.observations)[indices],
            actions=torch.tensor(self.actions)[indices],
            rewards=torch.tensor(self.rewards)[indices],
            dones=torch.tensor(self.dones)[indices],
            next_observations=torch.tensor(self.next_observations)[indices],
        )

# Um, `test_replay_buffer_single` and `test_replay_buffer_wraparound` are
# passing, but `test_replay_buffer_deterministic` is failing. Oh—the tests are
# trying to ensure determinism by setting the RNG seed, which fails because I
# stubbornly used `random.randint` instead of the setting `self.rng` to signal
# my independence from just obeying what they teacher said to do. I think we're
# OK to proceed.
#
# Actually, `test_epsilon_greedy_policy` is also going to be using custom
# seeds, and I want that test coverage, so I'll fix it to use the seed now.

def linear_schedule(current_step, start_e, end_e, exploration_fraction, total_timesteps):
    # tests expect boring Latin kwargs
    start_ε = start_e
    end_ε = end_e
    age_of_exploration_end = exploration_fraction * total_timesteps
    if current_step > age_of_exploration_end:
        return end_ε
    else:
        return start_ε - (current_step/age_of_exploration_end) * (start_ε - end_ε)


def epsilon_greedy_policy(envs, q_network, rng, obs, epsilon):
    observation = obs
    ε = epsilon

    device = next(q_network.parameters()).device
    observation = torch.from_numpy(observation).to(device)

    flip = rng.random()
    if flip < ε:
        return np.array(rng.integers(0, envs.single_action_space.n, envs.num_envs))
    else:
        q_logits = q_network(observation)
        return q_logits.argmax(dim=1).cpu().numpy()  # thx GPT-4 for `argmax` syntax help

# RL is so famously hard to debug that we get a strenuous recommendation to
# learn to use simple "probe" environments.

# That's because the next step is training our agent. If that goes poorly,
# figuring out why is going to be painful.

# Our loss function is the expected squared temporal difference error.

# I think "envs" is a batch dimension? That's the only way I can make sense of
# earlier code with `rewards, dones, infos`, &c.

DQNArgs = solutions.DQNArgs

class DQNAgent:
    def __init__(self, envs, args, rb, q_network, target_network, rng):
        self.envs = envs
        self.args = args
        self.replay_buffer = rb
        self.next_observations = self.envs.reset()
        self.step = 0
        # tests make assertions about `agent.epsilon`
        self.ε = self.epsilon = args.start_e
        self.q_network = q_network
        self.target_network = target_network
        self.rng = rng

    def play_step(self):
        observations = self.next_observations
        actions = self.get_actions(observations)
        next_observations, rewards, dones, infos = self.envs.step(actions)

        premonitions = next_observations.copy()
        for environment_no, done in enumerate(dones):
            if done:
                premonitions[environment_no] = infos[environment_no]['terminal_observation']

        self.replay_buffer.add(observations, actions, rewards, dones, premonitions)

        self.next_observations = next_observations
        self.step += 1
        return infos

    def get_actions(self, observations):
        self.ε = self.epsilon = linear_schedule(self.step, self.args.start_e, self.args.end_e, self.args.exploration_fraction, self.args.total_timesteps)
        return epsilon_greedy_policy(self.envs, self.q_network, self.rng, observations, self.ε)


# The target network is a lagged copy of the main Q-network, for stability.


class DQNTrainer:
    def __init__(self, args):  # course-provided
        self.args = args
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, self.run_name)])
        self.start_time = time.time()
        self.rng = np.random.default_rng(args.seed)

	# Get obs & action shapes (we assume we're dealing with a single discrete action)
        num_actions = self.envs.single_action_space.n
        action_shape = ()
        obs_shape = self.envs.single_observation_space.shape
        num_observations = np.array(obs_shape, dtype=int).prod()

        self.q_network = QNetwork(num_observations, num_actions).to(device)
        self.target_network = QNetwork(num_observations, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = t.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

        self.replay_buffer = ReplayBuffer(len(self.envs.envs), obs_shape, action_shape, args.buffer_size, args.seed)
        self.agent = DQNAgent(self.envs, self.args, self.replay_buffer, self.q_network, self.target_network, self.rng)

    def add_to_replay_buffer(self, n):
        for _ in range(n):
            self.agent.play_step()


    def training_step(self):
        sample = self.replay_buffer.sample(1, device)
        # What are we doing here? The text says we want max_a Q(s_{t+1},a), but
        # the QNetwork seems to map observation to action dimensionalities (in
        # contrast to the two arguments in Q(s, a)).
        #
        # I guess, given an observation, it returns Q predictions for each
        # action, and I take the max?
        self.target_network.eval()
        with torch.inference_mode():
            action_values = self.target_network(sample.next_observations)

        y = sample.rewards[0] + self.args.gamma * action_values.max()
        temp



        hope = self.target_network()

        # Getting the max of the target network for the next observations,
        # Getting the predicted Q-values from the Q-network,
        # Using these to calculate the TD loss,
        # Perform a gradient step on the TD loss,
        # If self.agent.step % args.target_network_frequency == 0 then load the weights from the Q-network into the target network,


    def train(self):  # course-provided
        if self.args.use_wandb: wandb.init(
            project=self.args.wandb_project_name,
            entity=self.args.wandb_entity,
            name=self.run_name,
            monitor_gym=self.args.capture_video
        )

        print("Adding to buffer...")
        self.add_to_replay_buffer(self.args.buffer_size)

        progress_bar = tqdm(range(self.args.total_training_steps))
        last_logged_time = time.time()

        for step in progress_bar:

            last_episode_len = self.add_to_replay_buffer(self.args.train_frequency)

            if (last_episode_len is not None) and (time.time() - last_logged_time > 1):
                progress_bar.set_description(f"Step = {self.agent.step}, Episodic return = {last_episode_len}")
                last_logged_time = time.time()

            self.training_step()

        # Environments have to be closed before wandb.finish(), or else we get annoying errors 😠
        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()
