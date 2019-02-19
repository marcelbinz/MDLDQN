import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import torch

class BatchBandit(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_steps, num_actions, reward_var, mean_var, batch_size=32):
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.reward_var = reward_var
        self.mean_var = mean_var
        self.batch_size = batch_size

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(np.ones(2), np.ones(2))

        self.seed()
        self.reset()

    def reset(self):
        self.t = 0

        self.mean = torch.normal(torch.zeros(self.batch_size, self.num_actions), math.sqrt(self.mean_var) * torch.ones(self.batch_size, self.num_actions))

        return torch.zeros(self.batch_size, 2)

    def step(self, action):
        self.t += 1
        done = True if (self.t >= self.max_steps) else False

        regrets = 0
        reward = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            reward[i] = torch.normal(self.mean[i, action[i]], math.sqrt(self.reward_var))
            regrets += self.mean[i].max() - self.mean[i, action[i]]


        return torch.stack((reward, action.float()), dim=1), reward, done, {'regrets': regrets / self.batch_size}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class RandomBatchBandit(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, p_stop, num_actions, reward_var, mean_var, batch_size=32):
        self.num_actions = num_actions
        self.p_stop = p_stop
        self.reward_var = reward_var
        self.mean_var = mean_var
        self.batch_size = batch_size

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(np.ones(2), np.ones(2))

        self.seed()
        self.reset()

    def reset(self):
        self.t = 0

        self.max_steps = np.random.geometric(self.p_stop)

        self.mean = torch.normal(torch.zeros(self.batch_size, self.num_actions), math.sqrt(self.mean_var) * torch.ones(self.batch_size, self.num_actions))

        return torch.zeros(self.batch_size, 2)

    def step(self, action):
        self.t += 1
        done = True if (self.t >= self.max_steps) else False

        regrets = 0
        reward = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            reward[i] = torch.normal(self.mean[i, action[i]], math.sqrt(self.reward_var))
            regrets += self.mean[i].max() - self.mean[i, action[i]]


        return torch.stack((reward, action.float()), dim=1), reward, done, {'regrets': regrets / self.batch_size}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
